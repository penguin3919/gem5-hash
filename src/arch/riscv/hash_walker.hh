/*
 * Hash Page Table Walker for gem5 SE mode (RISC-V)
 *
 * Mirrors the structure of Walker in pagetable_walker.hh:
 *   - WalkerPort, WalkerState, WalkerSenderState layout is identical
 *   - Ready / Waiting states, inflight counter, sendPackets(), squash()
 *   - setupWalk() builds the first Packet, stepWalk() parses a response
 *     and builds the next one
 *
 * Every Packet reads one field that lies inside a single cache line:
 *
 *   Routing   routingTable[fp]  1 B   -> first slot of the fp chain
 *   Entry     entries[cur]     32 B   -> vaddr check, next link
 *   Floating  floatingCount     4 B   -> only after a main-chain miss
 *   OvSlot    overflow slot    32 B   -> up to 4 slots x 2 buckets
 *
 * Differences from the Sv39 Walker:
 *   - no SATP-based table root (base address comes from HashArchPageTable)
 *   - no A/D write-back (flags are fixed at map() time)
 *   - the TLB entry's pte field is synthesised from the hash flags so the
 *     unmodified TLB::checkPermissions() works on later TLB hits
 *   - no functional walk: TLB::translateFunctional() calls
 *     HashArchPageTable::physLookup() directly
 *
 * Stats (per walker, i.e. separately for the I-TLB and D-TLB):
 *   walks, reads (+ per field kind), avgReadsPerWalk, readsPerWalk
 *   distribution, overflowWalks, hits, pageFaults
 */

#ifndef __ARCH_RISCV_HASH_WALKER_HH__
#define __ARCH_RISCV_HASH_WALKER_HH__

#include <list>
#include <vector>

#include "arch/generic/mmu.hh"
#include "arch/riscv/hash_arch_page_table.hh"
#include "arch/riscv/pma_checker.hh"
#include "arch/riscv/pmp.hh"
#include "arch/riscv/tlb.hh"
#include "base/statistics.hh"
#include "base/types.hh"
#include "mem/packet.hh"
#include "params/RiscvHashWalker.hh"
#include "sim/clocked_object.hh"
#include "sim/faults.hh"
#include "sim/system.hh"

namespace gem5
{

class ThreadContext;

namespace RiscvISA
{

class HashWalker : public ClockedObject
{
  protected:
    // ── Port ──────────────────────────────────────────────────
    class WalkerPort : public RequestPort
    {
      public:
        WalkerPort(const std::string &_name, HashWalker *_walker)
            : RequestPort(_name), walker(_walker) {}

      protected:
        HashWalker *walker;
        bool recvTimingResp(PacketPtr pkt);
        void recvReqRetry();
    };

    friend class WalkerPort;
    WalkerPort port;

    // ── WalkerState ───────────────────────────────────────────
    class WalkerState
    {
        friend class HashWalker;

      private:
        enum State
        {
            Ready,
            Waiting,
            Routing,    // waiting for routingTable[fp]
            Entry,      // waiting for entries[cur]
            Floating,   // waiting for floatingCount
            OvSlot,     // waiting for overflow slot
        };

      protected:
        HashWalker             *walker;
        ThreadContext          *tc;
        RequestPtr              req;
        State                   state;
        State                   nextState;
        unsigned                inflight;
        TlbEntry                entry;
        PacketPtr               read;
        std::vector<PacketPtr>  writes;   // structural parity; always empty
        Fault                   timingFault;
        BaseMMU::Translation   *translation;
        BaseMMU::Mode           mode;
        SATP                    satp;
        STATUS                  status;
        PrivilegeMode           pmode;
        bool                    timing;
        bool                    retrying;
        bool                    started;
        bool                    squashed;

        // Hash walk position
        HashArchPageTable      *hashPT;
        Addr                    vpn;      // page-aligned vaddr
        size_t                  bucket;   // main bucket index
        uint64_t                fp;       // fingerprint
        uint8_t                 slot;     // current chain slot
        unsigned                ovPass;   // 0: overflow hash1, 1: hash2
        size_t                  ovBucket; // current overflow bucket
        unsigned                ovSlot;   // current overflow slot
        unsigned                walkReads;// reads issued by this walk

      public:
        WalkerState(HashWalker *_walker,
                    BaseMMU::Translation *_translation,
                    const RequestPtr &_req)
            : walker(_walker), req(_req),
              state(Ready), nextState(Ready),
              inflight(0),
              translation(_translation),
              timing(false), retrying(false),
              started(false), squashed(false),
              hashPT(nullptr), vpn(0), bucket(0), fp(0),
              slot(HASH_SLOT_NONE), ovPass(0), ovBucket(0), ovSlot(0),
              walkReads(0)
        {}

        void initState(ThreadContext *_tc, BaseMMU::Mode _mode,
                       bool _isTiming = false);
        Fault startWalk();
        bool  recvPacket(PacketPtr pkt);
        unsigned numInflight() const;
        bool  isRetrying();
        bool  wasStarted();
        bool  isTiming();
        void  retry();
        void  squash();
        std::string name() const { return walker->name(); }

      private:
        void  setupWalk(Addr vaddr);
        Fault stepWalk(PacketPtr &write);
        void  sendPackets();
        void  endWalk();
        Fault pageFault();

        // Queue the next single-field read and the state that parses it
        void  nextRead(Addr physAddr, size_t size, State parseState);
        // Try to finish the walk with a PTE read from memory
        bool  matchPTE(const HashPTE &pte, Fault &fault);

        PacketPtr makeReadPkt(Addr physAddr, size_t size, State kind);
    };

    friend class WalkerState;
    std::list<WalkerState *> currStates;

    struct WalkerSenderState : public Packet::SenderState
    {
        WalkerState *senderWalk;
        explicit WalkerSenderState(WalkerState *w) : senderWalk(w) {}
    };

  public:
    Fault start(ThreadContext *_tc, BaseMMU::Translation *translation,
                const RequestPtr &req, BaseMMU::Mode mode);
    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;

  protected:
    TLB          *tlb;
    System       *sys;
    PMAChecker   *pma;
    PMP          *pmp;
    RequestorID   requestorId;
    unsigned      numSquashable;
    Cycles        hashLatency;  // extra cycles before first walker packet

    struct HashWalkerStats : public statistics::Group
    {
        HashWalkerStats(statistics::Group *parent);

        statistics::Scalar walks;
        statistics::Scalar reads;
        statistics::Scalar routingReads;
        statistics::Scalar entryReads;
        statistics::Scalar floatingReads;
        statistics::Scalar ovSlotReads;
        statistics::Scalar overflowWalks;
        statistics::Scalar hits;
        statistics::Scalar pageFaults;
        statistics::Formula avgReadsPerWalk;
        statistics::Distribution readsPerWalk;
    } stats;

    void startWalkWrapper();
    EventFunctionWrapper startWalkWrapperEvent;

    // Fires after hashLatency cycles to send the first packet.
    // Used only in Timing mode when hashLatency > 0.
    void sendFirstPacket();
    EventFunctionWrapper sendFirstPacketEvent;

    bool recvTimingResp(PacketPtr pkt);
    void recvReqRetry();
    bool sendTiming(WalkerState *sendingState, PacketPtr pkt);

  public:
    void setTLB(TLB *_tlb) { tlb = _tlb; }

    using Params = RiscvHashWalkerParams;

    HashWalker(const Params &params)
        : ClockedObject(params),
          port(name() + ".port", this),
          tlb(nullptr),
          sys(params.system),
          pma(params.pma_checker),
          pmp(params.pmp),
          requestorId(sys->getRequestorId(this)),
          numSquashable(params.num_squash_per_cycle),
          hashLatency(params.hash_latency),
          stats(this),
          startWalkWrapperEvent([this]{ startWalkWrapper(); }, name()),
          sendFirstPacketEvent([this]{ sendFirstPacket(); }, name())
    {}
};

} // namespace RiscvISA
} // namespace gem5

#endif // __ARCH_RISCV_HASH_WALKER_HH__
