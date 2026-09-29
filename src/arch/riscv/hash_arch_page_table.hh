/*
 * Hash-based Architectural Page Table for gem5 SE mode (RISC-V)
 *
 * Inherits EmulationPageTable and mirrors every mapping into a hash
 * structure placed in simulated physical memory, so that HashWalker can
 * translate addresses with real (cacheable) memory accesses.
 *
 * Physical memory layout (all offsets multiples of the 64 B cache line):
 *
 *   baseAddr_
 *   +-- HashBucket[numBuckets]                1344 B each (21 lines)
 *   |     routingTable[256]  (1 B per fp)     line 0-3
 *   |     floatingCount      (4 B)            line 4
 *   |     padding                             line 4
 *   |     entries[32]        (32 B each)      line 5-20
 *   +-- OvBucket[numOvBuckets]                128 B each (2 lines)
 *         slots[4]           (32 B each)
 *
 * Every read the walker issues stays inside a single cache line:
 *   routingTable[fp] (1 B) -> entries[cur] (32 B, carries next link)
 *   -> floatingCount (4 B, main-chain miss only) -> overflow slot (32 B)
 *
 * numBuckets / numOvBuckets come from Process.hashPTBuckets /
 * Process.hashPTOvBuckets. Only the per-bucket struct sizes are fixed at
 * compile time; the bucket counts only enter the address arithmetic.
 *
 * Host memory only (never read by the walker):
 *   tabulation hash tables, fingerprint constant, vacancy map,
 *   overflow bucket occupancy counts, statistics.
 */

#ifndef __ARCH_RISCV_HASH_ARCH_PAGE_TABLE_HH__
#define __ARCH_RISCV_HASH_ARCH_PAGE_TABLE_HH__

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "arch/riscv/page_size.hh"
#include "base/logging.hh"
#include "base/types.hh"
#include "mem/page_table.hh"
#include "mem/port_proxy.hh"
#include "sim/se_workload.hh"
#include "sim/system.hh"

namespace gem5
{

// ============================================================
// Layout constants
// ============================================================
static constexpr size_t  HASH_CACHE_LINE    = 64;
static constexpr uint8_t HASH_SLOT_NONE     = 0xFF;

static constexpr size_t  HASH_DEFAULT_BUCKETS    = 1024;
static constexpr size_t  HASH_SLOTS         = 32;
static constexpr size_t  HASH_FP_BITS       = 8;
static constexpr size_t  HASH_NUM_FP_VALUES = 1 << HASH_FP_BITS;   // 256

static constexpr size_t  HASH_DEFAULT_OV_BUCKETS = 1024;
static constexpr size_t  OV_SLOTS_PER_BKT   = 4;

static_assert(HASH_SLOTS < HASH_SLOT_NONE,
              "slot index must fit in uint8_t with 0xFF reserved");

// ============================================================
// Structures placed in simulated physical memory
// ============================================================

// PTE: 32 B, two per cache line. The fp-chain link lives in the PTE
// so one read per chain step is enough.
struct HashPTE
{
    Addr     vaddr;       // page-aligned virtual address (collision key)
    Addr     paddr;       // page-aligned physical address
    uint64_t flags;       // HashArchPageTable::HashFlags
    uint8_t  valid;
    uint8_t  logBytes;    // 12 = 4 KiB
    uint8_t  next;        // next slot in fp chain (HASH_SLOT_NONE = end)
    uint8_t  _pad[5];

    HashPTE()
        : vaddr(0), paddr(0), flags(0), valid(0), logBytes(12),
          next(HASH_SLOT_NONE), _pad{} {}

    HashPTE(Addr _va, Addr _pa, uint64_t _fl, uint8_t _lb = 12)
        : vaddr(_va), paddr(_pa), flags(_fl), valid(1), logBytes(_lb),
          next(HASH_SLOT_NONE), _pad{} {}
};
static_assert(sizeof(HashPTE) == 32, "HashPTE must be 32 bytes");
static_assert(HASH_CACHE_LINE % sizeof(HashPTE) == 0,
              "HashPTE must not straddle a cache line");

// Header (routingTable + floatingCount) padded to whole cache lines
static constexpr size_t HASH_HDR_RAW  = HASH_NUM_FP_VALUES + sizeof(uint32_t);
static constexpr size_t HASH_HDR_SIZE =
    (HASH_HDR_RAW + HASH_CACHE_LINE - 1) / HASH_CACHE_LINE * HASH_CACHE_LINE;

struct HashBucket
{
    uint8_t  routingTable[HASH_NUM_FP_VALUES]; // fp -> first slot
    uint32_t floatingCount;                    // entries sent to overflow
    uint8_t  _pad[HASH_HDR_SIZE - HASH_HDR_RAW];
    HashPTE  entries[HASH_SLOTS];

    HashBucket() : floatingCount(0), _pad{} {
        for (size_t i = 0; i < HASH_NUM_FP_VALUES; i++)
            routingTable[i] = HASH_SLOT_NONE;
    }
};
static_assert(offsetof(HashBucket, entries) % HASH_CACHE_LINE == 0,
              "entries must start on a cache line");
static_assert(sizeof(HashBucket) % HASH_CACHE_LINE == 0,
              "HashBucket must be a whole number of cache lines");

// Overflow bucket: slots only (occupancy count kept in host memory)
struct OvBucket
{
    HashPTE slots[OV_SLOTS_PER_BKT];
};
static_assert(sizeof(OvBucket) % HASH_CACHE_LINE == 0,
              "OvBucket must be a whole number of cache lines");

// ============================================================
// HashArchPageTable
// ============================================================
class HashArchPageTable : public EmulationPageTable
{
  public:
    enum HashFlags : uint64_t
    {
        FLAG_VALID   = 1 << 0,
        FLAG_READ    = 1 << 1,
        FLAG_WRITE   = 1 << 2,
        FLAG_EXECUTE = 1 << 3,
        FLAG_USER    = 1 << 4,
    };

  private:
    System *system_;
    Addr    baseAddr_;
    bool    initialized_;
    const size_t numBuckets_;     // main buckets
    const size_t numOvBuckets_;   // overflow buckets (even)
    const size_t ovHalf_;         // two-choice split point

    // ── Host-memory-only state ────────────────────────────────
    uint64_t tab_[8][256];                 // main bucket hash
    uint64_t ovTab1_[8][256];              // overflow hash 1
    uint64_t ovTab2_[8][256];              // overflow hash 2
    uint64_t fpMul_;                       // fingerprint multiplier
    std::vector<uint32_t> ovCounts_;       // overflow occupancy (map-time)

    struct SlotStack
    {
        uint8_t data[HASH_SLOTS];
        uint8_t top;
        SlotStack() : top(0) {}
        void    push(uint8_t i) { data[top++] = i; }
        uint8_t pop()           { return data[--top]; }
        bool    empty() const   { return top == 0; }
    };
    std::vector<SlotStack> vacMap_;        // free slots per main bucket

    size_t numEntries_;
    size_t numOverflow_;

    // ── Hash functions ────────────────────────────────────────
    static uint64_t
    xorshift(uint64_t s)
    {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }

    void
    initHashes()
    {
        // Fixed seeds for reproducible layouts across runs
        uint64_t s = 0x123456789abcdef0ULL;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 256; j++)
                tab_[i][j] = (s = xorshift(s));

        s = 0xdeadbeefcafe1234ULL;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 256; j++) {
                ovTab1_[i][j] = (s = xorshift(s));
                ovTab2_[i][j] = (s = xorshift(s));
            }

        fpMul_ = xorshift(s) | 1;   // odd multiplier
    }

    static uint64_t
    tabulationHash(const uint64_t tab[8][256], uint64_t key)
    {
        uint64_t h = 0;
        for (int i = 0; i < 8; i++)
            h ^= tab[i][(key >> (i * 8)) & 0xFF];
        return h;
    }

    static Addr vpnKey(Addr vpn) { return vpn >> RiscvISA::PageShift; }

    // ── physProxy wrappers (functional access) ────────────────
    PortProxy &proxy() const { return system_->physProxy; }

    // A table created by fork()/clone() never gets initState() from
    // Process::initState(), so allocate on first use instead of writing
    // buckets at physical address 0.
    void
    ensureInit() const
    {
        if (!initialized_)
            const_cast<HashArchPageTable *>(this)->allocateLayout();
    }

    HashBucket
    readMainBucket(size_t idx) const
    {
        HashBucket bkt;
        proxy().readBlob(mainBucketAddr(idx), &bkt, sizeof(bkt));
        return bkt;
    }

    void
    writeMainBucket(size_t idx, const HashBucket &bkt)
    {
        proxy().writeBlob(mainBucketAddr(idx), &bkt, sizeof(bkt));
    }

    HashPTE
    readOvSlot(size_t b, size_t s) const
    {
        HashPTE pte;
        proxy().readBlob(ovSlotAddr(b, s), &pte, sizeof(pte));
        return pte;
    }

    void
    writeOvSlot(size_t b, size_t s, const HashPTE &pte)
    {
        proxy().writeBlob(ovSlotAddr(b, s), &pte, sizeof(pte));
    }

    // ── Chain helpers (links stored in entries) ───────────────
    static void
    chainInsert(HashBucket &bkt, uint64_t fp, uint8_t slot)
    {
        bkt.entries[slot].next = bkt.routingTable[fp];
        bkt.routingTable[fp]   = slot;
    }

    static void
    chainRemove(HashBucket &bkt, uint64_t fp, uint8_t slot)
    {
        uint8_t cur = bkt.routingTable[fp];
        if (cur == slot) {
            bkt.routingTable[fp] = bkt.entries[slot].next;
        } else {
            while (cur != HASH_SLOT_NONE && bkt.entries[cur].next != slot)
                cur = bkt.entries[cur].next;
            if (cur != HASH_SLOT_NONE)
                bkt.entries[cur].next = bkt.entries[slot].next;
        }
        bkt.entries[slot].next = HASH_SLOT_NONE;
    }

    // ── Overflow (two-choice hashing) ─────────────────────────
    bool
    ovUpdate(size_t b, Addr vpn, Addr pa, uint64_t flags)
    {
        for (size_t s = 0; s < OV_SLOTS_PER_BKT; s++) {
            HashPTE pte = readOvSlot(b, s);
            if (pte.valid && pte.vaddr == vpn) {
                pte.paddr = pa;
                pte.flags = flags;
                writeOvSlot(b, s, pte);
                return true;
            }
        }
        return false;
    }

    bool
    ovMap(Addr vpn, Addr pa, uint64_t flags)
    {
        size_t b1 = ovHash1(vpn), b2 = ovHash2(vpn);
        if (ovUpdate(b1, vpn, pa, flags) || ovUpdate(b2, vpn, pa, flags))
            return true;

        size_t b = (ovCounts_[b1] <= ovCounts_[b2]) ? b1 : b2;
        for (size_t s = 0; s < OV_SLOTS_PER_BKT; s++) {
            if (!readOvSlot(b, s).valid) {
                writeOvSlot(b, s, HashPTE(vpn, pa, flags));
                ovCounts_[b]++;
                return true;
            }
        }
        return false;
    }

    bool
    ovUnmap(Addr vpn)
    {
        for (size_t b : {ovHash1(vpn), ovHash2(vpn)}) {
            for (size_t s = 0; s < OV_SLOTS_PER_BKT; s++) {
                HashPTE pte = readOvSlot(b, s);
                if (pte.valid && pte.vaddr == vpn) {
                    writeOvSlot(b, s, HashPTE());
                    ovCounts_[b]--;
                    return true;
                }
            }
        }
        return false;
    }

    // ── Physical-structure operations (no EmulationPageTable calls) ──
    // Kept separate so remap() can move entries without touching the
    // host map twice.
    void
    physMap(Addr vpn, Addr pa, uint64_t hflags)
    {
        ensureInit();
        size_t   idx = mainHash(vpn);
        uint64_t fp  = fingerprint(vpn);
        HashBucket bkt = readMainBucket(idx);

        // Update in place if already present in main bucket
        for (uint8_t cur = bkt.routingTable[fp]; cur != HASH_SLOT_NONE;
             cur = bkt.entries[cur].next) {
            HashPTE &e = bkt.entries[cur];
            if (e.valid && e.vaddr == vpn) {
                e.paddr = pa;
                e.flags = hflags;
                writeMainBucket(idx, bkt);
                return;
            }
        }

        // A VPN may already live in overflow (spilled while the bucket was
        // full). Update it there instead of inserting a second copy into a
        // slot freed since then; otherwise a later unmap would remove only
        // the main copy and the walker would find the stale overflow one.
        if (bkt.floatingCount > 0 &&
            (ovUpdate(ovHash1(vpn), vpn, pa, hflags) ||
             ovUpdate(ovHash2(vpn), vpn, pa, hflags)))
            return;

        if (!vacMap_[idx].empty()) {
            uint8_t slot = vacMap_[idx].pop();
            bkt.entries[slot] = HashPTE(vpn, pa, hflags,
                                        RiscvISA::PageShift);
            chainInsert(bkt, fp, slot);
            writeMainBucket(idx, bkt);
            numEntries_++;
            return;
        }

        // Main bucket full: spill to overflow. Losing a mapping here would
        // make the walker fault while functional accesses still succeed,
        // so treat a full table as a hard error.
        fatal_if(!ovMap(vpn, pa, hflags),
                 "HashArchPageTable: bucket %d and both overflow buckets "
                 "are full (vpn %#x). Increase Process.hashPTBuckets / "
                 "hashPTOvBuckets.", idx, vpn);
        bkt.floatingCount++;
        writeMainBucket(idx, bkt);
        numOverflow_++;
    }

    void
    physUnmap(Addr vpn)
    {
        ensureInit();
        size_t   idx = mainHash(vpn);
        uint64_t fp  = fingerprint(vpn);
        HashBucket bkt = readMainBucket(idx);

        for (uint8_t cur = bkt.routingTable[fp]; cur != HASH_SLOT_NONE;
             cur = bkt.entries[cur].next) {
            if (bkt.entries[cur].valid && bkt.entries[cur].vaddr == vpn) {
                chainRemove(bkt, fp, cur);
                bkt.entries[cur] = HashPTE();
                vacMap_[idx].push(cur);
                writeMainBucket(idx, bkt);
                numEntries_--;
                return;
            }
        }

        if (ovUnmap(vpn)) {
            if (bkt.floatingCount > 0)
                bkt.floatingCount--;
            writeMainBucket(idx, bkt);
            numOverflow_--;
        }
    }

  public:
    HashArchPageTable(const std::string &name, uint64_t pid,
                      System *sys, Addr pageSize,
                      size_t numBuckets = HASH_DEFAULT_BUCKETS,
                      size_t numOvBuckets = HASH_DEFAULT_OV_BUCKETS)
        : EmulationPageTable(name, pid, pageSize),
          system_(sys), baseAddr_(0), initialized_(false),
          numBuckets_(numBuckets), numOvBuckets_(numOvBuckets),
          ovHalf_(numOvBuckets / 2),
          fpMul_(0), ovCounts_(numOvBuckets, 0),
          vacMap_(numBuckets), numEntries_(0), numOverflow_(0)
    {
        fatal_if(numBuckets_ == 0, "HashArchPageTable: numBuckets must be > 0");
        fatal_if(numOvBuckets_ < 2 || numOvBuckets_ % 2,
                 "HashArchPageTable: numOvBuckets must be even and >= 2 "
                 "(two-choice hashing splits it in half), got %d",
                 numOvBuckets_);
        initHashes();
        for (auto &st : vacMap_)
            for (size_t i = HASH_SLOTS; i-- > 0;)
                st.push(static_cast<uint8_t>(i));
    }

    // Allocate and initialise the physical layout (idempotent).
    void
    allocateLayout()
    {
        if (initialized_)
            return;

        auto *se = dynamic_cast<SEWorkload *>(system_->workload);
        fatal_if(!se, "HashArchPageTable: SEWorkload not found");

        size_t npages = (physLayoutSize() + _pageSize - 1) / _pageSize;
        baseAddr_ = se->allocPhysPages(npages);
        proxy().memsetBlob(baseAddr_, 0, npages * _pageSize);

        // routingTable must start as HASH_SLOT_NONE, not zero
        for (size_t i = 0; i < numBuckets_; i++)
            writeMainBucket(i, HashBucket());

        initialized_ = true;
    }

    // Called from Process::initState() via pTable->initState().
    void
    initState() override
    {
        if (shared)
            return;
        allocateLayout();
    }

    // ── Address helpers used by HashWalker ────────────────────
    Addr getBaseAddr() const { return baseAddr_; }

    Addr
    mainBucketAddr(size_t idx) const
    {
        return baseAddr_ + idx * sizeof(HashBucket);
    }

    Addr
    routingAddr(size_t idx, uint64_t fp) const
    {
        return mainBucketAddr(idx) + offsetof(HashBucket, routingTable) + fp;
    }

    Addr
    floatingAddr(size_t idx) const
    {
        return mainBucketAddr(idx) + offsetof(HashBucket, floatingCount);
    }

    Addr
    entryAddr(size_t idx, uint8_t slot) const
    {
        return mainBucketAddr(idx) + offsetof(HashBucket, entries) +
               slot * sizeof(HashPTE);
    }

    Addr
    ovSlotAddr(size_t b, size_t slot) const
    {
        return baseAddr_ + numBuckets_ * sizeof(HashBucket) +
               b * sizeof(OvBucket) + slot * sizeof(HashPTE);
    }

    size_t
    mainHash(Addr vpn) const
    {
        return tabulationHash(tab_, vpnKey(vpn)) % numBuckets_;
    }

    uint64_t
    fingerprint(Addr vpn) const
    {
        return (fpMul_ * vpnKey(vpn)) >> (64 - HASH_FP_BITS);
    }

    size_t
    ovHash1(Addr vpn) const
    {
        return tabulationHash(ovTab1_, vpnKey(vpn)) % ovHalf_;
    }

    size_t
    ovHash2(Addr vpn) const
    {
        return ovHalf_ + tabulationHash(ovTab2_, vpnKey(vpn)) % ovHalf_;
    }

    // ── EmulationPageTable interface ──────────────────────────
    void
    map(Addr vaddr, Addr paddr, int64_t size,
        uint64_t flags = 0, bool executable = false) override
    {
        EmulationPageTable::map(vaddr, paddr, size, flags);

        // Same permissions as HierarchySv39::reset() for a leaf PTE:
        // always R/W/U, X only for executable segments.
        uint64_t hflags = FLAG_VALID | FLAG_READ | FLAG_WRITE | FLAG_USER;
        if (executable)
            hflags |= FLAG_EXECUTE;

        for (int64_t off = 0; off < size; off += _pageSize)
            physMap((vaddr + off) & ~(_pageSize - 1), paddr + off, hflags);
    }

    void
    unmap(Addr vaddr, int64_t size) override
    {
        EmulationPageTable::unmap(vaddr, size);
        for (int64_t off = 0; off < size; off += _pageSize)
            physUnmap((vaddr + off) & ~(_pageSize - 1));
    }

    void
    remap(Addr vaddr, int64_t size, Addr new_vaddr) override
    {
        // Host map is moved by the base class; mirror the move in
        // physical memory using the phys* helpers only.
        EmulationPageTable::remap(vaddr, size, new_vaddr);

        for (int64_t off = 0; off < size; off += _pageSize) {
            Addr oldVpn = (vaddr + off)     & ~(_pageSize - 1);
            Addr newVpn = (new_vaddr + off) & ~(_pageSize - 1);
            HashPTE pte;
            if (physLookup(oldVpn, pte)) {
                physUnmap(oldVpn);
                physMap(newVpn, pte.paddr, pte.flags);
            }
        }
    }

    // Functional lookup through physProxy. Follows the same path as the
    // walker; used by TLB::translateFunctional.
    bool
    physLookup(Addr vpn, HashPTE &out) const
    {
        ensureInit();
        size_t   idx = mainHash(vpn);
        uint64_t fp  = fingerprint(vpn);
        HashBucket bkt = readMainBucket(idx);

        for (uint8_t cur = bkt.routingTable[fp]; cur != HASH_SLOT_NONE;
             cur = bkt.entries[cur].next) {
            if (bkt.entries[cur].valid && bkt.entries[cur].vaddr == vpn) {
                out = bkt.entries[cur];
                return true;
            }
        }

        if (bkt.floatingCount == 0)
            return false;

        for (size_t b : {ovHash1(vpn), ovHash2(vpn)})
            for (size_t s = 0; s < OV_SLOTS_PER_BKT; s++) {
                HashPTE pte = readOvSlot(b, s);
                if (pte.valid && pte.vaddr == vpn) {
                    out = pte;
                    return true;
                }
            }
        return false;
    }

    size_t getNumEntries()  const { return numEntries_; }
    size_t getNumOverflow() const { return numOverflow_; }

    size_t numBuckets()   const { return numBuckets_; }
    size_t numOvBuckets() const { return numOvBuckets_; }

    size_t
    physLayoutSize() const
    {
        return numBuckets_ * sizeof(HashBucket) +
               numOvBuckets_ * sizeof(OvBucket);
    }
};

} // namespace gem5

#endif // __ARCH_RISCV_HASH_ARCH_PAGE_TABLE_HH__
