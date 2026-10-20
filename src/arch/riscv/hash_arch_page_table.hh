/*
 * Hash-based Architectural Page Table for gem5 SE mode (RISC-V)
 *
 * Inherits EmulationPageTable and mirrors every mapping into a hash
 * structure placed in simulated physical memory, so that HashWalker can
 * translate addresses with real (cacheable) memory accesses.
 *
 * v5 layout: 8 B entries, one bucket = one 64 B cache line
 *
 *   baseAddr_
 *   +-- HashLine[numBuckets]       main buckets, 64 B each (8 entries)
 *   +-- HashLine[numOvBuckets]     overflow buckets, 64 B each (8 entries)
 *
 * Walker reads (every read is one whole cache line, 64 B):
 *   1. main bucket line        -> compare the 8 VPN tags inside the line
 *   2. overflow line ovHash1   -> only if the bucket's spill bit is set
 *   3. overflow line ovHash2
 *
 * Compared with v4 (32 B entries, 1344 B buckets):
 *   - no routingTable / fingerprint: the whole bucket arrives in one read
 *   - no floatingCount read: the walker sees a 1-bit "spilled" flag kept in
 *     slot 0 of the bucket line; the count itself lives in host memory
 *   - no fp chain, so no next field and no vacancy stack
 *
 * HashPTE (8 B):
 *   bits  0..26  VPN tag  = vaddr bits 12..38 (Sv39 VA, bit 38 is the sign)
 *   bits 27..52  PPN      (26 bit: physical address < 256 GiB)
 *   bits 53..56  R W X U  (the only permission bits the walker uses)
 *   bits 57..58  page size (0 = 4 KiB, 1 = 2 MiB, 2 = 1 GiB)
 *   bit  59      valid
 *   bit  60      spilled  (slot 0 of a main bucket only)
 *   bits 61..63  unused
 *   No ASID: every process has its own table (gem5 SE mode).
 *
 * numBuckets / numOvBuckets come from Process.hashPTBuckets /
 * Process.hashPTOvBuckets. Slot count for 1 GiB is 262,144
 * ((main + overflow) x 8), but the usable load depends on the split:
 * a page that finds its main bucket and both overflow candidates full
 * stops the simulation (fatal). Measured first fatal (SE-like layout):
 *   main + overflow     first fatal          load
 *   32256 + 512         ~156K pages (611 MiB) 60%
 *   30720 + 2048        ~195K pages (762 MiB) 75%
 *   28672 + 4096        ~217K pages (845 MiB) 83%
 *   24576 + 8192        ~235K pages (917 MiB) 90%
 * More overflow = more pages fit, but those pages cost 2-3 reads per walk.
 *
 * Host memory only (never read by the walker):
 *   tabulation hash tables, per-bucket overflow counts (floatingCount),
 *   overflow bucket occupancy, statistics.
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
static constexpr size_t HASH_CACHE_LINE         = 64;
static constexpr size_t HASH_DEFAULT_BUCKETS    = 4096;  // x 8 = 32,768 slots
static constexpr size_t HASH_DEFAULT_OV_BUCKETS = 512;   // x 8 =  4,096 slots

// ============================================================
// Structures placed in simulated physical memory
// ============================================================

// PTE: 8 B, eight per cache line
struct HashPTE
{
    uint64_t raw;

    static constexpr unsigned VPN_BITS = 27, PPN_BITS = 26;
    static constexpr unsigned VPN_SH = 0, PPN_SH = 27, FL_SH = 53,
                              SZ_SH = 57, V_SH = 59, SP_SH = 60;
    static constexpr uint64_t M(unsigned b) { return (1ULL << b) - 1; }

    static constexpr unsigned VA_BITS = VPN_BITS + 12;           // 39 (Sv39)
    static constexpr Addr     MAX_PADDR = 1ULL << (PPN_BITS + 12);

    // VPN tag of a page-aligned, canonical Sv39 virtual address
    static uint64_t tagOf(Addr vpn) { return (vpn >> 12) & M(VPN_BITS); }

    // True if bits 63..39 are copies of bit 38 (canonical Sv39 address)
    static bool
    canonical(Addr va)
    {
        uint64_t top = va >> (VA_BITS - 1);           // bits 63..38
        return top == 0 || top == M(64 - VA_BITS + 1);
    }

    static unsigned lbToSel(unsigned lb) { return lb == 12 ? 0 : lb == 21 ? 1 : 2; }
    static unsigned selToLb(unsigned s)  { return s == 0 ? 12 : s == 1 ? 21 : 30; }

    HashPTE() : raw(0) {}

    // hflags uses HashArchPageTable::HashFlags (bit 0 valid, 1..4 R W X U)
    HashPTE(Addr vpn, Addr pa, uint64_t hflags, unsigned logBytes = 12)
        : raw(0)
    {
        raw = tagOf(vpn)                      << VPN_SH
            | ((pa >> 12) & M(PPN_BITS))      << PPN_SH
            | ((hflags >> 1) & M(4))          << FL_SH
            | uint64_t(lbToSel(logBytes))     << SZ_SH
            | 1ULL                            << V_SH;
    }

    uint64_t tag()      const { return (raw >> VPN_SH) & M(VPN_BITS); }
    Addr     paddr()    const { return ((raw >> PPN_SH) & M(PPN_BITS)) << 12; }
    uint64_t flags()    const { return (((raw >> FL_SH) & M(4)) << 1) | 1; }
    unsigned logBytes() const { return selToLb((raw >> SZ_SH) & 3); }
    bool     valid()    const { return (raw >> V_SH) & 1; }
    bool     spilled()  const { return (raw >> SP_SH) & 1; }

    // Canonical (sign-extended) page-aligned virtual address
    Addr
    vaddr() const
    {
        Addr va = tag() << 12;
        if (va & (1ULL << (VA_BITS - 1)))
            va |= ~M(VA_BITS);
        return va;
    }

    bool matches(Addr vpn) const { return valid() && tag() == tagOf(vpn); }

    void
    setSpilled(bool on)
    {
        raw = (raw & ~(1ULL << SP_SH)) | (uint64_t(on) << SP_SH);
    }

    // Update paddr / flags, keeping tag, size and spill bit
    void
    set(Addr pa, uint64_t hflags)
    {
        raw &= ~((M(PPN_BITS) << PPN_SH) | (M(4) << FL_SH));
        raw |= ((pa >> 12) & M(PPN_BITS)) << PPN_SH
             | ((hflags >> 1) & M(4))     << FL_SH;
    }

    // Replace the entry but keep this slot's spill bit
    void
    assign(const HashPTE &e)
    {
        bool sp = spilled();
        raw = e.raw;
        setSpilled(sp);
    }
};
static_assert(sizeof(HashPTE) == 8, "HashPTE must be 8 bytes");

static constexpr size_t HASH_SLOTS = HASH_CACHE_LINE / sizeof(HashPTE);  // 8

// One bucket = one cache line (main and overflow alike)
struct HashLine
{
    HashPTE e[HASH_SLOTS];

    // Slot holding vpn, or -1
    int
    find(Addr vpn) const
    {
        for (size_t s = 0; s < HASH_SLOTS; s++)
            if (e[s].matches(vpn))
                return int(s);
        return -1;
    }

    // First free slot, or -1
    int
    freeSlot() const
    {
        for (size_t s = 0; s < HASH_SLOTS; s++)
            if (!e[s].valid())
                return int(s);
        return -1;
    }

    // The spill flag lives in slot 0 and is independent of its valid bit
    bool spilled() const { return e[0].spilled(); }
};
static_assert(sizeof(HashLine) == HASH_CACHE_LINE,
              "a bucket must be exactly one cache line");

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
    const size_t numBuckets_;     // main buckets (one line each)
    const size_t numOvBuckets_;   // overflow buckets (one line each, even)
    const size_t ovHalf_;         // two-choice split point

    // ── Host-memory-only state ────────────────────────────────
    uint64_t tab_[8][256];                 // main bucket hash
    uint64_t ovTab1_[8][256];              // overflow hash 1
    uint64_t ovTab2_[8][256];              // overflow hash 2
    std::vector<uint32_t> floatingCount_;  // per main bucket: entries in overflow
    std::vector<uint32_t> ovCounts_;       // overflow occupancy (map-time)

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
        // Fixed seeds for reproducible layouts across runs (same as v4)
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

    HashLine
    readLine(Addr a) const
    {
        HashLine l;
        proxy().readBlob(a, &l, sizeof(l));
        return l;
    }

    void
    writeLine(Addr a, const HashLine &l)
    {
        proxy().writeBlob(a, &l, sizeof(l));
    }

    // ── Overflow (two-choice hashing, one line per bucket) ────
    bool
    ovUpdate(size_t b, Addr vpn, Addr pa, uint64_t flags)
    {
        HashLine l = readLine(ovLineAddr(b));
        int s = l.find(vpn);
        if (s < 0)
            return false;
        l.e[s].set(pa, flags);
        writeLine(ovLineAddr(b), l);
        return true;
    }

    bool
    ovMap(Addr vpn, Addr pa, uint64_t flags)
    {
        size_t b1 = ovHash1(vpn), b2 = ovHash2(vpn);
        if (ovUpdate(b1, vpn, pa, flags) || ovUpdate(b2, vpn, pa, flags))
            return true;

        size_t b = (ovCounts_[b1] <= ovCounts_[b2]) ? b1 : b2;
        HashLine l = readLine(ovLineAddr(b));
        int s = l.freeSlot();
        if (s < 0)
            return false;
        l.e[s] = HashPTE(vpn, pa, flags);
        writeLine(ovLineAddr(b), l);
        ovCounts_[b]++;
        return true;
    }

    bool
    ovUnmap(Addr vpn)
    {
        for (size_t b : {ovHash1(vpn), ovHash2(vpn)}) {
            HashLine l = readLine(ovLineAddr(b));
            int s = l.find(vpn);
            if (s >= 0) {
                l.e[s] = HashPTE();
                writeLine(ovLineAddr(b), l);
                ovCounts_[b]--;
                return true;
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
        fatal_if(!HashPTE::canonical(vpn),
                 "HashArchPageTable: vaddr %#x is not a canonical Sv39 "
                 "address (the 8 B entry keeps a 27-bit VPN)", vpn);
        fatal_if(pa >= HashPTE::MAX_PADDR,
                 "HashArchPageTable: paddr %#x does not fit the 26-bit PPN",
                 pa);

        size_t   idx = mainHash(vpn);
        HashLine bkt = readLine(mainLineAddr(idx));

        // Update in place if already present in the main bucket
        int s = bkt.find(vpn);
        if (s >= 0) {
            bkt.e[s].set(pa, hflags);
            writeLine(mainLineAddr(idx), bkt);
            return;
        }

        // A VPN may already live in overflow (spilled while the bucket was
        // full). Update it there instead of inserting a second copy into a
        // slot freed since then; otherwise a later unmap would remove only
        // the main copy and the walker would find the stale overflow one.
        if (floatingCount_[idx] > 0 &&
            (ovUpdate(ovHash1(vpn), vpn, pa, hflags) ||
             ovUpdate(ovHash2(vpn), vpn, pa, hflags)))
            return;

        s = bkt.freeSlot();
        if (s >= 0) {
            bkt.e[s].assign(HashPTE(vpn, pa, hflags, RiscvISA::PageShift));
            writeLine(mainLineAddr(idx), bkt);
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
        floatingCount_[idx]++;
        bkt.e[0].setSpilled(true);
        writeLine(mainLineAddr(idx), bkt);
        numOverflow_++;
    }

    void
    physUnmap(Addr vpn)
    {
        ensureInit();
        size_t   idx = mainHash(vpn);
        HashLine bkt = readLine(mainLineAddr(idx));

        int s = bkt.find(vpn);
        if (s >= 0) {
            bkt.e[s].assign(HashPTE());          // keeps slot 0's spill bit
            writeLine(mainLineAddr(idx), bkt);
            numEntries_--;
            return;
        }

        if (floatingCount_[idx] > 0 && ovUnmap(vpn)) {
            floatingCount_[idx]--;
            bkt.e[0].setSpilled(floatingCount_[idx] > 0);
            writeLine(mainLineAddr(idx), bkt);
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
          floatingCount_(numBuckets, 0), ovCounts_(numOvBuckets, 0),
          numEntries_(0), numOverflow_(0)
    {
        fatal_if(numBuckets_ == 0, "HashArchPageTable: numBuckets must be > 0");
        fatal_if(numOvBuckets_ < 2 || numOvBuckets_ % 2,
                 "HashArchPageTable: numOvBuckets must be even and >= 2 "
                 "(two-choice hashing splits it in half), got %d",
                 numOvBuckets_);
        initHashes();
    }

    // Allocate and zero the physical layout (idempotent). An all-zero
    // line is an empty bucket: every entry invalid, spill bit clear.
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
    mainLineAddr(size_t idx) const
    {
        return baseAddr_ + idx * sizeof(HashLine);
    }

    Addr
    ovLineAddr(size_t b) const
    {
        return baseAddr_ + (numBuckets_ + b) * sizeof(HashLine);
    }

    size_t
    mainHash(Addr vpn) const
    {
        return tabulationHash(tab_, vpnKey(vpn)) % numBuckets_;
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
                physMap(newVpn, pte.paddr(), pte.flags());
            }
        }
    }

    // Functional lookup through physProxy. Follows the same path as the
    // walker; used by TLB::translateFunctional.
    bool
    physLookup(Addr vpn, HashPTE &out) const
    {
        // The tag keeps only bits 12..38, so a non-canonical address could
        // alias a mapped page in the same bucket. Never mapped: not found.
        if (!HashPTE::canonical(vpn))
            return false;
        ensureInit();
        HashLine bkt = readLine(mainLineAddr(mainHash(vpn)));
        int s = bkt.find(vpn);
        if (s >= 0) {
            out = bkt.e[s];
            return true;
        }
        if (!bkt.spilled())
            return false;

        for (size_t b : {ovHash1(vpn), ovHash2(vpn)}) {
            HashLine l = readLine(ovLineAddr(b));
            s = l.find(vpn);
            if (s >= 0) {
                out = l.e[s];
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
        return (numBuckets_ + numOvBuckets_) * sizeof(HashLine);
    }
};

} // namespace gem5

#endif // __ARCH_RISCV_HASH_ARCH_PAGE_TABLE_HH__
