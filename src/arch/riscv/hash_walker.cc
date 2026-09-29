/*
 * Hash Page Table Walker — implementation
 *
 * Control flow mirrors pagetable_walker.cc as closely as possible.
 * Comments mark where behaviour differs from the Sv39 Walker.
 */

#include "arch/riscv/hash_walker.hh"

#include <memory>

#include "arch/riscv/faults.hh"
#include "arch/riscv/page_size.hh"
#include "arch/riscv/tlb.hh"
#include "base/bitfield.hh"
#include "cpu/thread_context.hh"
#include "debug/HashPageTableWalker.hh"
#include "mem/packet_access.hh"
#include "mem/request.hh"
#include "arch/riscv/regs/misc.hh"
#include "sim/process.hh"

namespace gem5
{
namespace RiscvISA
{

// ============================================================
// HashWalker — top-level (mirrors Walker)
// ============================================================

Fault
HashWalker::start(ThreadContext *_tc, BaseMMU::Translation *_translation,
                  const RequestPtr &_req, BaseMMU::Mode _mode)
{
    WalkerState *newState = new WalkerState(this, _translation, _req);
    newState->initState(_tc, _mode, sys->isTimingMode());
    if (currStates.size()) {
        assert(newState->isTiming());
        DPRINTF(HashPageTableWalker,
                "Hash walks in progress: %d\n", currStates.size());
        currStates.push_back(newState);
        return NoFault;
    } else {
        currStates.push_back(newState);
        Fault fault = newState->startWalk();
        if (!newState->isTiming()) {
            currStates.pop_front();
            delete newState;
        }
        return fault;
    }
}

bool
HashWalker::WalkerPort::recvTimingResp(PacketPtr pkt)
{
    return walker->recvTimingResp(pkt);
}

bool
HashWalker::recvTimingResp(PacketPtr pkt)
{
    WalkerSenderState *senderState =
        dynamic_cast<WalkerSenderState *>(pkt->popSenderState());
    WalkerState *senderWalk = senderState->senderWalk;
    bool walkComplete = senderWalk->recvPacket(pkt);
    delete senderState;
    if (walkComplete) {
        for (auto iter = currStates.begin();
             iter != currStates.end(); iter++) {
            if (*iter == senderWalk) {
                currStates.erase(iter);
                break;
            }
        }
        delete senderWalk;
        if (currStates.size() && !startWalkWrapperEvent.scheduled())
            schedule(startWalkWrapperEvent, clockEdge());
    }
    return true;
}

void
HashWalker::WalkerPort::recvReqRetry()
{
    walker->recvReqRetry();
}

void
HashWalker::recvReqRetry()
{
    for (auto *ws : currStates)
        if (ws->isRetrying())
            ws->retry();
}

bool
HashWalker::sendTiming(WalkerState *sendingState, PacketPtr pkt)
{
    WalkerSenderState *ws = new WalkerSenderState(sendingState);
    pkt->pushSenderState(ws);
    if (port.sendTimingReq(pkt))
        return true;
    pkt->popSenderState();
    delete ws;
    return false;
}

Port &
HashWalker::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "port")
        return port;
    return ClockedObject::getPort(if_name, idx);
}

void
HashWalker::sendFirstPacket()
{
    // Called hashLatency cycles after startWalk() in Timing mode.
    // At this point the WalkerState is still at the front of currStates
    // and has already set state = Waiting, so sendPackets() can run.
    if (!currStates.empty())
        currStates.front()->sendPackets();
}

void
HashWalker::startWalkWrapper()
{
    unsigned num_squashed = 0;
    WalkerState *currState = currStates.front();
    while ((num_squashed < numSquashable) && currState &&
           currState->translation->squashed()) {
        currStates.pop_front();
        num_squashed++;

        DPRINTF(HashPageTableWalker,
                "Squashing hash walk for address %#x\n",
                currState->req->getVaddr());

        currState->translation->finish(
            std::make_shared<UnimpFault>("Squashed Inst"),
            currState->req, currState->tc, currState->mode);

        if (currState->numInflight() == 0)
            delete currState;
        else
            currState->squash();

        if (currStates.size())
            currState = currStates.front();
        else
            currState = nullptr;
    }
    if (currState && !currState->wasStarted())
        currState->startWalk();
}

// ============================================================
// WalkerState
// ============================================================

void
HashWalker::WalkerState::initState(ThreadContext *_tc,
                                    BaseMMU::Mode _mode,
                                    bool _isTiming)
{
    assert(state == Ready);
    started = false;
    tc      = _tc;
    mode    = _mode;
    timing  = _isTiming;
    // Same CSR snapshot as the Sv39 walker
    status  = tc->readMiscReg(MISCREG_STATUS);
    pmode   = walker->tlb->getMemPriv(tc, mode);
    satp    = tc->readMiscReg(MISCREG_SATP);

    auto *proc = tc->getProcessPtr();
    assert(proc && "HashWalker: no process on thread context");
    hashPT = dynamic_cast<HashArchPageTable *>(proc->pTable);
    fatal_if(!hashPT, "HashWalker: pTable is not a HashArchPageTable");
}

// ── startWalk ─────────────────────────────────────────────────
Fault
HashWalker::WalkerState::startWalk()
{
    Fault fault = NoFault;
    assert(!started);
    started = true;
    setupWalk(req->getVaddr());

    if (timing) {
        nextState   = state;
        state       = Waiting;
        timingFault = NoFault;
        if (walker->hashLatency > Cycles(0)) {
            // Defer the first packet by hashLatency cycles to model the
            // cost of computing the bucket address (hash + fingerprint).
            // Atomic mode skips this because event scheduling does not
            // work there; Atomic walks are used for warmup only.
            walker->schedule(walker->sendFirstPacketEvent,
                             walker->clockEdge(walker->hashLatency));
        } else {
            sendPackets();
        }
    } else {
        do {
            walker->port.sendAtomic(read);
            PacketPtr write = nullptr;
            fault = stepWalk(write);
            assert(fault == NoFault || read == nullptr);
            state     = nextState;
            nextState = Ready;
        } while (read);
        state     = Ready;
        nextState = Waiting;
    }
    return fault;
}

// ── setupWalk ─────────────────────────────────────────────────
// First read: routingTable[fp] of the main bucket (1 byte).
void
HashWalker::WalkerState::setupWalk(Addr vaddr)
{
    // Canonical form, identical to the Sv39 walker and TLB::doTranslate
    vaddr  = Addr(sext<VADDR_BITS>(vaddr));
    vpn    = vaddr & ~(hashPT->pageSize() - 1);
    bucket = hashPT->mainHash(vpn);
    fp     = hashPT->fingerprint(vpn);
    slot   = HASH_SLOT_NONE;
    ovPass = 0;
    ovSlot = 0;

    entry.vaddr = vaddr;
    entry.asid  = satp.asid;
    walkReads   = 0;
    walker->stats.walks++;

    DPRINTF(HashPageTableWalker,
            "Hash walk vaddr %#x vpn %#x bucket %d fp %#x\n",
            vaddr, vpn, bucket, fp);

    state     = Routing;
    nextState = Ready;
    read = makeReadPkt(hashPT->routingAddr(bucket, fp), sizeof(uint8_t),
                       Routing);
}

// ── nextRead ──────────────────────────────────────────────────
// Replace the consumed response with a Packet for the next field.
void
HashWalker::WalkerState::nextRead(Addr physAddr, size_t size,
                                  State parseState)
{
    delete read;
    read      = makeReadPkt(physAddr, size, parseState);
    nextState = parseState;
    DPRINTF(HashPageTableWalker, "  next read %#x (%d B)\n", physAddr, size);
}

// ── matchPTE ──────────────────────────────────────────────────
// Returns true if the walk is finished (hit or permission fault).
bool
HashWalker::WalkerState::matchPTE(const HashPTE &pte, Fault &fault)
{
    if (!pte.valid || pte.vaddr != vpn)
        return false;

    // Synthesise an Sv39-format PTE so TLB::checkPermissions() works both
    // here and on later TLB hits in TLB::doTranslate().
    PTESv39 sv = 0;
    sv.v   = 1;
    sv.r   = (pte.flags & HashArchPageTable::FLAG_READ)    ? 1 : 0;
    sv.w   = (pte.flags & HashArchPageTable::FLAG_WRITE)   ? 1 : 0;
    sv.x   = (pte.flags & HashArchPageTable::FLAG_EXECUTE) ? 1 : 0;
    sv.u   = (pte.flags & HashArchPageTable::FLAG_USER)    ? 1 : 0;
    sv.a   = 1;   // no A/D write-back in the hash walker
    sv.d   = 1;
    sv.ppn = pte.paddr >> PageShift;

    entry.pte      = sv;
    entry.paddr    = pte.paddr >> PageShift;
    entry.logBytes = pte.logBytes;

    fault = walker->tlb->checkPermissions(status, pmode, entry.vaddr,
                                          mode, sv);
    return true;
}

// ── stepWalk ──────────────────────────────────────────────────
// Parses the single field in `read` and either finishes the walk or
// queues exactly one more single-field read.
Fault
HashWalker::WalkerState::stepWalk(PacketPtr &write)
{
    assert(state != Ready && state != Waiting);
    write = nullptr;             // hash walk never writes back

    Fault fault    = NoFault;
    bool  finished = false;      // walk ends after this step
    bool  hit      = false;      // translation found and permitted

    switch (state) {
      case Routing: {
        slot = read->getLE<uint8_t>();
        if (slot != HASH_SLOT_NONE)
            nextRead(hashPT->entryAddr(bucket, slot), sizeof(HashPTE), Entry);
        else
            nextRead(hashPT->floatingAddr(bucket), sizeof(uint32_t),
                     Floating);
        break;
      }

      case Entry: {
        HashPTE pte;
        read->writeData(reinterpret_cast<uint8_t *>(&pte));
        if (matchPTE(pte, fault)) {
            finished = true;
            hit      = (fault == NoFault);
        } else if (pte.next != HASH_SLOT_NONE) {
            slot = pte.next;
            nextRead(hashPT->entryAddr(bucket, slot), sizeof(HashPTE), Entry);
        } else {
            nextRead(hashPT->floatingAddr(bucket), sizeof(uint32_t),
                     Floating);
        }
        break;
      }

      case Floating: {
        uint32_t floating = read->getLE<uint32_t>();
        if (floating == 0) {
            DPRINTF(HashPageTableWalker, "vpn %#x not mapped\n", vpn);
            fault    = pageFault();
            finished = true;
        } else {
            walker->stats.overflowWalks++;
            ovPass   = 0;
            ovSlot   = 0;
            ovBucket = hashPT->ovHash1(vpn);
            nextRead(hashPT->ovSlotAddr(ovBucket, ovSlot),
                     sizeof(HashPTE), OvSlot);
        }
        break;
      }

      case OvSlot: {
        HashPTE pte;
        read->writeData(reinterpret_cast<uint8_t *>(&pte));
        if (matchPTE(pte, fault)) {
            finished = true;
            hit      = (fault == NoFault);
        } else if (++ovSlot < OV_SLOTS_PER_BKT) {
            nextRead(hashPT->ovSlotAddr(ovBucket, ovSlot),
                     sizeof(HashPTE), OvSlot);
        } else if (ovPass == 0) {
            ovPass   = 1;
            ovSlot   = 0;
            ovBucket = hashPT->ovHash2(vpn);
            nextRead(hashPT->ovSlotAddr(ovBucket, ovSlot),
                     sizeof(HashPTE), OvSlot);
        } else {
            DPRINTF(HashPageTableWalker,
                    "vpn %#x not found in overflow\n", vpn);
            fault    = pageFault();
            finished = true;
        }
        break;
      }

      default:
        panic("HashWalker: unexpected state %d", state);
    }

    if (!finished)
        return NoFault;

    walker->stats.readsPerWalk.sample(walkReads);

    if (hit) {
        // PMP check on the final physical address (as in the Sv39 walker)
        Addr paddr = entry.paddr << PageShift;
        RequestPtr pmpReq = std::make_shared<Request>(
            paddr, 1 << entry.logBytes, read->req->getFlags(),
            walker->requestorId);
        fault = walker->pmp->pmpCheck(pmpReq, mode, pmode, tc, entry.vaddr);

        if (fault == NoFault) {
            // Same as the Sv39 walker: TLB::insert() asserts that an
            // existing entry for this page has an identical vaddr, so the
            // key must be page-aligned (matters when walks overlap, e.g. O3).
            entry.vaddr &= ~mask(entry.logBytes);
            walker->tlb->insert(entry.vaddr, entry);
            walker->stats.hits++;
            DPRINTF(HashPageTableWalker, "Translated %#x -> %#x\n",
                    entry.vaddr, entry.paddr << PageShift);
        }
    }
    endWalk();
    return fault;
}

// ── makeReadPkt ───────────────────────────────────────────────
PacketPtr
HashWalker::WalkerState::makeReadPkt(Addr physAddr, size_t size,
                                     State kind)
{
    walkReads++;
    walker->stats.reads++;
    switch (kind) {
      case Routing:  walker->stats.routingReads++;  break;
      case Entry:    walker->stats.entryReads++;    break;
      case Floating: walker->stats.floatingReads++; break;
      case OvSlot:   walker->stats.ovSlotReads++;   break;
      default: panic("HashWalker: bad read kind %d", kind);
    }

    // A single field never crosses a cache line (checked at compile time
    // in hash_arch_page_table.hh); assert it here as well.
    assert(physAddr / HASH_CACHE_LINE ==
           (physAddr + size - 1) / HASH_CACHE_LINE);

    RequestPtr request = std::make_shared<Request>(
        physAddr, size, Request::PHYSICAL, walker->requestorId);
    PacketPtr pkt = new Packet(request, MemCmd::ReadReq);
    pkt->allocate();
    return pkt;
}

// ── recvPacket (same control flow as Sv39) ────────────────────
bool
HashWalker::WalkerState::recvPacket(PacketPtr pkt)
{
    assert(pkt->isResponse());
    assert(inflight);
    assert(state == Waiting);
    inflight--;

    if (squashed)
        return (inflight == 0);

    if (pkt->isRead()) {
        assert(!read);
        pkt->headerDelay = pkt->payloadDelay = 0;

        state     = nextState;
        nextState = Ready;
        PacketPtr write = nullptr;
        read        = pkt;
        timingFault = stepWalk(write);
        state       = Waiting;
        assert(timingFault == NoFault || read == nullptr);
        if (write)
            writes.push_back(write);
        sendPackets();
    } else {
        delete pkt;
        sendPackets();
    }

    if (inflight == 0 && read == nullptr && writes.empty()) {
        state     = Ready;
        nextState = Waiting;
        if (timingFault == NoFault) {
            Addr vaddr = Addr(sext<VADDR_BITS>(req->getVaddr()));
            Addr paddr = walker->tlb->translateWithTLB(vaddr, satp.asid,
                                                       mode);
            req->setPaddr(paddr);
            walker->pma->check(req);
            // Same as the Sv39 walker: re-check PMP on the actual request
            timingFault = walker->pmp->pmpCheck(req, mode, pmode, tc);
        }
        translation->finish(timingFault, req, tc, mode);
        return true;
    }
    return false;
}

// ── sendPackets (same as Sv39) ────────────────────────────────
void
HashWalker::WalkerState::sendPackets()
{
    if (retrying)
        return;

    if (read) {
        PacketPtr pkt = read;
        read = nullptr;
        inflight++;
        if (!walker->sendTiming(this, pkt)) {
            retrying = true;
            read     = pkt;
            inflight--;
            return;
        }
    }
    while (!writes.empty()) {
        PacketPtr write = writes.back();
        writes.pop_back();
        inflight++;
        if (!walker->sendTiming(this, write)) {
            retrying = true;
            writes.push_back(write);
            inflight--;
            return;
        }
    }
}

void
HashWalker::WalkerState::endWalk()
{
    nextState = Ready;
    delete read;
    read = nullptr;
}

unsigned HashWalker::WalkerState::numInflight() const { return inflight; }
bool     HashWalker::WalkerState::isRetrying()        { return retrying; }
bool     HashWalker::WalkerState::isTiming()          { return timing; }
bool     HashWalker::WalkerState::wasStarted()        { return started; }
void     HashWalker::WalkerState::squash()            { squashed = true; }

void
HashWalker::WalkerState::retry()
{
    retrying = false;
    sendPackets();
}

Fault
HashWalker::WalkerState::pageFault()
{
    DPRINTF(HashPageTableWalker, "Raising page fault.\n");
    walker->stats.pageFaults++;
    return walker->tlb->createPagefault(entry.vaddr, mode);
}

// ============================================================
// Stats
// ============================================================

HashWalker::HashWalkerStats::HashWalkerStats(statistics::Group *parent)
    : statistics::Group(parent),
    ADD_STAT(walks, statistics::units::Count::get(),
             "Number of hash page table walks started"),
    ADD_STAT(reads, statistics::units::Count::get(),
             "Memory reads issued by the walker (one field per read)"),
    ADD_STAT(routingReads, statistics::units::Count::get(),
             "Reads of routingTable[fp]"),
    ADD_STAT(entryReads, statistics::units::Count::get(),
             "Reads of main-bucket PTEs (one per chain step)"),
    ADD_STAT(floatingReads, statistics::units::Count::get(),
             "Reads of floatingCount (main-chain misses)"),
    ADD_STAT(ovSlotReads, statistics::units::Count::get(),
             "Reads of overflow slots"),
    ADD_STAT(overflowWalks, statistics::units::Count::get(),
             "Walks that searched the overflow table"),
    ADD_STAT(hits, statistics::units::Count::get(),
             "Walks that found a permitted translation"),
    ADD_STAT(pageFaults, statistics::units::Count::get(),
             "Walks that raised a page fault"),
    ADD_STAT(avgReadsPerWalk,
             statistics::units::Rate<statistics::units::Count,
                                     statistics::units::Count>::get(),
             "Average memory reads per walk", reads / walks),
    ADD_STAT(readsPerWalk, statistics::units::Count::get(),
             "Distribution of memory reads per completed walk")
{
    readsPerWalk.init(1, 16, 1);
}

} // namespace RiscvISA
} // namespace gem5
