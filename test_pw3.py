from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.memory.single_channel import SingleChannelDDR4_2400
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_no_ptwc_hierarchy import (
    PrivateL1PrivateL2CacheNoPTWCHierarchy,
)
from gem5.components.processors.cpu_types import CPUTypes
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator
from gem5.isas import ISA
from m5.objects import RiscvHashWalker

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--arch-pt", action="store_true")
parser.add_argument("--hash-pt", action="store_true")
parser.add_argument("--hash-pt-buckets", type=int, default=1024)
parser.add_argument("--hash-pt-ov-buckets", type=int, default=1024)
parser.add_argument("--hash-latency", type=int, default=0,
                    help="extra cycles for hash address computation (default 0)")
parser.add_argument("--binary", required=True)
args = parser.parse_args()

if args.arch_pt and args.hash_pt:
    parser.error("--arch-pt and --hash-pt are mutually exclusive")

board = SimpleBoard(
    clk_freq="1GHz",
    processor=SimpleProcessor(
        cpu_type=CPUTypes.TIMING,
        num_cores=1,
        isa=ISA.RISCV,
    ),
    memory=SingleChannelDDR4_2400("1GB"),
    cache_hierarchy=PrivateL1PrivateL2CacheNoPTWCHierarchy(
        l1d_size="64kB", l1i_size="64kB", l2_size="256kB"
    ),
)

board.set_se_binary_workload(
    binary=BinaryResource(local_path=args.binary),
    use_arch_pt=args.arch_pt,
    use_hash_pt=args.hash_pt,
    hash_pt_buckets=args.hash_pt_buckets,
    hash_pt_ov_buckets=args.hash_pt_ov_buckets,
)

# Attach hash walkers when --hash-pt is set.
# Each TLB needs its own walker: a walker inserts into the TLB that owns it,
# so sharing one would put instruction translations into the D-TLB.
# Their ports are wired to the L2 bus by RiscvMMU.connectWalkerPorts().
if args.hash_pt:
    for core in board.get_processor().get_cores():
        core.core.mmu.itb.hash_walker = RiscvHashWalker(
            hash_latency=args.hash_latency)
        core.core.mmu.dtb.hash_walker = RiscvHashWalker(
            hash_latency=args.hash_latency)

Simulator(board=board).run()
print("Done")
