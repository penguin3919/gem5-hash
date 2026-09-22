from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.memory.single_channel import SingleChannelDDR4_2400
from gem5.components.cachehierarchies.classic.private_l1_cache_hierarchy import (
    PrivateL1CacheHierarchy,
)
from gem5.components.processors.cpu_types import CPUTypes
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator
from gem5.isas import ISA

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--arch-pt", action="store_true")
parser.add_argument("--binary", required=True)
args = parser.parse_args()

board = SimpleBoard(
    clk_freq="1GHz",
    processor=SimpleProcessor(
        cpu_type=CPUTypes.O3,
        num_cores=1,
        isa=ISA.RISCV,
    ),
    memory=SingleChannelDDR4_2400("1GB"),
    cache_hierarchy=PrivateL1CacheHierarchy(
        l1d_size="64kB", l1i_size="64kB"
    ),
)

board.set_se_binary_workload(
    binary=BinaryResource(local_path=args.binary),
    use_arch_pt=args.arch_pt,
)

Simulator(board=board).run()
print("Done")
