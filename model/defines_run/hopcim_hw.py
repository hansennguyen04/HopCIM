"""DeFiNES accelerator shaped like HopCIM: a 1,024-MAC array (one 64 x 16 tile) whose weights stay in the array
(per-MAC weight cells and an on-chip weight store at no access cost: the CIM holds all 61 tiles), column
accumulators, and the activation memories at CACTI's 28-nm costs from energy_components.json:
  layer-by-layer (HOPCIM_LB unset):  ACT (two padded maps, see below), 4.329 / 3.898 pJ per 56-bit word
  depth-first    (HOPCIM_LB=1):      in addition a line-buffer level (sized below), 0.627 / 1.414 pJ per word
A DRAM level is required at the top; it holds the network input and output and the initial weights only."""
import os
from classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from classes.hardware.architecture.operational_unit import Multiplier
from classes.hardware.architecture.operational_array import MultiplierArray
from classes.hardware.architecture.memory_instance import MemoryInstance
from classes.hardware.architecture.accelerator import Accelerator
from classes.hardware.architecture.core import Core

LB = os.environ.get("HOPCIM_LB") == "1"


def hierarchy(array):
    xbank = MemoryInstance(name="xbank", size=2 * 64 * 9, r_bw=64 * 8, w_bw=64, r_cost=0.0, w_cost=0.0, area=0, bank=1,
                           random_bank_access=True, r_port=1, w_port=1, rw_port=0, latency=1)
    acc = MemoryInstance(name="acc", size=24, r_bw=24, w_bw=24, r_cost=0.0, w_cost=0.0, area=0, bank=1,
                         random_bank_access=False, r_port=2, w_port=2, rw_port=0, latency=1)
    wcell = MemoryInstance(name="wcell", size=8 * 64, r_bw=8, w_bw=8, r_cost=0.0, w_cost=0.0, area=0, bank=1,
                           random_bank_access=False, r_port=1, w_port=1, rw_port=0, latency=1)
    wstore = MemoryInstance(name="cim_cells", size=61 * 64 * 16 * 8, r_bw=1024 * 8, w_bw=1024 * 8, r_cost=0.0, w_cost=0.0,
                            area=0, bank=1, random_bank_access=True, r_port=1, w_port=1, rw_port=0, latency=1)
    # DeFiNES keeps padded rows (7 columns): 3 rows x 7 x 64 bytes per depthwise layer, plus 2 x 64 bytes of fusion
    # vector per block; at the access energy of HopCIM's 150-word line buffer
    lb = MemoryInstance(name="line_buffers", size=4 * (3 * 7 * 64 + 2 * 64) * 8, r_bw=56, w_bw=56, r_cost=0.6273258, w_cost=1.4137504,
                        area=0, bank=1, random_bank_access=True, r_port=1, w_port=1, rw_port=0, latency=1,
                        min_r_granularity=8, min_w_granularity=8)
    # DeFiNES stores depthwise inputs with their padding (27 x 7 x 64 bytes); HopCIM pads on the fly. Room for two
    # padded maps, at the access energy of HopCIM's 2,500-word ACT.
    act = MemoryInstance(name="act", size=2 * 27 * 7 * 64 * 8, r_bw=56, w_bw=56, r_cost=4.329228, w_cost=3.89783, area=0, bank=1,
                         random_bank_access=True, r_port=1, w_port=1, rw_port=0, latency=1,
                         min_r_granularity=8, min_w_granularity=8)
    dram = MemoryInstance(name="dram", size=10 ** 10, r_bw=64, w_bw=64, r_cost=700, w_cost=750, area=0, bank=1,
                          random_bank_access=False, r_port=0, w_port=0, rw_port=1, latency=1)
    m = MemoryHierarchy(operational_array=array)
    m.add_memory(memory_instance=wcell, operands=('I2',),
                 port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},), served_dimensions=set())
    # the X banks: one register file for the whole array (not unrolled), so it also serves depthwise layers; it holds a
    # pass's input vector, which conv1 and pointwise reuse over their 4 passes (an input register shared along D1
    # cannot serve depthwise layers, where D1 is the channel, and DeFiNES then fails to allocate)
    m.add_memory(memory_instance=xbank, operands=('I1',),
                 port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},), served_dimensions='all')
    m.add_memory(memory_instance=acc, operands=('O',),
                 port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_2', 'th': 'r_port_2'},),
                 served_dimensions={(0, 1, 0), (0, 0, 1)})
    m.add_memory(memory_instance=wstore, operands=('I2',),
                 port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},), served_dimensions='all')
    if LB:
        m.add_memory(memory_instance=lb, operands=('I1', 'O'),
                     port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},
                                 {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_1', 'th': 'r_port_1'},),
                     served_dimensions='all')
    m.add_memory(memory_instance=act, operands=('I1', 'O'),
                 port_alloc=({'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': None, 'th': None},
                             {'fh': 'w_port_1', 'tl': 'r_port_1', 'fl': 'w_port_1', 'th': 'r_port_1'},),
                 served_dimensions='all')
    m.add_memory(memory_instance=dram, operands=('I1', 'I2', 'O'),
                 port_alloc=({'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                             {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': None, 'th': None},
                             {'fh': 'rw_port_1', 'tl': 'rw_port_1', 'fl': 'rw_port_1', 'th': 'rw_port_1'},),
                 served_dimensions='all')
    return m


def cores():
    array = MultiplierArray(Multiplier([8, 8], 0.0, 1), {'D1': 16, 'D2': 4, 'D3': 16})
    return {Core(1, array, hierarchy(array))}


accelerator = Accelerator("hopcim_df" if LB else "hopcim_lbl", cores(), None)
