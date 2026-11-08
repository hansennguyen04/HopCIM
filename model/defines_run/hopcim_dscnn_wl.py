"""DeFiNES workload: the MLPerf Tiny DS-CNN's conv1 and four depthwise-separable blocks (layers L0-L8 of HopCIM).
Pool and FC are left out: their traffic is 64 values. Spatial mapping onto a 16 x 4 x 16 array (1,024 MACs, as a
64-row x 16-column tile): conv1 K16 x FX4 x FY10, depthwise G16 x FX3 x FY3, pointwise K16 x C16."""
P = {'O': 16, 'O_final': 8, 'W': 8, 'I': 8}
LINKS = {'O': 'O', 'W': 'I2', 'I': 'I1'}
workload = {
    -1: {'equation': 'input', 'loop_dim_size': {'B': 1, 'K': 1, 'OY': 49, 'OX': 10}, 'precision': 8,
         'core_allocation': 1, 'memory_operand_links': {'O': 'I1'}},
    0: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 64, 'C': 1, 'OY': 25, 'OX': 5, 'FY': 10, 'FX': 4},
        'operand_precision': dict(P), 'operand_source': {'W': [], 'I': [-1]}, 'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}}, 'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 16), 'D2': ('FX', 4), 'D3': ('FY', 10)}, 'memory_operand_links': dict(LINKS)},
}
for b in range(4):
    dw, pw = 1 + 2 * b, 2 + 2 * b
    workload[dw] = {'equation': 'O[b][g][oy][ox]+=W[g][fy][fx]*I[b][g][ix][iy]',
                    'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
                    'loop_dim_size': {'B': 1, 'G': 64, 'OY': 25, 'OX': 5, 'FY': 3, 'FX': 3},
                    'operand_precision': dict(P), 'operand_source': {'W': [], 'I': [dw - 1]}, 'constant_operands': ['W'],
                    'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'G': 'K'}}, 'core_allocation': 1,
                    'spatial_mapping': {'D1': ('G', 16), 'D2': ('FX', 3), 'D3': ('FY', 3)}, 'memory_operand_links': dict(LINKS)}
    workload[pw] = {'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ix][iy]',
                    'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
                    'loop_dim_size': {'B': 1, 'K': 64, 'C': 64, 'OY': 25, 'OX': 5, 'FY': 1, 'FX': 1},
                    'operand_precision': dict(P), 'operand_source': {'W': [], 'I': [dw]}, 'constant_operands': ['W'],
                    'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'G'}}, 'core_allocation': 1,
                    'spatial_mapping': {'D1': ('K', 16), 'D3': ('C', 16)}, 'memory_operand_links': dict(LINKS)}
