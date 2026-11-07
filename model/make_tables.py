#!/usr/bin/env python3
"""Freeze the integer tables of the CIM-MFCC front-end (R4a) into mfcc_tables.npz.

The tables are those of the gate-2 datapath (kws_gate/gate2_eval.py, frozen), except the DCT:
its int16 coefficients are scaled to max |c| = 127 * 256 so that every coefficient splits into
two int8 CIM weights, c = 256 * hi + lo. Needs TensorFlow (mel matrix); run in ~/venvs/kwsgate.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "kws_gate"))
import gate2_eval as g  # noqa: E402

DCT_S = 127.0 * 256.0 / np.abs(g.DCT_TF[:, :g.NMFCC]).max()
DCTq = np.round(g.DCT_TF[:, :g.NMFCC] * DCT_S).astype(np.int64)                 # [40, 10]
lo = ((DCTq + 128) & 255) - 128                                                    # signed low byte
hi = (DCTq - lo) >> 8
assert np.all(np.abs(hi) <= 127) and np.all(256 * hi + lo == DCTq)

np.savez(os.path.join(HERE, "mfcc_tables.npz"),
         M1q=g.M1q, TWr=g.TWr, TWi=g.TWi, M3q=g.M3q, BIN=g.BIN, VALID=g.VALID, HANN_Q15=g.HANN_Q15,
         MELq=g.MELq, MEL_S=np.float64(g.MEL_S), LOG_LUT=g.LOG_LUT, DCTq=DCTq, DCT_S=np.float64(DCT_S))
print(f"MEL_S {g.MEL_S:.6f}  DCT_S {DCT_S:.6f} (gate-2 {g.DCT_S:.6f})  MELq nonzero {int(np.count_nonzero(g.MELq))}")
