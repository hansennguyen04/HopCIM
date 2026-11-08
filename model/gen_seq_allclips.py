#!/usr/bin/env python3
"""Vectors for the 4,890-clip run of tb_cim_seq (+define+ALLCLIPS, sim/run_seq_allclips.sh), in tb/seq_vec_all/
(not in git; regenerated from the dataset, pinned by hash in the run's log):
  feat.hex    FEAT words of every GSC test clip (integer CIM-MFCC, mfcc_int.cim_mfcc_int), 49 rows x 3 words each
  logits.hex  golden int8 logits (dscnn_int.forward = TFLite reference kernels), 12 per clip
  meta.hex    number of clips, the SEQ_ZP word, the golden class of each clip
  SHA256SUMS  hashes of the three files
Clip order: sorted file paths, as streamfuse_sched.test_clips. Usage: gen_seq_allclips.py [workers]"""
import glob
import hashlib
import os
import sys
from multiprocessing import Pool

import numpy as np

import dscnn_int as d
import exact_reuse as er
import mfcc_int as mi
from gen_seq_vectors import feat_words, write, zp_word
import gen_seq_vectors as gsv

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tb", "seq_vec_all")


def one(p):
    q = mi.cim_mfcc_int(er.read_wav(p)).reshape(49, 10)
    y = d.forward(q)
    return feat_words(q), [int(v) for v in y]


def main(workers=8):
    fs = sorted(glob.glob(os.path.join(er.TEST, "*", "*.wav")))
    with Pool(workers) as pool:
        res = pool.map(one, fs, chunksize=32)
    os.makedirs(OUT, exist_ok=True)
    gsv.OUT = OUT
    write("feat.hex", [w for f, _ in res for w in f])
    write("logits.hex", [v for _, y in res for v in y], 2)
    write("meta.hex", [len(fs), zp_word()] + [int(np.argmax(y)) for _, y in res])
    with open(os.path.join(OUT, "SHA256SUMS"), "w") as f:
        for n in ("feat.hex", "logits.hex", "meta.hex"):
            f.write(f"{hashlib.sha256(open(os.path.join(OUT, n), 'rb').read()).hexdigest()}  {n}\n")
    print(f"{len(fs)} clips -> {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
