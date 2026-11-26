#!/usr/bin/env python3
"""Vectors of ResNet8 for cim_gseq: tiles, parameters, descriptors, images and expected outputs (docs/NET2_GATES.md).
  python gen_net2_vectors.py [n_images] [first_image] [out_dir]     (default 20 images, tb/net2_vec)
The CIFAR-10 images are the first test images; golden values from model/resnet_int.py.
"""
import os
import sys

import numpy as np

from resnet_int import PARAMS, TOPO, WIRE, cifar, forward
from resnet_map import Net

HERE = os.path.dirname(os.path.abspath(__file__))
CONV_L = ["c0", "c1", "c2", "c4", "c5", "c6", "c8", "c9", "c10", "fc"]            # parameter layers 0 .. 9
CIN = {"c0": 0, "c1": 1, "c2": 1, "c4": 1, "c5": 2, "c6": 1, "c8": 2, "c9": 3, "c10": 2, "fc": 3}    # code (3: 64, 2: 32, 1: 16, 0: image)
# op: (type, k3, s2, cin, log2H - 3, src, srcB, dst) ; bank 3 = image
OPS = [("c0", 0, "conv", 3, 0), ("c1", 1, "conv", 0, 1), ("c2", 2, "conv", 1, 2), ("a3", 3, "add", 0, 0), ("c4", 4, "conv", 0, 1),
       ("c5", 5, "conv", 1, 2), ("c6", 6, "conv", 0, 1), ("a7", 7, "add", 1, 0), ("c8", 8, "conv", 0, 1), ("c9", 9, "conv", 1, 2),
       ("c10", 10, "conv", 0, 1), ("a11", 11, "add", 1, 0), ("pool", 12, "pool", 0, 0), ("fc", 13, "fc", 0, 0)]
# src, dst banks per op (index as in OPS): (srcA, srcB, dst)
BANKS = {"c0": (3, 0, 0), "c1": (0, 0, 1), "c2": (1, 0, 2), "a3": (0, 2, 0), "c4": (0, 0, 1), "c5": (1, 0, 2), "c6": (0, 0, 1),
         "a7": None, "c8": (0, 0, 1), "c9": (1, 0, 2), "c10": (0, 0, 1), "a11": None, "pool": (0, 0, 0), "fc": (0, 0, 0)}
HLOG = {"c0": 2, "c1": 2, "c2": 2, "a3": 2, "c4": 2, "c5": 1, "c6": 2, "a7": 1, "c8": 1, "c9": 0, "c10": 1, "a11": 0}
STRIDE2 = {"c4": 1, "c6": 1, "c8": 1, "c10": 1}
K3 = {"c0": 1, "c1": 1, "c2": 1, "c4": 1, "c5": 1, "c6": 0, "c8": 1, "c9": 1, "c10": 0}
OUTC = {"c0": 16, "c1": 16, "c2": 16, "a3": 16, "c4": 32, "c5": 32, "c6": 32, "a7": 32, "c8": 64, "c9": 64, "c10": 64, "a11": 64}


def u32(x):
    return int(x) & 0xFFFFFFFF


def descriptors(net):
    P = net.P
    out = []
    for idx, (n, _, kind, *_rest) in enumerate(OPS):
        w = [0] * 8
        if kind in ("conv", "fc"):
            name = n
            b, R, C, rows, outs = net.base[name]
            relu = int(P[f"{name}.relu"])
            if kind == "conv":
                src, dst = BANKS[n][0], BANKS[n][2]
                w0 = 0 | (K3[n] << 2) | (STRIDE2.get(n, 0) << 3) | (CIN[n] << 4) | (HLOG[n] << 6)
            else:
                src, dst = 0, 0
                w0 = 3 | (3 << 4)
            w0 |= (src << 8) | (dst << 10) | (relu << 12) | ((outs - 1) << 13) | (R << 19) | (C << 23) | (CONV_L.index(n) << 26)
            w[0] = w0
            w[1] = (b & 0x7F) | ((int(P[f"{name}.zi"]) & 0xFF) << 8) | ((int(P[f"{name}.zo"]) & 0xFF) << 16)
        elif kind == "add":
            z1, z2, q1, s1, q2, s2, qo, so = [int(v) for v in P[f"{n}.add"]]
            src = [str(s) for s in P[f"{n}.src"]]
            banks = {"c0": 0, "c2": 2, "c5": 2, "c6": 1, "c9": 2, "c10": 1}
            sA, sB = banks[src[0]], banks[src[1]]
            w[0] = 1 | (HLOG[n] << 6) | (sA << 8) | (0 << 10) | (int(P[f"{n}.relu"]) << 12) | ((OUTC[n] - 1) << 13) | (sB << 30)
            w[0] |= 0 << 0
            w[0] = (w[0] & ~(3 << 10)) | (0 << 10)          # destination bank 0
            w[1] = ((z1 & 0xFF) << 8) | ((int(P[f"{n}.zo"]) & 0xFF) << 16) | ((z2 & 0xFF) << 24)
            w[2], w[3], w[4] = q1, q2, qo
            w[5] = (-s1 & 31) | ((-s2 & 31) << 5) | ((-so & 31) << 10)
        else:                                                # pool: source bank 0
            w[0] = 2 | (0 << 8)
        out.append(w)
    return out


def main():
    nimg = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    od = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "..", "tb", "net2_vec")
    os.makedirs(od, exist_ok=True)
    net = Net()
    P = net.P
    with open(os.path.join(od, "tiles.hex"), "w") as f:
        for t in net.tiles:
            t8 = t.astype(np.int8).astype(np.uint8)
            for r in range(64):
                for q in range(4):
                    f.write("%08x\n" % int.from_bytes(bytes(t8[r, 4 * q:4 * q + 4].tolist()), "little"))
    prm = np.zeros((640, 3), np.int64)
    for L, n in enumerate(CONV_L):
        for c in range(P[f"{n}.b"].shape[0]):
            prm[64 * L + c] = (P[f"{n}.b"][c], P[f"{n}.q"][c], -P[f"{n}.sh"][c])
    with open(os.path.join(od, "params.hex"), "w") as f:
        for e in prm:
            for v in e:
                f.write("%08x\n" % u32(v))
    with open(os.path.join(od, "desc.hex"), "w") as f:
        for w in descriptors(net):
            for v in w:
                f.write("%08x\n" % u32(v))
    x, lab = cifar(10000)
    x = x[first:first + nimg]
    v = forward(x, net.P, keep=True)
    with open(os.path.join(od, "img.hex"), "w") as f:
        for i in range(len(x)):
            for y in range(32):
                for xx in range(32):
                    px = x[i, y, xx].astype(np.uint8)
                    f.write("%06x\n" % (int(px[0]) | int(px[1]) << 8 | int(px[2]) << 16))
    sizes = []
    with open(os.path.join(od, "exp.hex"), "w") as f:
        for i in range(len(x)):
            for _, n, _ in TOPO:
                if n in ("p12", "fc"):
                    continue
                f.write("\n".join("%02x" % (int(b) & 255) for b in v[n][i].reshape(-1)) + "\n")
    with open(os.path.join(od, "logits.hex"), "w") as f:
        for i in range(len(x)):
            f.write("\n".join("%02x" % (int(b) & 255) for b in v["fc"][i]) + "\n")
    with open(os.path.join(od, "pool.hex"), "w") as f:
        for i in range(len(x)):
            f.write("\n".join("%02x" % (int(b) & 255) for b in v["p12"][i]) + "\n")
    with open(os.path.join(od, "cls.hex"), "w") as f:
        for i in range(len(x)):
            f.write("%02x\n" % int(np.argmax(v["fc"][i])))
    json_meta = dict(images=len(x), first=first, n_ops=len(OPS))
    print("vectors:", json_meta, "in", od)


if __name__ == "__main__":
    main()
