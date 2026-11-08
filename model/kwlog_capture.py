#!/usr/bin/env python3
"""Capture the same-audio logger's UART stream (docs/SAME_AUDIO.md). Standard library only.

Every received byte goes unchanged to <out>/frames.bin: that file is the evidence, and
kwlog_analyse.py reads it. The frames are also parsed as they arrive, to show the prompts and the
firmware's reports and to stop at the end frame. <out>/capture.json holds the totals.
Usage: kwlog_capture.py <out dir> [--port /dev/ttyUSB1] [--seconds N] [--ready-file F] [--quiet]
"""
import argparse
import json
import os
import select
import sys
import termios
import time
import tty

from kwlog_frames import KEYWORDS, Parser, decode

BAUD = termios.B921600


def open_port(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOCTTY)
    tty.setraw(fd)
    a = termios.tcgetattr(fd)
    a[4] = a[5] = BAUD
    a[2] = (a[2] | termios.CLOCAL | termios.CREAD) & ~(termios.CRTSCTS | termios.PARENB | termios.CSTOPB)
    a[0] &= ~(termios.IXON | termios.IXOFF | termios.IXANY)
    termios.tcsetattr(fd, termios.TCSANOW, a)
    termios.tcflush(fd, termios.TCIFLUSH)
    return fd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--port", default="/dev/ttyUSB1")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this time (0: at the end frame)")
    ap.add_argument("--first-frame-timeout", type=float, default=60)
    ap.add_argument("--ready-file", help="created once the port is open")
    ap.add_argument("--quiet", action="store_true", help="do not print prompts and reports")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fd = open_port(a.port)
    if a.ready_file:
        open(a.ready_file, "w").close()
    p, n = Parser(), {}
    t0 = time.monotonic()
    first = None
    t_last = t0
    end = None
    last_hop = 0
    lost_hops_a = lost_hops_d = 0
    next_a = next_d = None
    level_rms = []
    with open(os.path.join(a.out, "frames.bin"), "wb") as f:
        try:
            while True:
                now = time.monotonic()
                if first is None and now - t0 > a.first_frame_timeout:
                    print("no frame arrived: is the board running the logger?", file=sys.stderr)
                    break
                if first is not None and a.seconds and now - first >= a.seconds:
                    break
                r, _, _ = select.select([fd], [], [], 0.2)
                if not r:
                    if first is not None and now - first > 5 and not a.seconds and now - max(first, t_last) > 5:
                        print("the stream stopped without an end frame", file=sys.stderr)
                        break
                    continue
                data = os.read(fd, 65536)
                if not data:
                    continue
                f.write(data)
                t_last = time.monotonic()
                for t, pl in p.feed(data):
                    if first is None:
                        first = time.monotonic()
                    n[t] = n.get(t, 0) + 1
                    d = decode(t, pl)
                    if t == "A":
                        if next_a is not None and d["hop"] != next_a:
                            lost_hops_a += d["hop"] - next_a
                        next_a = d["hop"] + 1
                        last_hop = d["hop"]
                    elif t == "D":
                        if next_d is not None and d["hop"] != next_d:
                            lost_hops_d += d["hop"] - next_d
                        next_d = d["hop"] + 1
                        last_hop = d["hop"]
                    elif t == "L":
                        level_rms.append(d["rms"])
                    if not a.quiet:
                        if t == "H":
                            print(f"logger: audio {d['audio']}, rounds {d['rounds']}, clock {d['clk']} Hz")
                            if d["rounds"]:
                                print("Say each word once when it appears, 20-50 cm from the board; "
                                      "say nothing at (stay quiet). The first word comes in about 6 s.")
                        elif t == "P":
                            if d["kind"] == "keyword":
                                print(f"[{d['trial']}] say: {d['word']}", flush=True)
                            elif d["kind"] == "other":
                                print(f"[{d['trial']}] say: {d['word']} (not a keyword)", flush=True)
                            else:
                                print(f"[{d['trial']}] (stay quiet)", flush=True)
                        elif t == "R":
                            print(f"    >> {KEYWORDS[d['cls']]}", flush=True)
                    if t == "E":
                        end = d
                if end is not None:
                    break
        except KeyboardInterrupt:
            print("interrupted", file=sys.stderr)
    res = dict(frames=n, bad_bytes=p.bad, bad_start=p.bad_start, end=end, seconds=(time.monotonic() - first) if first else 0,
               last_hop=last_hop, hops_lost_audio=lost_hops_a, hops_lost_decisions=lost_hops_d,
               level_rms_min=min(level_rms) if level_rms else None, level_rms_max=max(level_rms) if level_rms else None)
    with open(os.path.join(a.out, "capture.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"captured {sum(n.values())} frames in {res['seconds']:.1f} s: {n}; {p.bad} bytes skipped between frames, {p.bad_start} before the first; "
          f"hops lost: audio {lost_hops_a}, decisions {lost_hops_d}; end {end}")
    return 0 if first is not None else 1


if __name__ == "__main__":
    sys.exit(main())
