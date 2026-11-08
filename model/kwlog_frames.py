#!/usr/bin/env python3
"""Frames of the same-audio logger (firmware hopcim_kwlog, docs/SAME_AUDIO.md). Standard library only.

A5 <type> <payload> <Fletcher-16 over type + payload>. A byte stream that loses bytes or carries a
corrupted frame resynchronises at the next A5 whose checksum holds; `bad` counts what was skipped.
"""
import struct

KEYWORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
LABEL = KEYWORDS + ["_silence_", "_unknown_"]
OTHER = ["bird", "house", "happy", "seven", "wow", "dog", "cat", "tree", "marvin", "zero"]
LEN = {ord("H"): 7, ord("A"): 644, ord("D"): 19, ord("R"): 5, ord("P"): 8, ord("L"): 12, ord("M"): 10, ord("E"): 9}
SYNC = 0xA5


def fletcher(data):
    a = b = 0
    for x in data:
        a = (a + x) % 255
        b = (b + a) % 255
    return a, b


class Parser:
    def __init__(self):
        self.buf = bytearray()
        self.bad = 0                                  # bytes skipped between valid frames
        self.bad_start = 0                            # bytes skipped before the first valid frame (line settling)
        self.frames = 0

    def feed(self, data):
        """Append bytes; yield (type character, payload bytes) for every complete valid frame."""
        self.buf += data
        b, i, n = self.buf, 0, len(self.buf)
        while i < n:
            if b[i] != SYNC:
                i += 1
                self._skip()
                continue
            if i + 1 >= n:
                break
            ln = LEN.get(b[i + 1])
            if ln is None:
                i += 1
                self._skip()
                continue
            if i + 2 + ln + 2 > n:
                break
            body = bytes(b[i + 1: i + 2 + ln])
            if fletcher(body) == (b[i + 2 + ln], b[i + 3 + ln]):
                self.frames += 1
                yield chr(b[i + 1]), body[1:]
                i += 4 + ln
            else:
                i += 1
                self._skip()
        del b[:i]

    def _skip(self):
        if self.frames:
            self.bad += 1
        else:
            self.bad_start += 1


def decode(t, p):
    """A frame as a dict; samples and logits stay bytes-like for speed."""
    if t == "H":
        ver, audio, rounds, clk = struct.unpack("<BBBI", p)
        return dict(ver=ver, audio=audio, rounds=rounds, clk=clk)
    if t == "A":
        return dict(hop=struct.unpack_from("<I", p)[0], samples=p[4:])
    if t == "D":
        hop, cnt, c = struct.unpack_from("<IHB", p)
        return dict(hop=hop, cnt=cnt, cls=c & 15, ovr=c >> 4, logits=struct.unpack_from("<12b", p, 7))
    if t == "R":
        hop, c = struct.unpack("<IB", p)
        return dict(hop=hop, cls=c)
    if t == "P":
        hop, trial, kind, word = struct.unpack("<IHBB", p)
        name = KEYWORDS[word] if kind == 0 else (OTHER[word] if kind == 1 else "(quiet)")
        return dict(hop=hop, trial=trial, kind=("keyword", "other", "quiet")[kind], word=name)
    if t == "L":
        hop, mn, mx, mean, rms = struct.unpack("<IhhhH", p)
        return dict(hop=hop, min=mn, max=mx, mean=mean, rms=rms)
    if t == "M":
        hop, mx, mn, rms = struct.unpack("<IhhH", p)
        return dict(hop=hop, max=mx, min=mn, rms=rms)
    if t == "E":
        hop, lost, ovr = struct.unpack("<IIB", p)
        return dict(hop=hop, lost=lost, ovr=ovr)
    raise ValueError(t)
