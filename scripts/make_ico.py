#!/usr/bin/env python3
# Build a multi-resolution Windows .ico from a source PNG using sips (no PIL).
# Entries are PNG-compressed (valid on Vista+); sizes 16..256.
import os
import struct
import subprocess
import sys
import tempfile

SIZES = [16, 24, 32, 48, 64, 128, 256]


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: make_ico.py <src.png> <out.ico>")
    src, out = sys.argv[1], sys.argv[2]

    images = []
    with tempfile.TemporaryDirectory() as td:
        for s in SIZES:
            p = os.path.join(td, f"{s}.png")
            subprocess.run(
                ["sips", "-z", str(s), str(s), src, "--out", p],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            with open(p, "rb") as f:
                images.append((s, f.read()))

    header = struct.pack("<HHH", 0, 1, len(images))  # reserved, type=icon, count
    offset = 6 + 16 * len(images)
    entries = b""
    blob = b""
    for s, png in images:
        dim = 0 if s >= 256 else s  # 0 means 256 in the ICONDIRENTRY byte
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(png), offset)
        blob += png
        offset += len(png)

    with open(out, "wb") as f:
        f.write(header + entries + blob)
    print(f"wrote {out} ({len(images)} sizes: {', '.join(map(str, SIZES))})")


if __name__ == "__main__":
    main()
