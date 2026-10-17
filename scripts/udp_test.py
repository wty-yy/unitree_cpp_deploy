#!/usr/bin/env python3

import argparse
import socket
import struct
import time


PACKET_MAGIC = 0x4A504F53  # "JPOS"
PACKET_STRUCT = struct.Struct("<IIQ12f")


def parse_args():
    parser = argparse.ArgumentParser(description="Receive joint position packets over UDP.")
    parser.add_argument("--host", default="0.0.0.0", help="Local bind address")
    parser.add_argument("--port", type=int, default=15000, help="Local bind port")
    parser.add_argument("--timeout", type=float, default=0.5, help="Socket timeout in seconds")
    parser.add_argument("--quiet", action="store_true", help="Only print packet summaries")
    return parser.parse_args()


def main():
    args = parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(args.timeout)

    print(f"Listening on {args.host}:{args.port}")

    last_print = time.time()
    while True:
        try:
            data, addr = sock.recvfrom(1024)
        except socket.timeout:
            continue

        if len(data) != PACKET_STRUCT.size:
            print(f"skip {addr}: unexpected packet size {len(data)}")
            continue

        magic, seq, time_ns, *q = PACKET_STRUCT.unpack(data)
        if magic != PACKET_MAGIC:
            print(f"skip {addr}: bad magic 0x{magic:08x}")
            continue

        now = time.time()
        hz = 0.0 if now == last_print else 1.0 / max(now - last_print, 1e-9)
        last_print = now

        if args.quiet:
            print(f"seq={seq} t={time_ns} hz={hz:.1f}")
            continue

        q_str = " ".join(f"{v:+.4f}" for v in q)
        print(f"seq={seq:6d} t={time_ns} hz={hz:6.1f} q={q_str}")


if __name__ == "__main__":
    main()
