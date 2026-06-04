#!/usr/bin/env python3
"""Tiny static server for the WASM build with the COOP/COEP headers that
SharedArrayBuffer + pthreads require. Without them browsers refuse to share
buffers across workers, so the renderer hangs on "loading-workers"."""

import argparse
import http.server
import socketserver
from pathlib import Path

class CrossOriginIsolatedHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy",   "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control",                "no-store")
        super().end_headers()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--dir",  default="build/web")
    args = ap.parse_args()
    serve_dir = Path(args.dir).resolve()

    class Handler(CrossOriginIsolatedHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(serve_dir), **kw)

    with socketserver.ThreadingTCPServer((args.bind, args.port), Handler) as httpd:
        httpd.allow_reuse_address = True
        print(f"Serving {serve_dir} on http://{args.bind}:{args.port}/  (COOP/COEP enabled)")
        httpd.serve_forever()

if __name__ == "__main__":
    main()
