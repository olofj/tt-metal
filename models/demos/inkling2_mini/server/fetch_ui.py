# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Download Llama Chat, llama.cpp's web UI (MIT, (c) ggml authors), as prebuilt static files from npm.

    python -m models.demos.inkling2_mini.server.fetch_ui [--dir DIR]

Pinned to one release, checked by sha256. Writes DIR/dist (the app, what llama-server serves) and DIR/LICENSE
(its MIT notice, kept next to the files). The server uses the default DIR when it holds the files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

VERSION = "0.1.0"
URL = f"https://registry.npmjs.org/@ggml-org/llama-chat/-/llama-chat-{VERSION}.tgz"
SHA256 = "9e3702481e52ce3771238b2c6c9224e004c537753e58b1cdf7420b64f67a67ea"


def default_root():
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "inkling2-mini" / f"llama-chat-{VERSION}"


def default_dir():
    return default_root() / "dist"


def fetch(root):
    root = Path(root)
    data = urllib.request.urlopen(URL, timeout=120).read()
    got = hashlib.sha256(data).hexdigest()
    if got != SHA256:
        raise SystemExit(f"sha256 mismatch for {URL}: {got}, want {SHA256}")
    tmp = root.with_name(root.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name == "package/LICENSE" or m.name.startswith("package/dist/")]
        tf.extractall(tmp, members=members, filter="data")
    shutil.rmtree(root, ignore_errors=True)
    (tmp / "package").rename(root)
    tmp.rmdir()
    return root


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=default_root(), help=f"target (default {default_root()})")
    a = ap.parse_args()
    root = fetch(a.dir)
    print(f"Llama Chat {VERSION} in {root / 'dist'} (license: {root / 'LICENSE'})")
    print(f"start the server with --ui-dir {root / 'dist'} (or nothing, for the default dir)")


if __name__ == "__main__":
    main()
