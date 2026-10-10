#!/usr/bin/env python3
"""Download eight real Catalog 1 FRB HDF5 files with resumable bounded requests.

Raw archives only. Run prepare_chime_pilot.py and review its annotations next.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

EVENTS = ["FRB20180725A", "FRB20180729A", "FRB20180810B", "FRB20180904A",
          "FRB20180909A", "FRB20180910A", "FRB20180911A", "FRB20180916A"]
BASE = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/AstroDataCitationDOI/CISTI.CANFAR/21.0007/data/waterfalls/data/"
MIB = 1024 * 1024
RESOLVED_URLS = {}


def request_range(url, start, stop):
    for attempt in range(5):
        try:
            req = urllib.request.Request(RESOLVED_URLS.get(url, url), headers={"Range": f"bytes={start}-{stop}",
                                                      "User-Agent": "Corvus-research-pilot/0.1"})
            with urllib.request.urlopen(req, timeout=30) as response:
                # Reuse the archive's public read redirect in memory. Re-resolving
                # it for every range can add tens of seconds to each small read.
                RESOLVED_URLS[url] = response.geturl()
                headers = dict(response.headers)
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or not content_range.startswith(f"bytes {start}-{stop}/"):
                    raise ValueError(f"Archive did not honor bounded range {start}-{stop}")
                body = response.read(stop - start + 2)
                if len(body) != stop - start + 1:
                    raise ValueError("Incomplete archive range")
                return body, headers
        except (OSError, ValueError) as exc:
            RESOLVED_URLS.pop(url, None)
            if attempt == 4:
                raise RuntimeError(f"Could not fetch {url} range {start}:{stop}") from exc
            time.sleep(min(attempt + 1, 4))


def get_header(headers, name):
    return next((v for k, v in headers.items() if k.lower() == name.lower()), None)


def download(event, output):
    name = event + "_waterfall.h5"
    url = BASE + name
    _, headers = request_range(url, 0, 0)
    total = int(get_header(headers, "Content-Range").rsplit("/", 1)[1])
    if not 1 < total <= 40 * MIB:
        raise ValueError(f"Unexpected file size for {event}: {total}")
    digest = get_header(headers, "Digest") or ""
    published_md5 = base64.b64decode(digest.split("md5=", 1)[1]).hex() if "md5=" in digest else None
    final, partial = output / name, output / (name + ".part")
    if final.exists():
        data = final.read_bytes()
        if len(data) != total or (published_md5 and hashlib.md5(data).hexdigest() != published_md5):
            raise ValueError(f"Existing archive does not match publisher metadata: {final}")
    else:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > total:
            raise ValueError(f"Partial file exceeds archive size: {partial}")
        with partial.open("ab") as handle:
            while offset < total:
                stop = min(offset + MIB, total) - 1
                body, _ = request_range(url, offset, stop)
                handle.write(body)
                handle.flush()
                offset += len(body)
                if offset // MIB % 4 == 0 or offset == total:
                    print(f"{event}: {offset}/{total} bytes", flush=True)
        data = partial.read_bytes()
        if published_md5 and hashlib.md5(data).hexdigest() != published_md5:
            raise ValueError(f"Publisher checksum mismatch: {partial}")
        partial.replace(final)
    print(f"Verified {event}", flush=True)
    return {"event": event, "url": url, "filename": name, "bytes": total,
            "sha256": hashlib.sha256(data).hexdigest(), "publisher_md5": published_md5,
            "publisher_md5_verified": published_md5 is not None,
            "last_modified": get_header(headers, "Last-Modified")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("data/raw/chime"))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if not 1 <= args.workers <= 8:
        p.error("workers must be between 1 and 8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        receipts = list(pool.map(lambda e: download(e, args.output_dir), EVENTS))
    catalog_url = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/AstroDataCitationDOI/CISTI.CANFAR/25.0066/data/table/chimefrbcat2.csv"
    prefix, _ = request_range(catalog_url, 0, 65535)
    catalog_file = args.output_dir / "chimefrbcat2.csv.prefix-64KiB"
    catalog_file.write_bytes(prefix)
    catalog_receipt = {"url": catalog_url, "bytes": len(prefix),
                       "sha256": hashlib.sha256(prefix).hexdigest(), "byte_range": "0-65535",
                       "purpose": "Known-repeater grouping lookup for the selected early Catalog 1 events"}
    (args.output_dir / "catalog-prefix-provenance.json").write_text(json.dumps(catalog_receipt, indent=2) + "\n")
    provenance = {"dataset": "CHIME/FRB Catalog 1 measured HDF5 waveforms", "not_pulsar_data": True,
                  "archive": "https://www.canfar.net/storage/list/AstroDataCitationDOI/CISTI.CANFAR/21.0007/data",
                  "paper": "https://arxiv.org/abs/2106.04352", "acquired_utc": datetime.now(timezone.utc).isoformat(),
                  "selection": "Eight small publicly available early observations; convenience feasibility sample",
                  "files": receipts, "catalog_prefix": catalog_receipt,
                  "total_waveform_bytes": sum(r["bytes"] for r in receipts)}
    (args.output_dir / "acquisition.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Raw acquisition complete. Prepare and visually review source windows before training.")


if __name__ == "__main__":
    main()
