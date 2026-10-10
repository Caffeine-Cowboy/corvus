#!/usr/bin/env python3
"""Measure actual HTTP prediction round trips against a running local service."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import httpx
import numpy as np


async def run(url, requests, concurrency):
    async with httpx.AsyncClient(base_url=url, timeout=30) as client:
        summary = (await client.get("/api/v1/summary")).json()
        allowed = [m for m in summary["models"] if m["gate"]["passed"]]
        if not allowed:
            raise RuntimeError("No model passes serving gates; refusing to bypass the gate")
        model = next((m for m in allowed if m["runtime"] == "onnx"), allowed[0])
        records = (await client.get("/api/v1/records")).json()
        replay = await client.post("/api/v1/replay", json={"record_id": records[0]["id"], "model_id": model["id"], "corruption": "clean"})
        replay.raise_for_status()
        data = replay.json()
        payload = {"signal": data["clean_signal"], "sample_rate_hz": data["record"]["sample_rate_hz"], "model_id": model["id"]}
        for _ in range(10):
            response = await client.post("/api/v1/predict", json=payload)
            response.raise_for_status()
        semaphore = asyncio.Semaphore(concurrency)

        async def one():
            async with semaphore:
                start = time.perf_counter()
                response = await client.post("/api/v1/predict", json=payload)
                elapsed = (time.perf_counter() - start) * 1000
                return elapsed, response.status_code

        start = time.perf_counter()
        results = await asyncio.gather(*(one() for _ in range(requests)))
        elapsed = time.perf_counter() - start
        milliseconds = [r[0] for r in results]
        return {"created_utc": datetime.now(timezone.utc).isoformat(), "model": model["version"],
                "requests": requests, "concurrency": concurrency, "warmups": 10,
                "successful_requests": sum(status == 200 for _, status in results),
                "p50_ms": float(np.median(milliseconds)), "p95_ms": float(np.quantile(milliseconds, .95)),
                "p99_ms": float(np.quantile(milliseconds, .99)), "requests_per_second": requests / elapsed,
                "scope": "Local HTTP round trips: JSON serialization, validation, preprocessing, inference, response; connection pool reused; queue wait before semaphore excluded",
                "payload": "One repeated 10-second real ECG; not a varied production traffic workload"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("reports/lab-http.json"))
    args = p.parse_args()
    if not 1 <= args.requests <= 10000 or not 1 <= args.concurrency <= 32:
        p.error("requests must be 1..10000 and concurrency 1..32")
    report = asyncio.run(run(args.url, args.requests, args.concurrency))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
