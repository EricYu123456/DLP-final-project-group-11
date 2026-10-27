"""Download only pinned source files needed for the EmbedLLM experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests

from coldstart.data import EXPECTED_FILES, SOURCE_REPOSITORY, SOURCE_REVISION, verify_source_file
from coldstart.runtime import Deadline, register_job

MINIMAL = ("model_order.csv", "question_order.csv", "question_embeddings.pth",
           "train_y.pth", "val_y.pth", "test_y.pth")


def download_file(name: str, destination: Path, deadline: Deadline,
                  reuse_dir: Path | None = None, retries: int = 4) -> dict:
    deadline.check()
    target = destination / name
    expected = EXPECTED_FILES[name]
    if target.is_file() and target.stat().st_size == expected["size"]:
        checksum = verify_source_file(target, deadline.check)
        print(f"verified existing {name}", flush=True)
        return {"file": name, "sha256": checksum, "bytes": target.stat().st_size, "method": "existing"}
    if reuse_dir is not None and (reuse_dir / name).is_file():
        source = reuse_dir / name
        checksum = verify_source_file(source, deadline.check)
        deadline.check()
        shutil.copyfile(source, target)
        print(f"reused verified {name}", flush=True)
        return {"file": name, "sha256": checksum, "bytes": target.stat().st_size, "method": "verified_copy"}
    partial = destination / f"{name}.partial"
    last_error: Exception | None = None
    for attempt in range(retries):
        deadline.check()
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected["size"]:
            raise ValueError(f"Oversize partial file: {partial}")
        if offset == expected["size"]:
            partial.replace(target)
            checksum = verify_source_file(target, deadline.check)
            return {"file": name, "sha256": checksum, "bytes": target.stat().st_size, "method": "download"}
        # Query nonce avoids stale signed redirects; content remains revision-pinned.
        url = (f"https://huggingface.co/datasets/{SOURCE_REPOSITORY}/resolve/"
               f"{SOURCE_REVISION}/{name}?download=true&attempt={attempt}&offset={offset}")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            remaining = max(1.0, deadline.remaining_seconds())
            with requests.get(url, headers=headers, stream=True,
                              timeout=(min(20.0, remaining), min(30.0, remaining))) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0
                if response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-") or not content_range.endswith(f"/{expected['size']}"):
                        raise ValueError(f"Unexpected byte range for {name}: {content_range}")
                mode = "ab" if offset else "wb"
                with partial.open(mode) as output:
                    saved = offset
                    report_at = saved + 128 * 1024 * 1024
                    for block in response.iter_content(1024 * 1024):
                        deadline.check()
                        if not block:
                            continue
                        saved += len(block)
                        if saved > expected["size"]:
                            raise ValueError(f"Downloaded too many bytes for {name}")
                        output.write(block)
                        if saved >= report_at:
                            print(f"{name}: {saved:,}/{expected['size']:,} bytes", flush=True)
                            report_at = saved + 128 * 1024 * 1024
            if partial.stat().st_size != expected["size"]:
                raise IOError(f"Incomplete response for {name}: {partial.stat().st_size}")
            partial.replace(target)
            checksum = verify_source_file(target, deadline.check)
            print(f"downloaded and verified {name}", flush=True)
            return {"file": name, "sha256": checksum, "bytes": target.stat().st_size,
                    "method": "download", "url": url.split("?")[0]}
        except (requests.RequestException, IOError) as error:
            last_error = error
            print(f"{name}: attempt {attempt + 1} failed: {error}", flush=True)
            deadline.check()
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, max(0.0, deadline.remaining_seconds())))
    raise RuntimeError(f"Download failed for {name}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--reuse-dir", type=Path, default=ROOT / "tmp" / "research" / "agent_a")
    parser.add_argument("--full-csv", action="store_true", help="Also acquire the 3.26 GB source CSVs for a complete audit")
    parser.add_argument("--files", nargs="+", choices=tuple(EXPECTED_FILES))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--deadline", default=None, help="Optional ISO-8601 stop time, including timezone")
    args = parser.parse_args()
    deadline = Deadline(args.deadline)
    names = args.files or list(MINIMAL) + (["train.csv", "val.csv", "test.csv"] if args.full_csv else [])
    if args.workers < 1 or args.workers > 3:
        parser.error("--workers must be between 1 and 3")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with register_job(deadline=args.deadline, metadata={"task": "pinned_data_download", "files": names}):
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_file, name, args.output_dir, deadline, args.reuse_dir) for name in names]
            records = [future.result() for future in futures]
        deadline.check()
        manifest = {"schema_version": 1, "repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION,
                    "completed_at": datetime.now(timezone.utc).isoformat(), "deadline": args.deadline,
                    "files": records}
        (args.output_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Pinned data download complete", flush=True)


if __name__ == "__main__":
    main()
