#!/usr/bin/env python3
import time
import re
import json
import subprocess
from pathlib import Path

RCLONE_CONFIG = "/teamspace/studios/this_studio/bindwell/rclone.conf"
REMOTE_DIR = "gdrive:bindwelldata"
PATTERN = re.compile(r"^atomwell_epoch_(\d+)_step_(\d+)\.pt$")

def find_best_local_checkpoint():
    best = None
    best_epoch = -1
    best_step = -1

    for f in Path(".").glob("*.pt"):
        m = PATTERN.match(f.name)
        if not m:
            continue
        epoch = int(m.group(1))
        step = int(m.group(2))

        if epoch > best_epoch or (epoch == best_epoch and step > best_step):
            best = f
            best_epoch = epoch
            best_step = step

    return best, best_epoch, best_step

def delete_older_remote_models(best_epoch, best_step):
    # List files in remote as JSON
    result = subprocess.run(
        [
            "rclone",
            "--config", RCLONE_CONFIG,
            "lsjson",
            REMOTE_DIR
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    files = json.loads(result.stdout)

    for f in files:
        name = f["Name"]
        m = PATTERN.match(name)
        if not m:
            continue

        epoch = int(m.group(1))
        step = int(m.group(2))

        # Delete only if strictly older than current best
        if epoch < best_epoch or (epoch == best_epoch and step < best_step):
            remote_path = f"{REMOTE_DIR}/{name}"
            print(f"[CLEANUP] Deleting remote file: {remote_path}")
            subprocess.run(
                [
                    "rclone",
                    "--config", RCLONE_CONFIG,
                    "deletefile",
                    remote_path,
                ],
                check=True,
            )

def upload_checkpoint(path):
    cmd = [
        "rclone",
        "--config", RCLONE_CONFIG,
        "copy",
        str(path),
        REMOTE_DIR,
        "--progress",
        "--transfers", "8",
        "--checkers", "8",
    ]
    print(f"[UPLOAD] Uploading {path} to {REMOTE_DIR}")
    subprocess.run(cmd, check=True)

def main_loop():
    last_uploaded_epoch = -1
    last_uploaded_step = -1

    while True:
        best, best_epoch, best_step = find_best_local_checkpoint()

        if best is None:
            print("[INFO] No matching .pt files found locally.")
        else:
            # Only upload if there's a newer best than last time
            if (best_epoch > last_uploaded_epoch or
                (best_epoch == last_uploaded_epoch and best_step > last_uploaded_step)):

                print(f"[INFO] New best found: {best} (epoch={best_epoch}, step={best_step})")
                upload_checkpoint(best)
                delete_older_remote_models(best_epoch, best_step)

                last_uploaded_epoch = best_epoch
                last_uploaded_step = best_step
            else:
                print("[INFO] No newer checkpoint than last uploaded.")

        # Sleep 5 minutes
        print("[SLEEP] Waiting 5 minutes...")
        time.sleep(2 * 60)

if __name__ == "__main__":
    main_loop()
