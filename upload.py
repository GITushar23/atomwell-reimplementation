#!/usr/bin/env python3
import time
import re
import json
import subprocess
from pathlib import Path

MODEL_NAME = "atomwell"
EPOCH_PATTERN = r"(\d+)"
STEP_PATTERN = r"(\d+)"

RCLONE_CONFIG = "/teamspace/studios/this_studio/bindwell/rclone.conf"
REMOTE_DIR = "gdrive:bindwelldata"
CHECK_INTERVAL = 5 * 60
RETRY_DELAY = 60

PATTERN = re.compile(rf"^{MODEL_NAME}_epoch_{EPOCH_PATTERN}_step_{STEP_PATTERN}\.pt$")

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
    result = subprocess.run(
        ["rclone", "--config", RCLONE_CONFIG, "lsjson", REMOTE_DIR],
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

        if epoch < best_epoch or (epoch == best_epoch and step < best_step):
            remote_path = f"{REMOTE_DIR}/{name}"
            print(f"[CLEANUP] Deleting {remote_path}")
            subprocess.run(
                ["rclone", "--config", RCLONE_CONFIG, "deletefile", remote_path],
                check=True,
            )

def upload_checkpoint(path):
    cmd = [
        "rclone", "--config", RCLONE_CONFIG, "copy", str(path), REMOTE_DIR,
        "--progress", "--transfers", "8", "--checkers", "8",
    ]
    print(f"[UPLOAD] Uploading {path}")
    subprocess.run(cmd, check=True)

def main_loop():
    last_uploaded_epoch = -1
    last_uploaded_step = -1

    while True:
        try:
            best, best_epoch, best_step = find_best_local_checkpoint()

            if best is None:
                print("[INFO] No .pt files found")
            else:
                if (best_epoch > last_uploaded_epoch or
                    (best_epoch == last_uploaded_epoch and best_step > last_uploaded_step)):

                    print(f"[INFO] Found {best} (epoch={best_epoch}, step={best_step})")
                    upload_checkpoint(best)
                    delete_older_remote_models(best_epoch, best_step)

                    last_uploaded_epoch = best_epoch
                    last_uploaded_step = best_step
                else:
                    print("[INFO] No new checkpoint")

            print(f"[SLEEP] Waiting {CHECK_INTERVAL // 60} minutes...")
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print(f"[ERROR] {e}")
            print(f"[RETRY] Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)

if __name__ == "__main__":
    main_loop()
