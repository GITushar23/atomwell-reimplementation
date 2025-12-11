#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

RCLONE_CONFIG = "/teamspace/studios/this_studio/bindwell/rclone.conf"
REMOTE_DIR = "gdrive:bindwelldata"
MODEL_FILE = "atomwell_epoch_0_step_40150.pt"
CHUNKS_TO_DELETE = ["chunk_031_*.pkl", "chunk_030_*.pkl", "chunk_029_*.pkl", "chunk_028_*.pkl", "chunk_027_*.pkl"]
MAX_WORKERS = 3
RCLONE_TRANSFERS = "8"
RCLONE_CHECKERS = "8"

def run(cmd, cwd=None, shell=False):
    print(f"\n[RUN] {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=cwd, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}")
        raise

def task_flash_attention():
    run(["conda", "install", "-c", "conda-forge", "flash-attn", "-y"])
    print("\n[FLASH-ATTENTION] Completed")

def task_bindwell_rclone():
    home = Path.home()

    run(["unzip", "bindwell.zip"], cwd=home)

    run([
        "rclone", "--config", RCLONE_CONFIG, "copy",
        f"{REMOTE_DIR}/chunks", "./data/chunks",
        "--progress", "--transfers", RCLONE_TRANSFERS, "--checkers", RCLONE_CHECKERS,
    ], cwd=home)

    chunks_pattern = " ".join(CHUNKS_TO_DELETE)
    run(
        ["bash", "-lc", f"cd data/chunks/proteins && rm -f {chunks_pattern}"],
        cwd=home,
    )

    print("\n[BINDWELL + RCLONE] Completed")

def task_download_atomwell():
    run([
        "rclone", "--config", RCLONE_CONFIG, "copy",
        f"{REMOTE_DIR}/{MODEL_FILE}", ".",
        "--progress",
    ])

    print("\n[ATOMWELL MODEL] Completed")

def task_install_rclone():
    run("curl https://rclone.org/install.sh | sudo bash", shell=True)
    print("\n[RCLONE] Completed")


def main():
    task_install_rclone()
    tasks = {
        "flash_attention": task_flash_attention,
        "bindwell_rclone": task_bindwell_rclone,
        "download_atomwell": task_download_atomwell,
    }

    errors = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_name = {
            executor.submit(func): name
            for name, func in tasks.items()
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                future.result()
            except Exception as e:
                print(f"\n[FAILED] {name}: {e}")
                errors.append(name)

    if errors:
        print(f"\nFailed tasks: {', '.join(errors)}")
        sys.exit(1)
    else:
        print("\nAll tasks completed successfully")

if __name__ == "__main__":
    main()
