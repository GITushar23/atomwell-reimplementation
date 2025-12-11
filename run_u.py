import subprocess
import time
import sys

SCRIPT_TO_RUN = "upload.py"
RETRY_DELAY = 60  # seconds

while True:
    try:
        print("Running upload.py...")
        result = subprocess.run(
            [sys.executable, SCRIPT_TO_RUN],
            check=True
        )
        print("upload.py finished successfully.")
        break  # exit loop if no error

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(f"Retrying in {RETRY_DELAY} seconds...")
        time.sleep(RETRY_DELAY)

    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"Retrying in {RETRY_DELAY} seconds...")
        time.sleep(RETRY_DELAY)
