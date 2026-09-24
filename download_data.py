import os
import sys
import time
import urllib.request
import logging

from src.config import DATA_DIR, DATASET_FILES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def download_file(filename: str, url: str) -> str:
    """Download a single parquet file from Hugging Face dataset repository."""
    dest_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        logger.info(f"[EXISTS] {filename} already exists ({size_mb:.2f} MB). Skipping download.")
        return dest_path
    
    logger.info(f"[DOWNLOADING] {filename} from {url} ...")
    start_time = time.time()
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
        total_size = response.getheader('Content-Length')
        total_size = int(total_size) if total_size else None
        
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = (downloaded / total_size) * 100
                print(f"  Downloaded {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="\r")
            else:
                print(f"  Downloaded {downloaded / (1024*1024):.1f} MB", end="\r")
    
    elapsed = time.time() - start_time
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    logger.info(f"[DONE] Saved {filename} ({size_mb:.2f} MB) in {elapsed:.1f}s.")
    return dest_path

def download_all_data():
    """Download all required dataset parquet files."""
    for fname, url in DATASET_FILES.items():
        download_file(fname, url)
    logger.info("All dataset files verified/downloaded successfully.")

if __name__ == "__main__":
    download_all_data()
