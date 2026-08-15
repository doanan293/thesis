import concurrent.futures
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from corpus_pipeline.config.paths import RAW_ANKHANG_HTML_DIR, RESOURCES_ANKHANG_DIR

DEFAULT_BASE_DIR = RAW_ANKHANG_HTML_DIR
DEFAULT_URLS_FILE = RESOURCES_ANKHANG_DIR / "drug_urls.txt"


def download_page(url, base_dir=DEFAULT_BASE_DIR):
    base_dir = Path(base_dir)
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split("/") if p]
    if len(path_parts) < 2:
        category = "uncategorized"
        slug = path_parts[0] if path_parts else "index"
    else:
        category = path_parts[0]
        slug = path_parts[1]

    category_dir = base_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    file_path = category_dir / f"{slug}.html"

    # Skip if already downloaded
    if file_path.exists() and file_path.stat().st_size > 1000:
        return url, "skipped"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
    )

    try:
        # Respectful delay between requests
        time.sleep(0.2)
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read()
            with file_path.open("wb") as f:
                f.write(html)
        return url, "success"
    except Exception as e:
        return url, f"error: {e!s}"


def download_all_pages() -> None:
    urls_file = DEFAULT_URLS_FILE
    if not urls_file.exists():
        print(f"URLs file {urls_file} not found. Please run collect_urls.py first.")
        return

    with urls_file.open("r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    total_urls = len(urls)
    print(f"Starting download of {total_urls} drug pages...")

    success_count = 0
    skipped_count = 0
    error_count = 0

    # Use 10 threads to download in parallel. This is fast but doesn't overload the server.
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(download_page, url): url for url in urls}

        for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
            url = future_to_url[future]
            try:
                url, status = future.result()
                if status == "success":
                    success_count += 1
                elif status == "skipped":
                    skipped_count += 1
                else:
                    error_count += 1
                    print(f"\nFailed to download {url}: {status}")
            except Exception as e:
                error_count += 1
                print(f"\nThread exception for {url}: {e}")

            if (i + 1) % 50 == 0 or (i + 1) == total_urls:
                print(
                    f"Progress: {i + 1}/{total_urls} finished. (Success: {success_count}, Skipped: {skipped_count}, Error: {error_count})"
                )

    print("\nDownload completed!")
    print(f"Total: {total_urls}")
    print(f"Success: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")
