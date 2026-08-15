import concurrent.futures
import re
import urllib.request
from urllib.parse import urlparse

from corpus_pipeline.config.paths import RESOURCES_ANKHANG_DIR

ALL_URLS_PATH = RESOURCES_ANKHANG_DIR / "all_urls.txt"
DRUG_URLS_PATH = RESOURCES_ANKHANG_DIR / "drug_urls.txt"


def fetch_url(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def collect_drug_urls() -> None:
    print("Fetching main sitemap index...")
    main_sitemap_url = "https://www.nhathuocankhang.com/sitemap-sanpham.xml"
    xml_content = fetch_url(main_sitemap_url)
    if not xml_content:
        print("Failed to fetch main sitemap.")
        return

    sub_sitemaps = re.findall(r"<loc>(.*?)</loc>", xml_content)
    print(f"Found {len(sub_sitemaps)} sub-sitemaps.")

    all_urls = set()

    # Use ThreadPoolExecutor to fetch sub-sitemaps in parallel
    print("Fetching sub-sitemaps in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_url = {executor.submit(fetch_url, url): url for url in sub_sitemaps}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
            future_to_url[future]
            xml_sub = future.result()
            if xml_sub:
                urls_in_sub = re.findall(r"<loc>(.*?)</loc>", xml_sub)
                all_urls.update(urls_in_sub)
            if (i + 1) % 50 == 0 or (i + 1) == len(sub_sitemaps):
                print(
                    f"Progress: {i + 1}/{len(sub_sitemaps)} sub-sitemaps processed. Collected {len(all_urls)} URLs so far."
                )

    print(f"Total unique URLs collected: {len(all_urls)}")

    # Define drug/medicine related path keywords
    # Medicines on An Khang usually have paths like:
    # /thuoc-giam-dau-khang-viem/...
    # /thuoc-tiet-nieu-hoocmon/...
    # /thuoc-ve-than-kinh/...
    # /tim-mach-huyet-ap/...
    # /duong-tieu-hoa/...
    # /khang-sinh-khang-nam/...
    # /thuoc-ke-don/...
    # We can filter URLs by looking at the path segment.
    # Let's write out all URLs first, and also create a filtered list.

    RESOURCES_ANKHANG_DIR.mkdir(parents=True, exist_ok=True)

    # Write all URLs
    all_urls_list = sorted(all_urls)
    with ALL_URLS_PATH.open("w", encoding="utf-8") as f:
        for u in all_urls_list:
            f.write(u + "\n")
    print(f"Saved all URLs to {ALL_URLS_PATH}")

    # Filter drug-related URLs
    # Exclude obvious non-drug categories
    non_drug_categories = {
        "sua-rua-mat",
        "kem-chong-nang",
        "kem-duong-da",
        "tinh-chat-duong-da",
        "mat-na-cham-soc-da",
        "dau-goi-dau",
        "kem-danh-rang",
        "ban-chai-danh-rang",
        "nuoc-suc-mieng",
        "bang-ve-sinh",
        "bao-cao-su",
        "ta-cho-be",
        "sua-bot-cong-thuc",
        "khan-uot",
        "son-duong-moi",
        "sua-tam",
        "kem-tri-mun",
        "xit-khoang",
        "nuoc-hoa-hong",
        "tay-trang",
        "kem-duong-the",
        "kem-chong-muoi",
        "dau-xa",
        "gel-rua-tay",
        "khan-giay",
        "dung-dich-ve-sinh",
        "mieng-dan-mun",
        "mat-na",
        "sua-mat",
        "tay-te-bao-chet",
        "bong-tay-trang",
    }

    drug_urls = []
    for u in all_urls_list:
        parsed = urlparse(u)
        path_parts = [p for p in parsed.path.split("/") if p]
        if not path_parts:
            continue
        category = path_parts[0]

        # If category starts with 'thuoc-' or is a known medicine category
        is_drug = False
        if category.startswith("thuoc-") or category in {
            "tim-mach-huyet-ap",
            "duong-tieu-hoa",
            "khang-sinh-khang-nam",
            "tri-ho-hen-phe-quan",
            "thuoc-nho-mat-tai-mui-hong",
            "tri-giun-san",
            "thuoc-dieu-tri-ung-thu",
            "chong-di-ung",
            "thuoc-ke-don",
            "khang-nam-khang-virus",
            "thuoc-dung-ngoai-da",
            "giam-dau-ha-sot",
            "vitamin-va-khoang-chat",
        }:
            is_drug = True

        # Double check against non-drug categories just in case
        if category in non_drug_categories:
            is_drug = False

        if is_drug:
            drug_urls.append(u)

    with DRUG_URLS_PATH.open("w", encoding="utf-8") as f:
        for u in drug_urls:
            f.write(u + "\n")
    print(f"Filtered {len(drug_urls)} drug URLs and saved to {DRUG_URLS_PATH}")
