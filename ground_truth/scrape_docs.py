import json
import os
from pathlib import Path

# Relative path to the English documentation inside the cloned repo (can be adjusted to a subfolder if needed)
DOCS_DIR = Path("./nopCommerce-Docs/en") # fetches 209 markdown pages & parses
BASE_URL = "https://docs.nopcommerce.com/en"

DOCS_DIR = Path("./nopCommerce-Docs/en/running-your-store/order-management") # fetches 9 markdown pages & parses
BASE_URL = "https://docs.nopcommerce.com/en/running-your-store/order-management"

OUTPUT_FILE = "nopcommerce_docs_raw.json"

def parse_local_docs():
    if not DOCS_DIR.exists():
        print(f"Error: Cannot find directory {DOCS_DIR}. Please ensure that you have cloned the nopCommerce-Docs repo.")
        return

    print(f"Scanning local documentation at: {DOCS_DIR}")
    scraped_data = []

    # Recursively find all markdown files
    md_files = list(DOCS_DIR.glob("**/*.md"))
    print(f"Found {len(md_files)} markdown pages to parse.")

    for index, file_path in enumerate(md_files, 1):
        try:
            # Calculate the relative path to reconstruct the online URL
            rel_path = file_path.relative_to(DOCS_DIR)
            url_path = str(rel_path).replace(".md", ".html")
            live_url = f"{BASE_URL}/{url_path}"

            # Read the raw markdown content
            with open(file_path, "r", encoding="utf-8") as f:
                raw_markdown = f.read()

            # Format the title based on file name
            title = file_path.stem.replace("-", " ").title()
            clean_text = clean_markdown_content(raw_markdown)

            scraped_data.append({
                "url": live_url,
                "title": title,
                "raw_text": clean_text
            })

            if index % 50 == 0:
                print(f"Processed {index}/{len(md_files)} files...")

        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    # Save to the JSON structure
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, indent=2, ensure_ascii=False)

    print(f"Done! Parsed {len(scraped_data)} pages locally into {OUTPUT_FILE}")

def clean_markdown_content(text):
    """Optional: Basic cleanup of markdown syntax"""
    # Markdown headers (#, ##) are beneficial for LLM context, so keeping lines mostly intact.
    lines = [line.strip() for line in text.split("\n")]
    # Filter out empty lines to keep the file compact
    return "\n".join([line for line in lines if line])


if __name__ == "__main__":
    parse_local_docs()