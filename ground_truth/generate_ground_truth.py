import csv
import json
import os
import sys
from pathlib import Path
from openai import OpenAI

# Input and Output file paths
INPUT_JSON_FILE = "nopcommerce_docs_raw.json"
OUTPUT_CSV_FILE = "ground_truth.csv"

# OpenAI API Key setup
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    print(
        "Error: OPENAI_API_KEY environment variable is not set. Please set it before running."
    )
    sys.exit(1)

client = OpenAI(api_key=api_key)


def generate_ground_truths_from_doc(doc_title, raw_text):
    """Uses OpenAI API to extract Area and Ground Truth pairs from a single document."""
    system_prompt = (
        "You are an expert technical writer and data annotator. "
        "Your task is to analyze product documentation and extract concise, atomic ground truth statements. "
        "For each key feature, setting, workflow, or rule mentioned in the text, extract a clear ground truth factual sentence. "
        "Assign an 'Area' label to each ground truth (e.g., 'Document Title - Subheading' or 'Document Title')."
    )

    user_prompt = f"""
Document Title: {doc_title}

Raw Documentation Text:
{raw_text}

Please extract ground truth statements from the above document.
Return your response ONLY as a JSON object with a key 'ground_truths' containing an array of objects.
Each object must have exactly two keys:
- "Area": A descriptive area or section name (e.g., "{doc_title} - Section Name")
- "Ground Truth": A concise, clear, factual summary statement of the feature, setting, or workflow.

Example JSON output format:
{{
  "ground_truths": [
    {{
      "Area": "Order Settings - Checkout",
      "Ground Truth": "The 'Checkout disabled' setting disables the entire checkout process."
    }}
  ]
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # You can switch to "gpt-4o" for higher complexity docs
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)
        return parsed.get("ground_truths", [])

    except Exception as e:
        print(f"Error processing document '{doc_title}': {e}")
        return []


def main():
    if not Path(INPUT_JSON_FILE).exists():
        print(f"Error: Could not find input file '{INPUT_JSON_FILE}'.")
        return

    print(f"Loading scraped documentation from '{INPUT_JSON_FILE}'...")
    with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)

    print(
        f"Found {len(docs)} documents. Beginning Ground Truth extraction via OpenAI..."
    )

    all_ground_truths = []

    for index, doc in enumerate(docs, 1):
        title = doc.get("title", "Unknown")
        raw_text = doc.get("raw_text", "")

        print(
            f"[{index}/{len(docs)}] Extracting ground truths for: {title}..."
        )

        if not raw_text.strip():
            print(f"  Skipping '{title}' (empty content).")
            continue

        extracted = generate_ground_truths_from_doc(title, raw_text)
        all_ground_truths.extend(extracted)
        print(f"  Extracted {len(extracted)} ground truth rows.")

    # Write output to CSV
    print(f"Writing results to '{OUTPUT_CSV_FILE}'...")
    with open(
        OUTPUT_CSV_FILE, "w", newline="", encoding="utf-8-sig"
    ) as csv_file:
        fieldnames = ["Area", "Ground Truth"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for row in all_ground_truths:
            writer.writerow(
                {
                    "Area": row.get("Area", ""),
                    "Ground Truth": row.get("Ground Truth", ""),
                }
            )

    print(
        f"Done! Successfully generated {len(all_ground_truths)} rows in '{OUTPUT_CSV_FILE}'."
    )


if __name__ == "__main__":
    main()