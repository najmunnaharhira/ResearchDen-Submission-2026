"""Generate a provenance-tracked Banglish AI-review dataset with local Ollama.

The output is intentionally labelled ``ollama_llama3``. It must not be
represented as Claude, ChatGPT, or Gemini data.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


MODEL = "llama3:latest"
PROMPT_VERSION = "banglish-ecommerce-v1"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DOMAINS = [
    "electronics", "clothing", "beauty", "home", "groceries",
    "mobile accessories", "shoes", "delivery", "customer service",
]


def request_batch(batch_size: int, batch_number: int) -> list[str]:
    domains = random.Random(42000 + batch_number).sample(DOMAINS, k=5)
    prompt = f"""
Return one JSON object with exactly one key named "reviews". Its value must be
an array of exactly {batch_size} distinct fictional e-commerce product reviews.

Requirements:
- Write natural Banglish: Bengali expressed with Latin letters, optionally
  mixed with ordinary English product words.
- Use only Latin characters, numbers, spaces, emoji, and punctuation. Do not
  use Bengali script.
- Each review must contain 5 to 45 words.
- Mix positive, negative, and neutral experiences.
- Vary spelling, punctuation, length, tone, and formality.
- Mention products, packaging, delivery, seller behaviour, size, price,
  defects, returns, or quality across these areas: {', '.join(domains)}.
- Do not number the reviews. Do not include explanations or labels.
- Avoid repeating sentence templates or opening phrases.
""".strip()
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.95,
            "top_p": 0.9,
            "seed": 42000 + batch_number,
            "num_predict": 800,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        outer = json.loads(response.read().decode("utf-8"))
    parsed = json.loads(outer["response"])
    reviews = parsed.get("reviews", [])
    if not isinstance(reviews, list):
        return []
    return [str(item).strip() for item in reviews if isinstance(item, str)]


def valid(text: str) -> bool:
    words = text.split()
    return 5 <= len(words) <= 60 and not any("\u0980" <= char <= "\u09ff" for char in text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    if args.output.exists():
        with args.output.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if valid(row.get("text", "")) and row["text"].lower() not in seen:
                    rows.append(row)
                    seen.add(row["text"].lower())
        print(f"Resuming with {len(rows)} valid existing reviews.", flush=True)

    batch_number = len(rows) // max(args.batch_size, 1)
    consecutive_failures = 0
    while len(rows) < args.target:
        batch_number += 1
        try:
            generated = request_batch(args.batch_size, batch_number)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            consecutive_failures += 1
            print(f"Batch {batch_number} failed: {exc}", flush=True)
            if consecutive_failures >= 5:
                raise RuntimeError("Five consecutive Ollama batches failed") from exc
            time.sleep(2)
            continue

        added = 0
        for text in generated:
            key = text.lower()
            if valid(text) and key not in seen:
                rows.append({
                    "text": text,
                    "domain": "ecommerce",
                    "generator": "ollama_llama3",
                    "model": MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                })
                seen.add(key)
                added += 1
                if len(rows) >= args.target:
                    break
        consecutive_failures = 0 if added else consecutive_failures + 1

        with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "text", "domain", "generator", "model", "prompt_version", "generated_at_utc"
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Batch {batch_number}: +{added}; total {len(rows)}/{args.target}", flush=True)

    print(f"Saved {len(rows)} reviews to {args.output}")


if __name__ == "__main__":
    main()
