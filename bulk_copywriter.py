"""
DecodeLabs - Project 2: Automated Copywriting & Tone Transformer (Step 2: bulk + async)

Reads many products from a CSV file, generates copy for all of them at the same
time, and saves everything into a results CSV file.

Slides covered:
    - Slide 8-10 : asyncio (async def / await), asyncio.gather and asyncio.as_completed
    - Slide 11   : asyncio.Semaphore (limits parallel requests) + Tenacity retry
                   (randomized exponential backoff)
    - Slide 14   : argparse CLI
    - Slide 15-16: Pydantic validation at the end, same master template as Step 1

Needs copywriter.py (Step 1) in the same folder.

Setup (inside the venv):
    pip install tenacity
Run:
    python bulk_copywriter.py --input products.csv --output results.csv
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time

from openai import AsyncOpenAI
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

# Reuse the Step 1 building blocks (template, platform rules, validation model)
from copywriter import (
    MAX_COMPLETION_TOKENS,
    MODEL,
    PLATFORMS,
    TIMEOUT_SECONDS,
    build_prompt,
    clean_text,
    parse_result,
)

OUTPUT_COLUMNS = [
    "product", "platform", "tone", "temperature", "headline", "body",
    "hashtags", "call_to_action", "characters", "status", "error",
]


# --------------------------------------------------------------------------
# Retry shield (slide 11: Tenacity, exponential backoff with random jitter)
# --------------------------------------------------------------------------
def is_retryable(exc: BaseException) -> bool:
    """Retry only temporary problems: rate limits, busy servers, timeouts."""
    status = getattr(exc, "status_code", None)
    if status in (429, 500, 502, 503, 529):
        return True
    text = str(exc).lower()
    return any(w in text for w in ("overloaded", "unavailable", "rate limit", "timed out", "timeout", "connection"))


def announce_retry(retry_state) -> None:
    wait = retry_state.next_action.sleep
    print(f"   [retry {retry_state.attempt_number}] server busy, {wait:.1f}s baad dobara koshish...")


@retry(
    retry=retry_if_exception(is_retryable),
    wait=wait_random_exponential(multiplier=2, max=30),
    stop=stop_after_attempt(5),
    before_sleep=announce_retry,
    reraise=True,
)
async def call_model(client: AsyncOpenAI, prompt: str, temperature: float, top_p: float) -> str:
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    return response.choices[0].message.content or ""


# --------------------------------------------------------------------------
# One row -> one piece of copy
# --------------------------------------------------------------------------
def error_row(row: dict, message: str) -> dict:
    return {
        "product": row.get("product", ""),
        "platform": row.get("platform", ""),
        "tone": row.get("tone", ""),
        "temperature": row.get("temperature", ""),
        "status": "FAILED",
        "error": message,
    }


async def generate_one(client: AsyncOpenAI, sem: asyncio.Semaphore, index: int, row: dict, top_p: float) -> dict:
    """Validate the row, then generate copy. Never raises: failures become FAILED rows."""
    product = clean_text(row.get("product", ""), 80)
    description = clean_text(row.get("description", ""), 400)
    platform = clean_text(row.get("platform", ""), 20).lower()
    tone = clean_text(row.get("tone", ""), 40)

    if not product or not tone:
        return error_row(row, "product ya tone khali hai")
    if platform not in PLATFORMS:
        return error_row(row, f"platform sahi nahi, in mein se ho: {list(PLATFORMS)}")

    raw_temp = (row.get("temperature") or "").strip()
    try:
        temperature = float(raw_temp) if raw_temp else PLATFORMS[platform]["temperature"]
    except ValueError:
        return error_row(row, "temperature number nahi hai")
    if not 0.0 <= temperature <= 1.5:
        return error_row(row, "temperature 0.0 se 1.5 ke beech ho")

    prompt = build_prompt(product, description, platform, tone)
    limit = PLATFORMS[platform]["max_chars"]
    last_problem = "unknown error"

    try:
        async with sem:  # Semaphore gate: only N requests run at the same time
            for _ in range(2):  # one extra try if the JSON is broken or the text is too long
                raw = await call_model(client, prompt, temperature, top_p)
                try:
                    result = parse_result(raw)
                except (ValueError, ValidationError, json.JSONDecodeError):
                    last_problem = "model ne sahi JSON nahi diya"
                    prompt += "\n\nYour last reply was not valid JSON. Reply with ONLY the JSON object."
                    continue

                length = len(result.full_text())
                if length > limit:
                    last_problem = f"copy {length} characters ki thi, limit {limit}"
                    prompt += f"\n\nYour last reply was {length} characters. It MUST be under {limit}. Shorten it."
                    continue

                return {
                    "product": product,
                    "platform": platform,
                    "tone": tone,
                    "temperature": temperature,
                    "headline": result.headline,
                    "body": result.body,
                    "hashtags": " ".join(result.hashtags),
                    "call_to_action": result.call_to_action,
                    "characters": length,
                    "status": "OK",
                    "error": "",
                }
    except Exception as exc:  # network / API errors after all retries
        return error_row(row, f"{type(exc).__name__}: {exc}")

    return error_row(row, last_problem)


# --------------------------------------------------------------------------
# Running everything together (slide 10: gather vs as_completed)
# --------------------------------------------------------------------------
async def run_all(rows: list, concurrency: int, top_p: float, mode: str) -> list:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY set nahi hai.")
    if not MODEL:
        sys.exit("LLM_MODEL set nahi hai.")

    # max_retries=0: Tenacity handles retries, so the SDK must not retry on its own.
    client = AsyncOpenAI(
        api_key=key,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        timeout=TIMEOUT_SECONDS,
        max_retries=0,
    )
    sem = asyncio.Semaphore(concurrency)

    async def tagged(i: int, row: dict):
        """Return (position, result) so we always know which row a result belongs to."""
        return i, await generate_one(client, sem, i, row, top_p)

    tasks = [asyncio.create_task(tagged(i, row)) for i, row in enumerate(rows)]

    results: list = [None] * len(rows)
    try:
        if mode == "gather":
            # Results come back in the SAME order as the input rows.
            done = await asyncio.gather(*tasks, return_exceptions=True)
            for i, item in enumerate(done):
                if isinstance(item, BaseException):
                    results[i] = error_row(rows[i], str(item))
                else:
                    results[i] = item[1]
        else:
            # as_completed: print each result the moment it finishes (order is not fixed).
            for finished in asyncio.as_completed(tasks):
                position, item = await finished
                results[position] = item
                print(f"   finished: {item.get('product')} [{item.get('platform')}] -> {item.get('status')}")
    finally:
        await client.close()
    return results


# --------------------------------------------------------------------------
# CSV in / CSV out
# --------------------------------------------------------------------------
def read_rows(path: str) -> list:
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return [
                {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
                for row in reader
                if any((v or "").strip() for v in row.values())
            ]
    except FileNotFoundError:
        sys.exit(f"File nahi mili: {path}")


def write_rows(path: str, results: list) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig so Excel shows emojis
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk Copywriting & Tone Transformer (async)")
    p.add_argument("--input", default="products.csv", help="CSV: product,description,platform,tone,temperature")
    p.add_argument("--output", default="results.csv", help="Where to save the results")
    p.add_argument("--concurrency", type=int, default=3, help="Max requests at the same time (Semaphore)")
    p.add_argument("--top-p", type=float, default=0.9, help="Top_P for every row")
    p.add_argument("--mode", choices=["gather", "completed"], default="gather",
                   help="gather = keep input order, completed = show results as they finish")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        sys.exit("--concurrency kam az kam 1 ho.")
    if not 0.0 <= args.top_p <= 1.0:
        sys.exit("Top_P 0.0 se 1.0 ke beech ho.")

    rows = read_rows(args.input)
    if not rows:
        sys.exit("CSV mein koi row nahi mili.")

    print(f"{len(rows)} products | concurrency: {args.concurrency} | mode: {args.mode} | model: {MODEL}\n")

    started = time.perf_counter()
    results = asyncio.run(run_all(rows, args.concurrency, args.top_p, args.mode))
    elapsed = time.perf_counter() - started

    write_rows(args.output, results)

    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\nDone: {ok} kamyab, {len(results) - ok} fail | waqt: {elapsed:.1f} second")
    print(f"Results save ho gaye: {args.output}")
    for r in results:
        if r["status"] != "OK":
            print(f" - {r.get('product') or '(khali)'}: {r['error']}")


if __name__ == "__main__":
    main()
