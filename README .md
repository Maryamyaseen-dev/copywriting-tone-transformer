# Automated Copywriting & Tone Transformer

A Python tool that turns a raw product description into marketing copy for
different platforms (LinkedIn, Instagram, Email, X) in a chosen tone.
Built for DecodeLabs Generative AI Training, Project 2.

## Features
- CLI built with `argparse`: product name, description, platform, tone
- Dynamic master prompt template built with f-strings
- Platform-specific rules (for example X: maximum 280 characters)
- Inference tuning with `Temperature` and `Top_P`
- Input cleaning before anything reaches the model
- Output validation with Pydantic, plus an automatic length check
- Bulk mode: many products from a CSV file, generated at the same time
  with `asyncio.gather`, limited by an `asyncio.Semaphore`
- Automatic retry with randomized exponential backoff (Tenacity)

## Files
| File | Purpose |
|---|---|
| `copywriter.py` | Single product from the command line |
| `bulk_copywriter.py` | Many products from a CSV file (async) |
| `products.csv` | Example input |
| `requirements.txt` | Python packages |

## Setup (Windows PowerShell)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

$env:OPENAI_API_KEY="your_key_here"
$env:OPENAI_BASE_URL="https://api.groq.com/openai/v1"
$env:LLM_MODEL="openai/gpt-oss-120b"
```
Never put your API key inside the code or upload it to GitHub.

## Single product
```
python copywriter.py --product "Running Shoe" --description "Lightweight shoe for daily runners" --platform instagram --tone witty
```
Options: `--temperature 0.0-1.5`, `--top-p 0.0-1.0`, `--show-prompt`

## Many products from a CSV
CSV columns: `product,description,platform,tone,temperature` (temperature is optional).
```
python bulk_copywriter.py --input products.csv --output results.csv --concurrency 3
```
Options: `--mode gather` (keeps input order) or `--mode completed` (shows results as they finish).

## What I learned
- Low Temperature gives consistent copy (good for emails); high Temperature gives
  more varied copy (good for social media).
- Async code does not make one request faster, but it cuts the total time by
  overlapping the waiting.
- A Semaphore and retry with backoff are needed to survive rate limits (HTTP 429).
- Reasoning models count their "thinking" tokens, so token limits need care.

## Note
The OpenAI Batch API part of the training slides is not included, because this
project was run on Groq. Bulk processing is done with the async pipeline instead.
