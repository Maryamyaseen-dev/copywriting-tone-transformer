"""
DecodeLabs - Project 2: Automated Copywriting & Tone Transformer (Step 1)

What this version does:
    1. CLI (argparse): Product_Name, Description, Platform, Tone
    2. Cleans (sanitizes) the user's input
    3. Builds a master prompt with an f-string template
       + adds platform-specific rules (e.g. X/Twitter max 280 characters)
    4. Sends it to the LLM with Temperature and Top_P
    5. Validates the answer with a Pydantic model + length check

Run (PowerShell, inside the venv, same variables as Project 1):
    $env:OPENAI_API_KEY="your_key"
    $env:OPENAI_BASE_URL="https://api.groq.com/openai/v1"
    $env:LLM_MODEL="openai/gpt-oss-120b"
    python copywriter.py --product "Running Shoe" --platform instagram --tone witty
"""

import argparse
import json
import os
import sys
import time

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# --------------------------------------------------------------------------
# 1. Settings
# --------------------------------------------------------------------------
MODEL = os.environ.get("LLM_MODEL", "")
TIMEOUT_SECONDS = 60
MAX_RETRIES = 4
MAX_COMPLETION_TOKENS = 1500  # reasoning models also count their "thinking" tokens

# Per-platform rules (slide 6: platform-specific filtering) and a default
# Temperature (slide 7: low = consistent, high = creative).
PLATFORMS = {
    "linkedin": {
        "rules": "Professional but human. 80-150 words. No emojis, or at most one. 3 relevant hashtags.",
        "max_chars": 3000,
        "temperature": 0.5,
    },
    "instagram": {
        "rules": "Short, catchy, visual. 40-100 words. Use 2-4 emojis. 5 relevant hashtags.",
        "max_chars": 2200,
        "temperature": 0.8,
    },
    "email": {
        "rules": "Headline is the email SUBJECT LINE (under 60 characters). Body is 80-150 words, "
                 "clear and friendly. No hashtags.",
        "max_chars": 2000,
        "temperature": 0.2,
    },
    "x": {
        "rules": "Maximum 280 characters in total (headline + body + hashtags together). "
                 "Punchy. At most 2 hashtags.",
        "max_chars": 280,
        "temperature": 0.8,
    },
}


# --------------------------------------------------------------------------
# 2. Pydantic model: the strict shape of the final answer (slide 16, step 1)
# --------------------------------------------------------------------------
class CopyResult(BaseModel):
    headline: str = Field(min_length=1)
    body: str = Field(min_length=1)
    hashtags: list[str] = Field(default_factory=list)
    call_to_action: str = ""

    def full_text(self) -> str:
        parts = [self.headline, self.body, self.call_to_action, " ".join(self.hashtags)]
        return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# 3. Input cleaning (slide 5: sanitize before it touches the model)
# --------------------------------------------------------------------------
def clean_text(value: str, max_len: int) -> str:
    value = " ".join(value.split())              # remove extra spaces and newlines
    value = value.replace("{", "").replace("}", "")  # no template characters
    return value[:max_len].strip()


# --------------------------------------------------------------------------
# 4. Master template (slide 6: f-strings + platform constraints)
# --------------------------------------------------------------------------
def build_prompt(product: str, description: str, platform: str, tone: str) -> str:
    rules = PLATFORMS[platform]["rules"]
    return f"""You are a professional marketing copywriter.

Write marketing copy for the product below.

Product name: {product}
Product description: {description or "Not provided. Use only general, safe claims."}
Platform: {platform}
Tone: {tone}

Platform rules: {rules}

Brand safety rules:
- Do not invent prices, discounts, statistics or awards.
- Do not make medical or legal claims.
- Keep the tone "{tone}" the whole way through.

Reply with ONLY valid JSON, no extra text, in exactly this shape:
{{"headline": "...", "body": "...", "hashtags": ["#..."], "call_to_action": "..."}}"""


# --------------------------------------------------------------------------
# 5. Calling the model (Temperature and Top_P live here)
# --------------------------------------------------------------------------
def make_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY set nahi hai.")
    if not MODEL:
        sys.exit("LLM_MODEL set nahi hai.")
    return OpenAI(
        api_key=key,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        timeout=TIMEOUT_SECONDS,
    )


def is_busy_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status in (429, 500, 502, 503, 529) or "overloaded" in text or "unavailable" in text


def call_model(client: OpenAI, prompt: str, temperature: float, top_p: float) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            if is_busy_error(exc) and attempt < MAX_RETRIES:
                wait = 4 * attempt
                print(f"[Server busy, {wait} second baad dobara koshish...]")
                time.sleep(wait)
            else:
                raise
    return ""


# --------------------------------------------------------------------------
# 6. Validation of the answer
# --------------------------------------------------------------------------
def parse_result(raw: str) -> CopyResult:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model ne JSON nahi diya.")
    data = json.loads(raw[start:end + 1])
    return CopyResult(**data)


def generate_copy(client, product, description, platform, tone, temperature, top_p) -> CopyResult:
    prompt = build_prompt(product, description, platform, tone)
    limit = PLATFORMS[platform]["max_chars"]

    result = None
    for attempt in range(2):  # one automatic retry if the answer is too long or broken
        raw = call_model(client, prompt, temperature, top_p)
        try:
            result = parse_result(raw)
        except (ValueError, ValidationError, json.JSONDecodeError):
            prompt += "\n\nYour last reply was not valid JSON. Reply with ONLY the JSON object."
            continue

        length = len(result.full_text())
        if length <= limit:
            return result
        prompt += f"\n\nYour last reply was {length} characters. It MUST be under {limit} characters. Shorten it."

    if result is None:
        raise ValueError("Model se sahi JSON nahi mila.")
    raise ValueError(f"Copy {limit} characters se lambi rahi. Dobara chalayein.")


# --------------------------------------------------------------------------
# 7. CLI (slide 14)
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Automated Copywriting & Tone Transformer")
    p.add_argument("--product", help="Product name")
    p.add_argument("--description", default="", help="Short product description")
    p.add_argument("--platform", choices=list(PLATFORMS), help="Target platform")
    p.add_argument("--tone", help="Tone, e.g. witty, formal, friendly")
    p.add_argument("--temperature", type=float, help="0.0 to 1.5 (default depends on platform)")
    p.add_argument("--top-p", type=float, default=0.9, help="0.0 to 1.0 (default 0.9)")
    p.add_argument("--show-prompt", action="store_true", help="Print the compiled prompt")
    return p.parse_args()


def ask_if_missing(value, question):
    return value if value else input(question).strip()


def main() -> None:
    args = parse_args()

    product = clean_text(ask_if_missing(args.product, "Product name: "), 80)
    description = clean_text(args.description, 400)
    platform = args.platform or input(f"Platform {list(PLATFORMS)}: ").strip().lower()
    tone = clean_text(ask_if_missing(args.tone, "Tone (e.g. witty): "), 40)

    if not product or not tone:
        sys.exit("Product aur tone khali nahi ho sakte.")
    if platform not in PLATFORMS:
        sys.exit(f"Platform in mein se ho: {list(PLATFORMS)}")

    temperature = args.temperature if args.temperature is not None else PLATFORMS[platform]["temperature"]
    if not 0.0 <= temperature <= 1.5:
        sys.exit("Temperature 0.0 se 1.5 ke beech ho.")
    if not 0.0 <= args.top_p <= 1.0:
        sys.exit("Top_P 0.0 se 1.0 ke beech ho.")

    if args.show_prompt:
        print("----- COMPILED PROMPT -----")
        print(build_prompt(product, description, platform, tone))
        print("---------------------------\n")

    print(f"Platform: {platform} | Tone: {tone} | Temperature: {temperature} | Top_P: {args.top_p}\n")

    client = make_client()
    try:
        result = generate_copy(client, product, description, platform, tone, temperature, args.top_p)
    except Exception as exc:
        sys.exit(f"[Error] {exc}")

    print(f"HEADLINE : {result.headline}\n")
    print(f"{result.body}\n")
    if result.call_to_action:
        print(f"CTA      : {result.call_to_action}")
    if result.hashtags:
        print(f"HASHTAGS : {' '.join(result.hashtags)}")
    print(f"\n({len(result.full_text())} characters)")


if __name__ == "__main__":
    main()
