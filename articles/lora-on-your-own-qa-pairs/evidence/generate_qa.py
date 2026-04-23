"""Generate Q&A pairs from nvidia-learn articles via NIM 8B.

Output: /home/nvidia/lora-work/qa.jsonl with {"question":..., "answer":..., "source":...}
"""
import json
import re
import sys
import time
import concurrent.futures as cf
from pathlib import Path

import urllib.request

NIM_URL = "http://localhost:8000/v1/chat/completions"
NIM_MODEL = "meta/llama-3.1-8b-instruct"
ARTICLES = Path("/home/nvidia/nvidia-learn/articles")
OUT = Path("/home/nvidia/lora-work/qa.jsonl")

STOP_SLUGS = {"_drafts", "lora-on-your-own-qa-pairs", "lora-fine-tune-nemotron-on-spark"}


def strip_frontmatter(md: str) -> tuple[dict, str]:
    if md.startswith("---"):
        end = md.find("\n---", 3)
        fm_block = md[3:end]
        body = md[end + 4 :]
        fm = {}
        for line in fm_block.strip().split("\n"):
            m = re.match(r"^(\w+):\s*(.*)$", line)
            if m:
                fm[m.group(1)] = m.group(2).strip().strip("'\"")
        return fm, body
    return {}, md


def chunks(text: str, words_per_chunk: int = 900, overlap: int = 150):
    words = text.split()
    i = 0
    while i < len(words):
        yield " ".join(words[i : i + words_per_chunk])
        i += words_per_chunk - overlap


PROMPT_TEMPLATE = """Generate up to {n} question/answer training pairs from the passage below.

ABSOLUTE RULES:
1. Every answer MUST be extractable from THIS passage alone. Quote or closely paraphrase the passage.
2. NEVER invent facts, numbers, flags, version strings, container names, or commands that are not literally in the passage.
3. NEVER produce a question whose answer is "the passage does not mention" or similar — skip it instead.
4. Questions should be self-contained (a reader without the passage should still understand what is being asked).
5. Prefer questions that pin down specific details actually present in the passage: numbers with units, named tools / products / containers / commands / flags, decisions with reasons stated in the passage.
6. It is fine to produce fewer than {n} pairs if the passage does not support {n} good questions. Quality > quantity.

OUTPUT FORMAT:
- Return ONLY a JSON array of objects with keys "question" and "answer".
- No markdown fences. No prose before or after. No trailing commentary.

PASSAGE from article titled "{title}":
<<<PASSAGE
{passage}
PASSAGE>>>

JSON array:"""


def call_nim(prompt: str, n_pairs: int, title: str, passage: str, retries: int = 3):
    body = {
        "model": NIM_MODEL,
        "messages": [{"role": "user", "content": prompt.format(n=n_pairs, title=title, passage=passage)}],
        "temperature": 0.25,
        "top_p": 0.9,
        "max_tokens": 1800,
    }
    req = urllib.request.Request(
        NIM_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
                content = data["choices"][0]["message"]["content"].strip()
                # strip code fences if the model added them
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*", "", content)
                    content = re.sub(r"\s*```$", "", content)
                # find first [ and last ] if there's prose wrap
                if not content.startswith("["):
                    s = content.find("[")
                    e = content.rfind("]")
                    if s != -1 and e > s:
                        content = content[s : e + 1]
                return json.loads(content)
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FAIL {title}: {e}", file=sys.stderr)
                return []
            time.sleep(2 + attempt * 2)


def process_article(article_dir: Path):
    slug = article_dir.name
    if slug in STOP_SLUGS:
        return []
    md_path = article_dir / "article.md"
    if not md_path.exists():
        return []
    fm, body = strip_frontmatter(md_path.read_text())
    if fm.get("status") == "upcoming":
        return []
    title = fm.get("title", slug)
    print(f"  processing {slug} ({len(body.split())} words)", file=sys.stderr)
    pairs = []
    REFUSAL_MARKERS = (
        "does not mention",
        "does not specify",
        "does not provide",
        "does not discuss",
        "not present",
        "not in the passage",
        "not stated",
        "no information",
        "cannot be determined",
        "unclear from the passage",
    )
    for idx, passage in enumerate(list(chunks(body))):
        n = 5
        got = call_nim(PROMPT_TEMPLATE, n, title, passage)
        for p in got or []:
            if not (isinstance(p, dict) and "question" in p and "answer" in p):
                continue
            q = p["question"].strip()
            a = p["answer"].strip()
            if not q or not a:
                continue
            # drop refusals
            low = a.lower()
            if any(m in low for m in REFUSAL_MARKERS):
                continue
            # drop questions that equal their answer (echo failures)
            if q == a:
                continue
            pairs.append({
                "question": q,
                "answer": a,
                "source": slug,
                "chunk": idx,
            })
    return pairs


def main():
    article_dirs = [p for p in sorted(ARTICLES.iterdir()) if p.is_dir() and p.name not in STOP_SLUGS]
    all_pairs = []
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(process_article, d): d.name for d in article_dirs}
        for fut in cf.as_completed(futures):
            slug = futures[fut]
            try:
                pairs = fut.result()
                all_pairs.extend(pairs)
                print(f"DONE {slug}: +{len(pairs)} pairs (total {len(all_pairs)})", file=sys.stderr)
            except Exception as e:
                print(f"ERROR {slug}: {e}", file=sys.stderr)

    with OUT.open("w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(all_pairs)} pairs to {OUT}")


if __name__ == "__main__":
    main()
