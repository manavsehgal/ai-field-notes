#!/usr/bin/env python3
"""Bench llama-nemotron-embed-1b-v2 on DGX Spark."""
import json
import statistics
import time
import urllib.request

URL = "http://localhost:8001/v1/embeddings"
MODEL = "nvidia/llama-nemotron-embed-1b-v2"

# Representative ~512-token chunk (English prose). tiktoken-ish: ~4 chars/token,
# so aim for ~2000 chars of natural text.
CHUNK = (
    "The DGX Spark is a compact personal AI workstation built around the "
    "NVIDIA GB10 Grace Blackwell Superchip, combining an Arm-based Grace CPU "
    "with a Blackwell GPU and 128 gigabytes of unified memory. "
    "It is designed for developers and researchers who want to experiment "
    "with large language models, fine-tuning workflows, and agentic systems "
    "without renting cloud GPUs. The unified memory architecture lets the "
    "CPU and GPU share the same pool, which is especially useful for models "
    "whose weights would not otherwise fit on a typical consumer GPU. "
    "NVIDIA packages the system with a curated software stack called DGX OS, "
    "based on Ubuntu, and an expanding catalog of pre-built playbooks for "
    "inference engines, fine-tuning frameworks, and retrieval pipelines. "
    "In this article we run the Nemotron Retriever embedding NIM, which "
    "converts text passages into 2048-dimensional vectors with support for "
    "Matryoshka truncation down to 384, 512, 768, or 1024 dimensions. "
    "Embeddings are the substrate of modern retrieval-augmented generation: "
    "every query, every document, every snippet is mapped into a shared "
    "vector space where semantic similarity becomes a geometry problem. "
    "Running the embedding endpoint locally on the Spark means the box "
    "becomes the entire retrieval plane, with no outbound dependency on a "
    "hosted API for the hot path of a second-brain system. "
) * 2

def embed(inputs, input_type="passage"):
    body = json.dumps({
        "model": MODEL,
        "input": inputs,
        "input_type": input_type,
        "encoding_format": "float",
        "truncate": "END",
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        obj = json.loads(r.read())
    return obj, time.perf_counter() - t0

def cosine(a, b):
    import math
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb)

def main():
    # 1) probe: vector dim + token count
    obj, t = embed([CHUNK], input_type="passage")
    dim = len(obj["data"][0]["embedding"])
    usage_tokens = obj.get("usage", {}).get("total_tokens", None)
    print(f"probe: dim={dim}, tokens={usage_tokens}, t={t:.3f}s")

    # 2) warmup: 3 single-doc calls
    for _ in range(3):
        embed([CHUNK])

    # 3) batch=1, 20 sequential calls
    t_b1 = []
    for _ in range(20):
        _, t = embed([CHUNK])
        t_b1.append(t)
    p50_1 = statistics.median(t_b1)
    p95_1 = sorted(t_b1)[int(len(t_b1) * 0.95) - 1]
    rps_1 = 1 / p50_1
    tok_per_sec_1 = (usage_tokens or 0) * rps_1
    print(f"batch=1 : p50={p50_1*1000:.1f}ms  p95={p95_1*1000:.1f}ms  ~{rps_1:.1f} docs/s  ~{tok_per_sec_1:.0f} tok/s")

    # 4) batch=8, 10 runs
    t_b8 = []
    for _ in range(10):
        _, t = embed([CHUNK] * 8)
        t_b8.append(t)
    p50_8 = statistics.median(t_b8)
    docs_per_sec_8 = 8 / p50_8
    tok_per_sec_8 = (usage_tokens or 0) * docs_per_sec_8
    print(f"batch=8 : p50={p50_8*1000:.1f}ms/req  {docs_per_sec_8:.1f} docs/s  ~{tok_per_sec_8:.0f} tok/s")

    # 5) batch=32, 5 runs
    t_b32 = []
    for _ in range(5):
        _, t = embed([CHUNK] * 32)
        t_b32.append(t)
    p50_32 = statistics.median(t_b32)
    docs_per_sec_32 = 32 / p50_32
    tok_per_sec_32 = (usage_tokens or 0) * docs_per_sec_32
    print(f"batch=32: p50={p50_32*1000:.1f}ms/req  {docs_per_sec_32:.1f} docs/s  ~{tok_per_sec_32:.0f} tok/s")

    # 6) cosine-sim sanity: query vs near + far passage
    query = "How do unified-memory architectures help large language model inference?"
    near = "Grace Blackwell's shared CPU/GPU memory lets LLM weights exceed typical discrete-GPU VRAM limits."
    far = "Espresso is brewed by forcing pressurized hot water through finely-ground coffee."
    q_obj, _ = embed([query], input_type="query")
    p_obj, _ = embed([near, far], input_type="passage")
    q = q_obj["data"][0]["embedding"]
    p_near = p_obj["data"][0]["embedding"]
    p_far = p_obj["data"][1]["embedding"]
    sim_near = cosine(q, p_near)
    sim_far = cosine(q, p_far)
    print(f"cosine(query, near)={sim_near:.4f}  cosine(query, far)={sim_far:.4f}")

    result = {
        "model": MODEL,
        "embedding_dim": dim,
        "input_tokens_per_chunk": usage_tokens,
        "batch_1": {"p50_ms": round(p50_1 * 1000, 1), "p95_ms": round(p95_1 * 1000, 1), "docs_per_sec": round(rps_1, 2), "tok_per_sec": round(tok_per_sec_1)},
        "batch_8": {"p50_ms": round(p50_8 * 1000, 1), "docs_per_sec": round(docs_per_sec_8, 2), "tok_per_sec": round(tok_per_sec_8)},
        "batch_32": {"p50_ms": round(p50_32 * 1000, 1), "docs_per_sec": round(docs_per_sec_32, 2), "tok_per_sec": round(tok_per_sec_32)},
        "cosine": {"query_vs_near": round(sim_near, 4), "query_vs_far": round(sim_far, 4)},
    }
    print(json.dumps(result, indent=2))
    with open("06-benchmark.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
