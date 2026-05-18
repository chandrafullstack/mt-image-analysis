"""
Research assistant — Claude tool-use agent with RAG over the project's
markdown docs (EXPERIMENTS_LOG.md, README.md, etc.) and pandas tools
over the per-mitochondrion CSV.

Design goals:
- Zero new dependencies (uses anthropic + sklearn TF-IDF, both already in env)
- Streaming-friendly final answers (kept simple: sync for now)
- Bounded: tool-calling loop capped at 8 iterations
- Cites sources by file path + heading

Usage from FastAPI:
    from src.research_agent import answer_question
    reply = answer_question("Why did we get 16k dots from 14 images?")
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "outputs" / "metrics" / "features_with_gratio.csv"

# Markdown sources to index. Researcher cares most about EXPERIMENTS_LOG.
DOCS = [
    PROJECT_ROOT / "EXPERIMENTS_LOG.md",
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "RESEARCHER_STEP_BY_STEP.md",
    PROJECT_ROOT / "PROJECT_CHECKPOINT_2026-05-17.md",
]

# Lazy-loaded singletons
_chunks: list[dict] | None = None
_vectorizer: TfidfVectorizer | None = None
_matrix = None


def _chunk_markdown(text: str, source: str) -> list[dict]:
    """Split a markdown doc on H2/H3 headings; keep heading as title."""
    lines = text.splitlines()
    chunks = []
    cur_title = "(intro)"
    cur_body: list[str] = []

    def flush():
        body = "\n".join(cur_body).strip()
        if body:
            chunks.append({"source": source, "title": cur_title, "text": body})

    for ln in lines:
        m = re.match(r"^#{1,3}\s+(.*)", ln)
        if m:
            flush()
            cur_title = m.group(1).strip()
            cur_body = []
        else:
            cur_body.append(ln)
    flush()
    # Split very long chunks (>2000 chars) into rolling windows so retrieval is sharper
    refined = []
    for c in chunks:
        if len(c["text"]) <= 2000:
            refined.append(c)
        else:
            words = c["text"].split()
            step = 250  # ~250 words per sub-chunk
            for i in range(0, len(words), step):
                refined.append({
                    "source": c["source"],
                    "title": c["title"],
                    "text": " ".join(words[i:i + step + 50]),  # 50-word overlap
                })
    return refined


def _build_index() -> None:
    """Load + chunk + TF-IDF the markdown corpus once."""
    global _chunks, _vectorizer, _matrix
    chunks: list[dict] = []
    for path in DOCS:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        chunks.extend(_chunk_markdown(text, path.name))
    if not chunks:
        _chunks, _vectorizer, _matrix = [], None, None
        return
    vec = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
    mat = vec.fit_transform([c["title"] + "\n" + c["text"] for c in chunks])
    _chunks, _vectorizer, _matrix = chunks, vec, mat


def _ensure_index() -> None:
    if _chunks is None:
        _build_index()


# ---------- Tools exposed to the LLM ----------

def tool_search_docs(query: str, k: int = 4) -> str:
    """Top-k semantically relevant markdown chunks with source + heading."""
    _ensure_index()
    if not _chunks or _vectorizer is None:
        return "No documents indexed."
    qv = _vectorizer.transform([query])
    sims = cosine_similarity(qv, _matrix)[0]
    top = np.argsort(-sims)[:k]
    out = []
    for idx in top:
        c = _chunks[int(idx)]
        snippet = c["text"][:1200]
        out.append(f"[{c['source']} — {c['title']}] (score={sims[int(idx)]:.2f})\n{snippet}")
    return "\n\n---\n\n".join(out)


def tool_csv_stats(group_by: str | None = None, filter_expr: str | None = None) -> str:
    """
    Aggregate stats from features_with_gratio.csv.
    group_by: e.g. 'resolution_group', 'label_final', 'source_file'
    filter_expr: pandas .query() expression, e.g. "area_um2 > 0.5 and label_final == 'UNHEALTHY'"
    """
    if not DATA_PATH.exists():
        return "Metrics CSV not found."
    df = pd.read_csv(DATA_PATH)
    if filter_expr:
        try:
            df = df.query(filter_expr)
        except Exception as exc:
            return f"Bad filter_expr: {exc}"
    if df.empty:
        return "No rows match."

    metric_cols = [c for c in ["area_um2", "aspect_ratio", "g_ratio", "form_factor"] if c in df.columns]
    if group_by and group_by in df.columns:
        agg = df.groupby(group_by).agg(
            count=("label", "count"),
            **{c: (c, "median") for c in metric_cols},
        ).round(3)
        return f"Rows: {len(df)}\n\n{agg.to_string()}"
    summary = {
        "n_rows": int(len(df)),
        **{f"median_{c}": float(round(df[c].median(), 3)) for c in metric_cols},
        **{f"mean_{c}": float(round(df[c].mean(), 3)) for c in metric_cols},
    }
    if "label_final" in df.columns:
        summary["label_breakdown"] = df["label_final"].value_counts().to_dict()
    return json.dumps(summary, indent=2)


def tool_image_summary(source_file: str) -> str:
    """Per-image breakdown for one source file."""
    if not DATA_PATH.exists():
        return "Metrics CSV not found."
    df = pd.read_csv(DATA_PATH)
    if "source_file" not in df.columns:
        return "source_file column missing — re-run inference."
    sub = df[df["source_file"].str.contains(source_file, case=False, na=False)]
    if sub.empty:
        return f"No mitos found for image matching '{source_file}'."
    cols = [c for c in ["area_um2", "aspect_ratio", "g_ratio", "form_factor"] if c in sub.columns]
    out = {
        "image_match": sub["source_file"].iloc[0],
        "n_mitos": int(len(sub)),
        "resolution_group": sub.get("resolution_group", pd.Series(["unknown"])).iloc[0],
        "pixel_size_um": float(sub.get("pixel_size_um", pd.Series([0.0])).iloc[0]),
        **{f"median_{c}": float(round(sub[c].median(), 3)) for c in cols},
    }
    if "label_final" in sub.columns:
        out["health_breakdown"] = sub["label_final"].value_counts().to_dict()
    return json.dumps(out, indent=2)


# ---------- Tool schemas for Claude ----------

TOOLS = [
    {
        "name": "search_docs",
        "description": (
            "Search the project's markdown documentation (experiments log, "
            "README, checkpoint notes). Use this for questions about decisions, "
            "history, methodology, pixel-size saga, why things were done, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "k": {"type": "integer", "description": "Number of chunks (default 4).", "default": 4},
            },
            "required": ["query"],
        },
    },
    {
        "name": "csv_stats",
        "description": (
            "Compute aggregate statistics from the per-mitochondrion CSV. "
            "Use for quantitative questions about counts, medians, "
            "health breakdowns, comparisons across resolution groups."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {"type": "string", "description": "Column to group by, e.g. 'resolution_group', 'label_final', 'source_file'."},
                "filter_expr": {"type": "string", "description": "Pandas query expression to subset rows first."},
            },
        },
    },
    {
        "name": "image_summary",
        "description": "Get per-image breakdown for a specific source file (substring match).",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_file": {"type": "string", "description": "Image filename or substring."},
            },
            "required": ["source_file"],
        },
    },
]


def _dispatch(name: str, args: dict) -> str:
    if name == "search_docs":
        return tool_search_docs(args["query"], args.get("k", 4))
    if name == "csv_stats":
        return tool_csv_stats(args.get("group_by"), args.get("filter_expr"))
    if name == "image_summary":
        return tool_image_summary(args["source_file"])
    return f"Unknown tool: {name}"


SYSTEM_PROMPT = """You are the mitochondria-analysis project's research assistant.
You help a researcher (and reviewers) understand:
1. WHAT the pipeline does (segmentation -> cropping -> measurement -> classification)
2. WHY decisions were made (the experiment log captures all of this, including dead ends)
3. WHAT the current data shows (CSV stats per image / per resolution group)

Always prefer calling tools over guessing.
- For "why did we..." or "what happened with..." -> search_docs
- For "how many...", "what's the median...", "compare X vs Y" -> csv_stats
- For "tell me about image X" -> image_summary

Cite sources by filename + heading when you used search_docs.
If a tool returns no useful info, say so honestly. Don't invent numbers.
Keep answers concise (3-6 sentences) unless a long summary is explicitly requested.

AFTER your main answer, ALWAYS append a footer with 3 short follow-up
question suggestions the user could ask next. Use exactly this format:

FOLLOWUPS:
- <short question 1>
- <short question 2>
- <short question 3>

Each suggestion must be a single self-contained question under 80 chars.
Make them genuinely relevant — not generic. If the user asked about an
image, suggest comparisons. If they asked about a decision, suggest
related decisions or quantitative checks.
"""


def answer_question(question: str, max_iters: int = 8) -> dict:
    """
    Run the agent loop. Returns:
        {"answer": str, "tool_calls": [{"name", "input", "output_snippet"}], "iterations": int}
    """
    # Lazy-load .env if available (same pattern as claude_score_crops.py)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
        except ImportError:
            pass
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {
            "answer": "ANTHROPIC_API_KEY is not set. Add it to .env and restart the dashboard.",
            "tool_calls": [],
            "iterations": 0,
        }

    import anthropic
    client = anthropic.Anthropic()

    messages: list[dict] = [{"role": "user", "content": question}]
    tool_log: list[dict] = []

    for i in range(max_iters):
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # If the model used a tool, execute and continue
        if resp.stop_reason == "tool_use":
            # Capture full assistant turn (text + tool_use blocks)
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    try:
                        result = _dispatch(block.name, block.input)
                    except Exception as exc:
                        result = f"Tool error: {exc}"
                    tool_log.append({
                        "name": block.name,
                        "input": block.input,
                        "output_snippet": result[:300],
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        # Final answer
        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        full_text = "\n".join(text_parts).strip() or "(empty response)"

        # Split out FOLLOWUPS footer if present
        answer = full_text
        followups: list[str] = []
        m = re.search(r"\n\s*FOLLOWUPS:\s*\n(.+)$", full_text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            answer = full_text[: m.start()].rstrip()
            for line in m.group(1).splitlines():
                line = line.strip()
                if line.startswith("-") or line.startswith("*"):
                    q = line.lstrip("-*").strip()
                    if q:
                        followups.append(q[:120])
            followups = followups[:3]

        return {
            "answer": answer,
            "followups": followups,
            "tool_calls": tool_log,
            "iterations": i + 1,
        }

    return {
        "answer": "(stopped: too many tool-call iterations)",
        "followups": [],
        "tool_calls": tool_log,
        "iterations": max_iters,
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How many mitochondria did we detect in the 200nm images?"
    result = answer_question(q)
    print("\n=== ANSWER ===")
    print(result["answer"])
    print(f"\n=== TOOL CALLS ({result['iterations']} iters) ===")
    for tc in result["tool_calls"]:
        print(f"  {tc['name']}({tc['input']}) -> {tc['output_snippet'][:120]}...")
