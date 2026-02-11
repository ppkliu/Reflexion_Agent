"""ripgrep-all (rga) wrapper — unified search tool for heterogeneous document knowledge bases.

Provides:
  1. rga_search()        — plain Python function (used by agno @tool and standalone)
  2. RgaSearchNanobot     — nanobot Tool(ABC) subclass
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from src.config import get_config


def rga_search(
    query: str,
    root_dir: Optional[str] = None,
    file_pattern: Optional[str] = None,
    context_lines: Optional[int] = None,
    max_matches: Optional[int] = None,
) -> str:
    """Search documents in the knowledge directory using ripgrep-all.

    Supports pdf, docx, xlsx, pptx, epub, md, txt, code, zip archives, etc.

    Args:
        query: Regex or keyword to search for.
        root_dir: Directory to search. Defaults to config knowledge.root_dir.
        file_pattern: Optional glob filter, e.g. '*.pdf'.
        context_lines: Lines of context around each match.
        max_matches: Maximum matches per file.

    Returns:
        Formatted markdown string with search results.
    """
    cfg = get_config().knowledge
    root_dir = root_dir or cfg.root_dir
    context_lines = context_lines if context_lines is not None else cfg.rga_context_lines
    max_matches = max_matches if max_matches is not None else cfg.rga_max_matches

    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        return f"Knowledge directory not found: {root}"

    cmd = [
        "rga",
        "--json",
        f"-C{context_lines}",
        f"--max-count={max_matches}",
        query,
        str(root),
    ]
    if file_pattern:
        cmd.extend(["--glob", file_pattern])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return "Error: rga (ripgrep-all) is not installed. Install with: cargo install ripgrep-all"
    except subprocess.TimeoutExpired:
        return "Error: rga search timed out after 30s."

    if result.returncode != 0 and not result.stdout:
        stderr = result.stderr.strip()
        if "No such file or directory" in stderr:
            return f"rga error: directory not found — {root}"
        return f"rga error (code {result.returncode}): {stderr[:400]}"

    # rga --json outputs JSONL (one JSON object per line)
    matches = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if obj.get("type") == "match":
            data = obj.get("data", {})
            path_obj = data.get("path", {})
            filename = path_obj.get("text", "?") if isinstance(path_obj, dict) else str(path_obj)
            line_number = data.get("line_number", "?")
            lines = data.get("lines", {})
            text = lines.get("text", "").strip() if isinstance(lines, dict) else str(lines).strip()

            matches.append(
                f"**{filename}** (line {line_number}):\n```\n{text}\n```"
            )

    if not matches:
        return f"No matches found for '{query}' in {root}"

    header = f"Found {len(matches)} match(es) for `{query}`:\n\n"
    return header + "\n---\n".join(matches)
