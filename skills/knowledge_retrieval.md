---
name: knowledge_retrieval
description: Best practices for using rga_search to retrieve knowledge from documents
always_on: true
tags: [search, knowledge, rga]
---
When searching the knowledge base with rga_search:

1. **Use specific keywords** — Prefer precise terms over vague phrases. "transformer attention mechanism" beats "how transformers work".
2. **Try multiple queries** — If the first search returns no results, rephrase or use synonyms. Technical terms may vary across documents.
3. **Use file_pattern when possible** — If you know the document type (e.g. "*.pdf" for papers, "*.py" for code), filter to reduce noise.
4. **Read context carefully** — rga returns surrounding lines. Read the full context, not just the matched line — important details are often in adjacent lines.
5. **Combine evidence** — When multiple passages are returned, synthesize them rather than relying on a single match.
