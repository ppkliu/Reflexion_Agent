# Reflexion + Tree of Thoughts 整合框架 — 實作計畫

## 技術棧確認（基於 2026-02 實際 API 調查）

| 組件 | 角色 | 實際 API 基礎 |
|------|------|---------------|
| **ripgrep-all (rga)** | 文檔知識庫檢索引擎 | CLI → subprocess 包裝 |
| **HKUDS/nanobot** | 輕量 agent loop + ToT 樹展開 | `AgentLoop` + 自訂 `Tool(ABC)` + `SubagentManager` |
| **agno** | 持久記憶 + 多 agent 協作 + Reflexion 記憶 | `Agent` + `Workflow/Step/Loop` + `SqliteDb` + `MemoryManager` + `@tool` |

---

## 專案結構

```
doc_Reflexion/
├── pyproject.toml                    # 依賴管理
├── config.yaml                       # 全域設定（model、paths、thresholds）
├── knowledge/                        # 放置文檔知識庫的目錄
│   └── .gitkeep
├── data/                             # runtime 資料
│   └── reflexion.db                  # agno SqliteDb 儲存
├── src/
│   ├── __init__.py
│   ├── config.py                     # 載入 config.yaml
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── rga_search.py             # rga CLI 包裝（同時提供 nanobot Tool 與 agno @tool）
│   │   └── rga_nanobot_tool.py       # nanobot Tool(ABC) 子類版本
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── episodic_store.py         # 基於 agno SqliteDb 的 episodic trial 記憶
│   │   └── reflection_retriever.py   # 檢索相關 reflections（keyword + 可選 vector）
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── actor.py                  # Actor agent（agno Agent，帶 rga_search tool）
│   │   ├── evaluator.py              # Evaluator agent（LLM-as-judge）
│   │   └── reflector.py              # Reflector agent（生成 verbal reflection）
│   │
│   ├── tot/
│   │   ├── __init__.py
│   │   ├── node.py                   # ToTNode 資料結構
│   │   ├── bfs.py                    # ToT-BFS 實作（beam search）
│   │   ├── evaluator.py              # state value 評估（呼叫 LLM）
│   │   └── utils.py                  # parse numbered thoughts、extract score 等
│   │
│   ├── reflexion/
│   │   ├── __init__.py
│   │   ├── loop.py                   # Reflexion 主迴圈（跨 trial）
│   │   └── prompts.py                # Actor/Reflector/Evaluator 的 prompt 模板
│   │
│   ├── hybrid/
│   │   ├── __init__.py
│   │   └── reflexion_tot.py          # Reflexion(外層) + ToT(內層) 混合迴圈
│   │
│   └── workflow/
│       ├── __init__.py
│       └── agno_workflow.py          # 用 agno Workflow V2 組裝完整 pipeline
│
├── scripts/
│   ├── run_baseline.py               # Phase 1: ReAct + rga baseline
│   ├── run_reflexion.py              # Phase 3: 純 Reflexion loop
│   ├── run_tot.py                    # Phase 4: 單次 ToT-BFS
│   └── run_hybrid.py                 # Phase 5: Reflexion + ToT 混合
│
└── tests/
    ├── test_rga_search.py
    ├── test_episodic_store.py
    ├── test_tot_bfs.py
    └── test_reflexion_loop.py
```

---

## 實作階段（6 個 Phase）

### Phase 0 — 環境搭建 + rga 工具

**目標：** rga_search 工具可用，能從 knowledge/ 目錄搜到文件內容

**檔案：**
- `pyproject.toml` — 依賴：agno, nanobot-ai, pyyaml, pydantic
- `config.yaml` — knowledge_dir, model 設定, thresholds
- `src/config.py` — 載入設定
- `src/tools/rga_search.py` — subprocess 包裝 rga，輸出格式化 markdown
- `src/tools/rga_nanobot_tool.py` — nanobot `Tool(ABC)` 子類

**驗證：** `rga --json "test" ./knowledge` 正常輸出

---

### Phase 1 — ReAct + rga baseline

**目標：** 單一 agno Agent 能用 rga_search 查文檔並回答問題

**檔案：**
- `src/agents/actor.py` — agno `Agent` + `@tool` rga_search
- `scripts/run_baseline.py` — 輸入問題 → actor 回答

**驗證：** 能正確引用 knowledge/ 內 pdf/md 內容回答

---

### Phase 2 — agno 持久記憶

**目標：** 跨 session 的 episodic trial 記憶

**檔案：**
- `src/memory/episodic_store.py`
  - `EpisodicStore` 類別
  - `init_table()` — CREATE TABLE reflexion_trials
  - `save_trial(task_category, task_key, trial_id, query, trajectory_digest, final_answer, score, reflection, used_reflections)`
  - `get_relevant_reflections(task_key, top_k=4)` — ORDER BY score DESC, 最近優先
- `src/memory/reflection_retriever.py`
  - 格式化 reflections 為 prompt block

**驗證：** 存入 5 筆 trial → 取回 top-3 reflections

---

### Phase 3 — Reflexion Loop

**目標：** 跨 trial 的 verbal self-reflection，accuracy 隨 trial 上升

**檔案：**
- `src/agents/evaluator.py` — agno Agent，輸出 JSON `{success, score, reason}`
- `src/agents/reflector.py` — agno Agent，生成 bullet-point reflection
- `src/reflexion/prompts.py` — prompt 模板集（actor_system, reflector_system, evaluator_system）
- `src/reflexion/loop.py`
  - `ReflexionRunner` 類別
  - `run(task_query, instruction, category, max_trials=7, min_score=0.82)` → dict
  - 迴圈：retrieve → actor → evaluate → reflect → store → loop/break

**驗證：** 同一問題跑 5 次 trial，觀察 score 是否遞增

---

### Phase 4 — ToT-BFS

**目標：** 單次問題的 tree search，替代 plain generation

**檔案：**
- `src/tot/node.py` — `ToTNode` dataclass（thought, state, parent, value, depth, children）
- `src/tot/evaluator.py` — `evaluate_state(state, task) → float` 用 LLM 評分
- `src/tot/utils.py` — `parse_numbered_thoughts()`, `extract_float_score()`
- `src/tot/bfs.py`
  - `run_tot_bfs(query, system_prompt, k=4, max_depth=5, beam_width=3, llm_call, tool_executor)` → dict
  - 回傳 `{final_answer, confidence, path, depth_reached}`

**驗證：** 複雜推理問題上，ToT 正確率 > plain generation

---

### Phase 5 — Reflexion + ToT 混合

**目標：** ToT 做內層推理，Reflexion 做跨 trial 學習

**檔案：**
- `src/hybrid/reflexion_tot.py`
  - `HybridReflexionToT` 類別
  - `run(task_query, instruction, category, max_trials, tot_config)` → dict
  - 每個 trial 內跑 `run_tot_bfs()` → evaluate → reflect on ToT tree → store

**驗證：** 同樣問題類型，score 跨 trial 上升 + ToT 樹品質改善

---

### Phase 6（選配）— agno Workflow V2 組裝

**目標：** 用 agno 的 `Workflow` + `Step` + `Loop` + `Condition` 正式組裝

**檔案：**
- `src/workflow/agno_workflow.py`
  - 用 `Loop(end_condition=..., steps=[...])` 實現 Reflexion 主迴圈
  - 用 `Condition` 做 success/fail 分支
  - 用 `Parallel` 做 ToT 多分支展開（如需要）

---

## 關鍵設計決策

1. **LLM 選擇：** Actor 用 Claude Sonnet 4.5（性價比），Evaluator 用同模型，Reflector 用同模型。可透過 config.yaml 切換。

2. **記憶檢索：** Phase 2-3 先用 keyword + recency（SQL LIKE + ORDER BY timestamp），Phase 6+ 可升級 vector search。

3. **rga 而非 embedding：** 知識庫不做預先 chunk+embed，保持極簡。rga 即時搜 regex，回傳帶 context 的片段。

4. **nanobot 與 agno 分工：**
   - nanobot 負責 ToT 的樹節點展開（輕量 subagent，每個 node 一個 subprocess call）
   - agno 負責 Agent 定義、記憶持久化、Workflow 編排
   - 兩者共用 rga_search tool（各自包裝格式不同）

5. **混合策略（來自參考文件）：**
   - 模式 A：簡單問題 → 直接 ReAct + rga
   - 模式 B：中等 → ToT 單次 → 若低分進 Reflexion
   - 模式 C：高重複性 → 強制 Reflexion，內部用 ToT

---

## config.yaml 結構

```yaml
model:
  provider: anthropic
  id: claude-sonnet-4-5-20250929
  max_tokens: 4096
  temperature: 0.7

knowledge:
  root_dir: ./knowledge
  rga_context_lines: 6
  rga_max_matches: 4

reflexion:
  max_trials: 7
  min_success_score: 0.82
  reflection_top_k: 4

tot:
  branch_factor: 4
  max_depth: 5
  beam_width: 3
  search_algo: bfs

db:
  path: ./data/reflexion.db
```
