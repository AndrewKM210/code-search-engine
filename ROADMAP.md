# Roadmap

This project started as a RAG-based code search engine (Qdrant + fine-tuned sentence
embeddings) and is growing into an agentic coding assistant: an LLM that can choose tools
(search, read files, grep) to answer questions about a codebase, measured against a
one-shot baseline rather than assumed to be better.

**Status legend:** ✅ Done · 🚧 In progress · ⬜ Planned · ⏭️ Skipped

| Milestone | Status |
|---|---|
| [M0 — Harden the base](#m0--harden-the-base) | ✅ Done |
| [M1 — Agentic loop + tools](#m1--agentic-loop--tools) | 🚧 In progress |
| [M2 — Generation eval (LLM-as-judge)](#m2--generation-eval-llm-as-judge) | ⬜ Planned |
| [M3 — MCP server](#m3--mcp-server) | ⬜ Planned |
| [M4 — LLM benchmark (local + API)](#m4--llm-benchmark-local--api) | ⬜ Planned |
| [M5 — Multi-agent orchestration](#m5--multi-agent-orchestration) | ⬜ Planned (gated on M2 results) |
| [M6 — QLoRA capstone](#m6--qlora-capstone) | ⬜ Planned (stretch) |
| [M7 — Hosted demo + observability](#m7--hosted-demo--observability) | ⬜ Planned |

---

## M0 — Harden the base ✅ Done

Baseline cleanup so the project runs on any machine and has a real test/CI safety net.

- [x] Add `resolve_device()` (CUDA > Apple MPS > CPU), replacing hardcoded `device="cuda"`
- [x] Add a `device: auto` config key, threaded through the CLI and the indexing script
- [x] Default CoSQA indexing to the base embedding model; gate the fine-tuned path behind a `--finetuned` flag
- [x] Add a pytest suite: search result formatting, retrieval metrics (MRR/nDCG/recall), payload-key handling, `resolve_device` overrides
- [x] Add a GitHub Actions workflow running the test suite on every push/PR
- [x] Restructure entry points: CLI and Streamlit app moved into `apps/`, `scripts/` reserved for one-off/research tooling
- [x] Add ruff (format + lint) and mypy (type check), wired into a `Makefile` and a separate GitHub Actions lint workflow

**Result carried forward:** fine-tuned embeddings improve retrieval over the base model
(MRR 0.854 → 0.893, nDCG 0.886 → 0.918, Recall 0.981 → 0.990 on CoSQA validation). Every
later milestone reuses this embedding model.

---

## M1 — Agentic loop + tools

Replace the fixed plan→search→critique pipeline with a loop where the LLM chooses which
tool to call, and can take multiple steps (search, then read a file, then search again).

- [x] Add a `self_repo` Qdrant collection name to `main_config.yaml`
- [x] Add a script that indexes this repository's own source into `self_repo` via `index_from_directory`
- [x] Implement a `read_file(path)` tool
- [x] Implement a `list_directory(path)` tool
- [x] Implement a `grep(pattern)` tool
- [x] Wrap `CodeSearchEngine.search` as a `search_code(query)` tool with a name/description/args schema
- [ ] Define a Pydantic schema for a structured tool call (name + arguments)
- [ ] Add native tool-calling to `LLMClient` (Ollama `tools` param, parse `message.tool_calls`) for models that support it
- [ ] Add a prompt + JSON-parse fallback tool-calling path for models without native support, with retry on malformed output
- [ ] Implement the new tool-choosing agent loop, reusing the existing `AgentStep` interface
- [ ] Wire the new agent into the CLI behind a flag, alongside the existing one-shot baseline
- [ ] Add tests for the new agent loop (mocked LLM + tools)

---

## M2 — Generation eval (LLM-as-judge)

Measure whether the agentic loop actually beats the one-shot baseline, instead of assuming it.

- [ ] Write a gold set of ~15–30 Q&A pairs about this repo
- [ ] Implement an LLM-as-judge scorer (correctness + faithfulness)
- [ ] Build an eval harness that runs both agents (baseline vs. tool loop) over the gold set
- [ ] Run the eval and record task-success numbers for both agents
- [ ] Document the before/after comparison

---

## M3 — MCP server

Expose the same tools over the Model Context Protocol so any MCP-aware client can use them.

- [ ] Add the `mcp` SDK as a dependency
- [ ] Wrap `search_code`, `read_file`, `list_directory`, `grep` as MCP tools
- [ ] Add an MCP server entry point
- [ ] Document how to connect it to an MCP client (e.g. Claude Desktop/Code)
- [ ] Manually verify a live query end-to-end through the MCP client

---

## M4 — LLM benchmark (local + API)

Compare model choices on quality, latency, and cost using the M2 eval set.

- [ ] Make `LLMClient`'s model swappable via config (model name + native-tools flag)
- [ ] Benchmark candidate local models (phi3, Qwen2.5-Coder, etc.) on the M2 eval set
- [ ] Add one free-tier API model behind the same `LLMClient` interface
- [ ] Record a quality / latency / cost comparison table
- [ ] Document the recommended default model and why

---

## M5 — Multi-agent orchestration

Only pursued if it measurably improves on M2 — not added for its own sake.

- [ ] Add LangGraph as a dependency
- [ ] Implement retriever / coder / reviewer nodes with a supervisor graph
- [ ] Re-run the M2 eval against the graph
- [ ] Document the go/no-go decision based on the measured result

---

## M6 — QLoRA capstone *(stretch)*

A focused fine-tuning loop on a free GPU tier, kept from blocking earlier milestones.

- [ ] Prepare a small fine-tuning dataset (e.g. tool-call formatting or repo conventions)
- [ ] Run QLoRA training on a free-tier T4 (Colab/Kaggle)
- [ ] Merge adapter weights into the base model
- [ ] Evaluate fine-tuned vs. base model on the focused task
- [ ] Quantize the merged model
- [ ] Serve the quantized model locally via Ollama

---

## M7 — Hosted demo + observability

- [ ] Write a FastAPI backend wrapping the agent
- [ ] Containerize the backend with Docker
- [ ] Add a hard spend cap and per-IP rate limit for demo LLM calls
- [ ] Deploy the backend to a free-tier host
- [ ] Build a minimal static frontend and deploy it
- [ ] Add tracing (Langfuse/Phoenix) for latency and cost per request
- [ ] Record a short demo video and link it from the README
