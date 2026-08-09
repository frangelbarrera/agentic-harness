# Changelog

All notable changes to Agentic Harness will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Playbook Library** (`arnes.playbooks.library`) — a knowledge layer with 13 domain-specific task templates: `mobile_app`, `web_app`, `cli_tool`, `rest_api`, `osint`, `financial_analysis`, `security_audit`, `data_analysis`, `devops`, `graphic_design`, `content_creation`, `academic_research`, `generic`. Each template encodes the recommended specialist sequence (action graph), clarifying questions, domain context (tools, reference repos, conventions), and known risks.
- **TaskRouter** — deterministic keyword-based classifier that maps a natural-language request to a task domain (no LLM call needed, works offline). Used by `arnes plan` to enrich the planner's system prompt with domain knowledge.
- **Actor-critic review loops** — `arnes run --loops` flag + `step.review` YAML config. After each specialist step, a critic (`@reviewer` by default) evaluates the output; if not approved, the actor is re-invoked with the critic's feedback, up to `max_iterations` (default 3). Emits `REVIEW_ITERATION` and `REVIEW_COMPLETED` events to the Thread for audit.
- **`ReviewLoop` schema** in `arnes.playbooks.schema` — configurable critic, max_iterations, pass_threshold, interactive flag, and focus prompt.
- **OpenRouter support** — `openrouter/` is now a recognised vendor in `get_provider()`, giving access to 336+ hosted models via LiteLLM. 9 additional vendors also recognised: `mistral`, `cohere`, `azure`, `meta`, `deepseek`, `fireworks`, `together`, `perplexity`, `xai`.
- **`arnes plan --list-templates`** — lists all 13 domain templates.
- **`arnes plan --template <name>`** — forces a specific template, overriding the router.
- **28 new tests** for the Playbook Library (router classification, template rendering, schema validation) + 23 new tests for the review loop (actor-critic iteration, verdict extraction, exhaustion, step-level override).
- **`SECURITY_CREDITS.md`** — stub file referenced by SECURITY.md for crediting vulnerability reporters.
- **Proactive Planner** — `arnes plan` CLI command + MCP tool that researches feasibility, estimates cost, identifies risks, and emits a YAML playbook for human review before any code runs.
- **Seven additional specialists**: `@researcher`, `@security-auditor`, `@devops-engineer`, `@data-scientist`, `@product-manager`, `@market-analyst`, `@cost-estimator`. Each ships a system prompt, a pydantic output schema, and a tool allowlist.
- **`POST /runs/stream` SSE endpoint** — wires `PlaybookExecutor.stream()` to the MCP HTTP transport. Subscribers receive a finite stream of `event: <type>\ndata: <json>\n\n` frames: one `server_info` up-front, one frame per Thread event, and a final `run_result` frame carrying the aggregate accounting.
- **`docs/ethics.md`** — normative ethics doc covering transparency, user control, and responsible AI use. Versioned `ethics-v1.0`.
- **`docs/comparison.md`** — full feature matrix vs LangChain / CrewAI / OpenAI Agents SDK, with a "decision guide" section.
- **`docs/benchmarks.md`** — HumanEval-style stub documentation. Ships `arnes/benchmarks/humaneval_stub.py` with 3 hand-authored problems + `check()` + `pass_at_k()` helpers.
- **`docs/statistics.md`** — statistical-significance testing methodology (bootstrap CIs, Mann-Whitney U / Welch's t-test / Fisher's exact test, effect-size reporting, Benjamini-Hochberg correction, power analysis).
- **Benchmark runner** (`arnes/benchmarks/`) — `BenchmarkRunner` with multi-seed runs, concurrent execution, and p95 duration reporting. Pluggable `BenchmarkSuite` protocol (`BasicBenchmarkSuite` ships by default).
- **`arnes benchmark` CLI command** — runs the basic suite against a deterministic seeded mock LLM (no network, $0 spend).
- **Streaming in the ReAct tool-use loop** (`Specialist.stream()`) — streaming no longer bypasses tool execution; each iteration can carry `tool_calls` that get executed before the next streaming pass.
- **`Harness.stream_with_audit()`** — returns `(chunks, thread)` tuple so streaming runs leave the same audit trail as non-streaming runs.
- **`arnes run --stream`** and **`arnes stream`** CLI commands for step-level and token-level streaming.
- **Real token-by-token streaming** in `OllamaProvider.stream_complete()` and `LiteLLMProvider.stream_complete()`.
- **2 additional vcrpy cassettes** under `tests/snapshot/cassettes/` (`test_coder_basic.yaml`, `test_reviewer_basic.yaml`). The snapshot suite now covers 3 specialists.
- **MkDocs documentation site** (`mkdocs.yml` + 11 docs pages under `docs/*.md`).
- **`CITATION.cff`** — full Citation File Format metadata for academic citations.
- **`Dockerfile.sandbox`** + `scripts/build-sandbox.sh` for the Tier 1 Docker sandbox image.
- **CodeQL workflow** (`.github/workflows/codeql.yml`) with `security-extended` query suite and weekly schedule.
- **GitHub issue templates**, PR template, `FUNDING.yml`.
- **`.gitattributes`** — normalizes line endings to LF across all platforms to prevent CRLF/LF drift in CI.
- 5 additional `EventType` values now emitted: `MODEL_ROUTED`, `PARALLEL_BRANCH_STARTED`, `PARALLEL_BRANCH_COMPLETED`, `RUN_PAUSED`, `REFUSAL_TRIGGERED`.
- **Streamed tool-call reassembly** — `OllamaProvider.stream_complete()` now accumulates `tool_calls` arriving on any non-final NDJSON line (deduplicated by id) and re-emits the full set, so a call delivered on an earlier chunk and repeated (or not repeated) on `done` survives to the consumer. `LiteLLMProvider.stream_complete()` accumulates `delta.tool_calls` fragments (index/id/name + split JSON `arguments` pieces) and yields a single fully-assembled `tool_calls` chunk after the stream ends.
- **Specialist JSON self-correction** — new opt-in `SpecialistConfig.max_json_retries` (default 0 = unchanged behavior). When JSON output is expected and the model returns a tool-call-free response that does not parse, the specialist feeds the attempt back with a "return ONLY JSON" prompt and re-invokes the model, bounded by `max_json_retries` and `max_iterations`.

### Changed
- **Cross-platform CI** — `pyproject.toml` no longer promotes third-party deprecation warnings to errors (the previous `filterwarnings = ["error"]` policy failed on macOS/Windows due to platform-specific dependency warnings). Tests that used Unix-only APIs (`resource`, `os.geteuid`, bare `os.symlink`) now degrade gracefully on Windows.
- **Windows CI step** — `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` are set on Windows runners so rich/litellm box-drawing characters don't crash on cp1252.
- `arnes/mcp/server.py`: 716 → 447 lines. HTTP transport + security helpers extracted to `arnes/mcp/http.py`.
- `arnes/mcp/sse.py`: 112 → 242 lines. New `playbook_event_stream()` function wraps `PlaybookExecutor.stream()` and converts each event to an SSE frame.
- `arnes/tools/builtin.py`: 668 → 460 lines. All security helpers extracted to `arnes/tools/_security.py` (re-exported here for backwards compatibility).
- `arnes/middleware/cost_guard.py`: 611 → 580 lines. `BudgetExceeded` + `CostBudget` extracted to `arnes/middleware/budget.py`.
- `arnes/cli/main.py`: 774 → 293 lines. Async helpers, mock LLM provider, and scaffolding moved to `helpers.py` / `scaffolding.py`.
- `arnes/playbooks/executor.py`: removed the legacy delegating wrappers. Internal call sites now use the canonical functions from `arnes.playbooks.events` / `arnes.playbooks.template` directly. File went from 1015 → ~720 lines.
- `arnes/specialists/base.py`: `Specialist.stream()` rewritten to participate in the ReAct loop.
- `MANIFESTO.md`: gained a Problem Statement section + a Constructive Vision section. The 10 declarations are unchanged (immutable).
- `README.md`: gained "Why Agentic Harness?", "Who is Agentic Harness for?", "Reproducibility", and "Benchmark results" sections.
- `Thread.append()`: O(N²) → O(1) by mutating in place (8.8x speedup at 1000 events). Documented as append-only, not immutable.
- Sandbox auto-detection: `PlaybookExecutor` detects Docker via `shutil.which("docker")` and enables the sandbox automatically.
- CostGuard 95% pause: now emits `HumanApprovalRequestedEvent` and `RUN_PAUSED` in interactive mode.
- All GitHub Actions pinned to commit SHAs (was floating @v4 tags) for supply chain security.
- `pip-audit` now blocking in CI (was `|| true`).
- Anti-hallucination: hedging detection now skipped in JSON mode.
- DRY: extracted `build_middleware_stack()` helper to centralize the TokenOptimizer → VerificationLayer → CostGuard wrapping order.
- `arnes stream` CLI now uses `Harness.stream_with_audit()` + `Thread.to_markdown()` for consistency with the rest of the audit-log system.

### Fixed
- TokenOptimizer cache_key now includes `response_schema` (was cache poisoning across schemas).
- `aiohttp` added to `mcp` optional dependencies (was ImportError at runtime).
- Parallel-step template resolution: outputs now wrapped in `{"output": ...}` structure.
- Middleware double-wrapping: replaced broken `hasattr(provider, "_provider")` with `_arnes_wrapped` marker.
- CostGuard 95% pause was a no-op (`_paused = True` then immediately `False`).
- HITL fingerprint was computed but never compared against approved set.
- MCP server path traversal in `_validate_playbook` and `_list_playbooks`.
- Windows CI: cross-platform PATH, `SYSTEMROOT`/`COMSPEC`, `asyncio.to_thread` for stdin.
- 133 ruff lint errors → 0. 50 mypy --strict errors → 0.
- Dangling symlink escape: `is_symlink()` without `exists()` check (catches dangling symlinks that point outside `working_dir`).
- SSRF: IP pinning with Host header + SNI to prevent DNS rebinding.
- Shell regex: added `python -c`, `eval`, `exec`, `find -delete`, `base64 -d` patterns.
- Double-call bug in `arnes stream`: the CLI invoked the specialist's `stream()` twice on the same input, doubling cost. Fixed by capturing the streamed chunks into a single async iterator and replaying them for both the terminal and the run-log writer.

### Removed
- `PUBLISHING_GUIDE.md` — internal launch playbook; not user-facing.
- `docs/audits/` — historical evaluation reports; not user-facing.
- `docs/audits.md` — index page for the removed audit reports.

## [0.1.0a1] — 2026-07-28

### Added
- **Core**: Thread (append-only event log) + stateless reducer pattern.
- **Events**: 14 typed events (UserMessage, AssistantMessage, ToolCall, ToolResult, StepStarted, StepCompleted, StepFailed, ConditionalBranch, HumanApprovalRequested, HumanApprovalReceived, CostThreshold, RunCompleted, RunFailed).
- **Tools**: 5 built-in tools (shell, http, fs_read, fs_write, human_approval) with SSRF protection, path traversal protection, symlink escape detection, and dangerous command blocking.
- **Specialists**: 5 pre-built specialists (@planner, @coder, @reviewer, @tester, @debugger) with system prompts, structured output schemas, and ReAct tool-use loop.
- **Playbook DSL**: YAML declarative language compiled to DAG. Supports conditional branches (`if_not_met`), parallel branches, retry policies, HITL gates.
- **Playbook Compiler**: alias-based key translation, semantic validation, helpful error messages.
- **Playbook Executor**: async DAG executor with thread event tracking, template resolution (`{{ steps.X.output }}`), conditional branch handling.
- **LLM Providers**: vendor-neutral abstraction via LiteLLM. Default: Ollama (local, $0). Supports Anthropic, OpenAI, Google, Groq.
- **Token Optimizer middleware**: model routing (simple tasks → cheap model) + semantic cache (LRU eviction, TTL).
- **Verification Layer middleware**: structured output validation + refusal pattern (hedging detection forces "I don't know" over fabrication).
- **Cost Guard middleware**: hierarchical budget (org → project → agent → task), circuit breaker temporal (max USD/min), HITL pause at 95%, hard stop at 100%.
- **MCP Server**: stdio transport, exposes 4 tools (`arnes_run_playbook`, `arnes_list_specialists`, `arnes_list_playbooks`, `arnes_validate_playbook`) for Claude Desktop, Cursor, Cline, Zed.
- **CLI**: `arnes init`, `arnes run`, `arnes lint`, `arnes eval`, `arnes list specialists`, `arnes list playbooks`, `arnes mcp serve`.
- **Playbooks**: curated examples (hello-world, debug-python-issue, audit-pr, write-feature-tdd, and more).
- **Tests**: 74 tests covering thread, events, tools, middleware, specialists, playbooks, executor. Coverage: 66%.
- **Docs**: MANIFESTO.md (10 immutable declarations), README.md, CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md, AGENTS.md, CLAUDE.md.
- **CI/CD**: GitHub Actions for test matrix (Python 3.11/3.12/3.13 × Ubuntu/macOS/Windows), security scans (bandit + pip-audit), package build.

### Security
- Path traversal protection on fs_read/fs_write tools.
- SSRF protection on http tool with full DNS resolution (blocks localhost, private IPs, cloud metadata endpoints, DNS rebinding).
- Symlink escape detection on filesystem tools.
- Dangerous command blocking on shell tool (rm -rf /, fork bombs, mkfs, reverse shells, etc).
- Secret filtering: API keys in `args.env` are stripped before passing to subprocess.
- argsFingerprint on tool calls: HITL can detect rug-pull (LLM asking approval with args X but executing with args Y).
- Budget enforcement prevents denial-of-wallet attacks.
- `ARNES_DEV_MODE` gate: local shell execution requires explicit `ARNES_DEV_MODE=1` env var (double-gate with `sandbox_enabled`).

### Known Limitations (v0.1)
- Sandbox Docker Tier 1 is auto-detected but opt-in (shell executes locally in dev mode with `ARNES_DEV_MODE=1`).
- HITL gates auto-reject in non-interactive mode (real HITL via MCP coming in v0.2).
- No PyPI release yet (alpha tag only).

[Unreleased]: https://github.com/frangelbarrera/agentic-harness/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/frangelbarrera/agentic-harness/releases/tag/v0.1.0a1
