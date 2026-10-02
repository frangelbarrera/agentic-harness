<!--
  Social preview metadata. Set the repo's social preview image in
  GitHub Settings → Social preview. The Open Graph / Twitter tags below
  are picked up by some third-party renderers that parse the raw README.
-->
<meta property="og:title" content="Agentic Harness — The Open Agent Harness" />
<meta property="og:description" content="Write the manual. Agentic Harness compiles it into a team of specialists that follows it to the letter." />
<meta property="og:url" content="https://github.com/frangelbarrera/agentic-harness" />
<meta property="og:type" content="website" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="Agentic Harness — The Open Agent Harness" />
<meta name="twitter:description" content="Write the manual. Agentic Harness compiles it into a team of specialists that follows it to the letter." />

<div align="center">

<img src="docs/logo-agentic-harness.png" alt="Agentic Harness logo" width="200" />

# Agentic Harness

### The Open Agent Harness

**Write the manual. Agentic Harness compiles it into a team of specialists that follows it to the letter.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/frangelbarrera/agentic-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/frangelbarrera/agentic-harness/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/arnes?color=blue)](https://pypi.org/project/arnes/)

[Manifesto](MANIFESTO.md) · [Documentation](https://github.com/frangelbarrera/agentic-harness#readme) · [Examples](examples/) · [Contributing](CONTRIBUTING.md)

</div>

<p align="center">
  <img src="docs/agentic-harness-demo.gif" alt="Agentic Harness demo" width="700" />
</p>

---

> **If your framework needs a debugger for your debugger, it is the wrong framework.**

## Quick install

```bash
pip install agentic-harness
```

## Why Agentic Harness?

Most agent frameworks are black boxes: you can't see the prompts, can't control the costs, and can't audit what happened. Agentic Harness fixes this with three principles:

1. **Transparency** — Every prompt, decision, and cost is logged to a markdown audit log you can `git diff`.
2. **Cost control** — Hierarchical budget enforcement with circuit breaker. An agent can't burn your API budget silently.
3. **Vendor neutrality** — Default model is local (Ollama, $0). Switching providers is one line. No vendor lock-in.

## Who is Agentic Harness for?

- **Backend engineers** who need production agents with budgets, audit trails, and compliance
- **ML engineers** who need reproducible benchmarks across providers
- **Researchers** who need citation-ready, replayable experiments
- **DevOps teams** who need MCP server integration with auth and rate limiting

Not for you if you want a visual builder, hosted SaaS, or multi-agent crews (v0.4+).

---

## What it looks like

```
    ___
   /   |  _________ ___   _______
  / /| | / ___/ __ `__ \ / ___/ /
 / ___ |/ /  / / / / / // /__/ /
/_/  |_/_/  /_/ /_/ /_/ \____/_/
        The Open Agent Harness
```

A manual in YAML:

```yaml
# manuals/audit-pr.yaml
name: audit-pr
objective: Audit a Pull Request in a structured way
budget_usd: 0.50

steps:
  - id: read_diff
    specialist: "@reviewer"
    input:
      pr_number: 1234
      repo: "my-org/my-repo"
      focus: "Read the diff and structure it for analysis"

  - id: security_audit
    specialist: "@reviewer"
    input: "{{ steps.read_diff.output }}"
    focus: "Security review: auth flows, SQL injection, XSS, path traversal"
    if_not_met:
      action: call
      specialist: "@reviewer"
      input:
        focus: "Comment that the PR is blocked by security review"

  - id: parallel
    parallel:
      - id: lint
        specialist: "@reviewer"
        input:
          code: "{{ steps.read_diff.output }}"
          focus: "Code quality: idioms, naming, complexity"
      - id: tests
        specialist: "@tester"
        input:
          code: "{{ steps.read_diff.output }}"
          focus: "Verify tests cover the PR changes"

  - id: synthesis
    specialist: "@reviewer"
    input:
      diff: "{{ steps.read_diff.output }}"
      security: "{{ steps.security_audit.output }}"
      lint: "{{ steps.parallel.lint.output }}"
      tests: "{{ steps.parallel.tests.output }}"
      focus: "Synthesize into a final verdict: approve / request_changes / reject"
```

You run it (mock LLM, no network, $0 cost):

```bash
$ arnes run manuals/hello-world.yaml --mock
```

Agentic Harness compiles the manual into a DAG, wakes the specialists in sequence,
applies token optimization and verification layer on every LLM call, and
returns:

```
╭────────────────────────────────────────────────────────────────────╮
│ Agentic Harness — Executing playbook                                         │
│   Name: hello-world                                                │
│   Objective: Demonstrate the basic Agentic Harness flow with a simple manual │
│   Model: ollama/llama3.2                                           │
│   Budget: $0.50                                                    │
╰────────────────────────────────────────────────────────────────────╯
2026-07-30 16:42:44 [info] llm_call_tracked  budget=0.5 cost_usd=0.0 \
      model=ollama/llama3.2 tokens_in=335 tokens_out=15 total_spent=0.0
2026-07-30 16:42:44 [info] llm_call_tracked  budget=0.5 cost_usd=0.0 \
      model=ollama/llama3.2 tokens_in=370 tokens_out=38 total_spent=0.0

✅ Manual executed

Steps executed: 2
Steps failed: 0
Duration: 0.01s
Tokens in/out: 705/53
Total cost: $0.0000

Run log saved to: arnes-run-hello-world-20260730-164244.md
```

The run log is a markdown file with every step, every decision, every prompt
sent, every response received. You can diff it, version it, share it:

````markdown
# Audit log Agentic Harness — Thread 0b6ac82e-2600-42f5-a6ca-62e016df7961

**Total events:** 7

## [2026-07-30T16:42:44] step_started
**Step:** `plan`  ·  **Specialist:** `@planner`

## [2026-07-30T16:42:44] assistant_message
**Step:** `plan`  ·  **Specialist:** `@planner`
```json
{
  "model": "ollama/llama3.2",
  "tokens_in": 335,
  "tokens_out": 15,
  "cost_usd": 0.0,
  "cached": false
}
```

## [2026-07-30T16:42:44] step_completed
...
````

Want to see the whole flow end-to-end? Run the narrated demo script:

```bash
./scripts/demo.sh            # print to terminal
./scripts/demo.sh --record demo.tape && vhs demo.tape   # render a GIF
```

---

## Features

| Category | Feature | Status |
|---|---|---|
| **Agent loop** | Stateless reducer `(state, event) → state` | ✅ v0.1 |
| | ReAct tool-use loop in specialists | ✅ v0.1 |
| | AG-UI streaming compatible | 🚧 v0.2 |
| **Specialists** | 12 pre-built (planner, coder, reviewer, tester, debugger, researcher, security-auditor, devops-engineer, data-scientist, product-manager, market-analyst, cost-estimator) | ✅ v0.1 |
| | Playbook Library with 13 domain templates + TaskRouter | ✅ v0.1 |
| **Playbook DSL** | Declarative YAML compiled to DAG | ✅ v0.1 |
| | Conditional branches (`if_not_met`) | ✅ v0.1 |
| | Parallel branches (true `asyncio.gather`) | ✅ v0.1 |
| | Retry with backoff | 🚧 v0.2 (schema defined, execution pending) |
| | HITL gates (pause and request approval) | ⚠️ v0.1 (auto-reject in non-interactive) |
| | Actor-critic review loop (`--loops`, `step.review`) | ✅ v0.1 |
| **MCP** | Agentic Harness as MCP server (Claude Desktop, Cursor, Cline, Zed) | ✅ v0.1 |
| | Agentic Harness as MCP client (consume external MCP servers) | 🚧 v0.2 |
| | HTTP/SSE transport | 🚧 v0.2 (stdio only in v0.1) |
| **Token Optimization** | Automatic model routing by complexity | ✅ v0.1 |
| | Semantic cache | ✅ v0.1 |
| | Context compaction | 🚧 v0.2 |
| | Few-shot pruning | 🚧 v0.3 |
| **Verification Layer** | Structured outputs with pydantic | ✅ v0.1 |
| | Refusal pattern (no hallucination, says "I don't know") | ✅ v0.1 |
| | Confidence gate | 🚧 v0.2 |
| | Critic loop (actor-critic iterative refinement) | ✅ v0.1 |
| | Grounding RAG optional | 🚧 v0.4 |
| **Cost Guard** | Hierarchical budget (org → project → agent → task) | ✅ v0.1 |
| | Temporal circuit breaker (max USD/min) | ✅ v0.1 |
| | Automatic model fallback | ✅ v0.1 |
| | Cost HITL (pause at X% exceeded) | ⚠️ v0.1 (log warning, auto-pause pending) |
| **Sandbox** | Docker hardened (Tier 1 dev-local) | ✅ v0.1 (auto-detected when `docker` is on PATH; falls back to gated local exec via `ARNES_DEV_MODE=1`) |
| | gVisor (Tier 2 production) | 🚧 v0.4 |
| **Multi-agent** | Single-agent default | ✅ v0.1 |
| | Crew (sequential/hierarchical) | 🚧 v0.4 |
| | A2A with trust | 🚧 v0.5 |
| **Observability** | Structured event log | ✅ v0.1 |
| | Auditable markdown audit log | ✅ v0.1 |
| | OpenTelemetry exporter | 🚧 v0.3 |
| **Benchmarks** | BenchmarkRunner with multi-seed + concurrent + p95 | ✅ v0.1 |

---

## Agentic Harness vs the rest

| Dimension | LangChain | CrewAI | OpenAI Agents SDK | **Agentic Harness** |
|---|---|---|---|---|
| How you define agents | Python procedural | `Agent/Crew/Task` classes | `@agent` decorator | **Declarative YAML** |
| Distribution | pip library | pip library | pip library (OpenAI-only) | **MCP server + library** |
| Pre-built specialists | ❌ | ❌ | ❌ | **✅ 12 ready** |
| Curated playbooks | ❌ | ❌ | ❌ | **✅ 10 manuals + 13 domain templates** |
| Token optimization | Manual | ❌ | ❌ | **✅ Automatic middleware** |
| Anti-hallucination | DIY | ❌ | ❌ | **✅ 3 layers (structured + refusal + actor-critic)** |
| Budget enforcement | `max_tokens` basic | `max_tokens` basic | ❌ | **✅ Hierarchical + circuit breaker** |
| Vendor-neutral | Partial | ✅ | ❌ | **✅ 100% (default Ollama local)** |
| Prompts visible | ❌ | ❌ | ❌ | **✅ Files on disk** |

---

## Alignment with the 12-factor-agents manifesto

Agentic Harness aligns explicitly with the [12 factors](https://github.com/humanlayer/12-factor-agents):

| Factor | Description | Agentic Harness |
|---|---|---|
| 1 | Natural language > structured language | ✅ Declarative YAML |
| 2 | Tools are structured outputs | ✅ Pydantic schemas |
| 3 | Give agents composable, discrete tools | ✅ Specialist registry |
| 4 | Agents are switching loops, not while loops | ✅ Event-driven reducer |
| 5 | Simple but powerful primitives | ✅ Thread + Specialist + Tool |
| 6 | Use the right tool for the job | ✅ Model routing |
| 7 | Humans are tools, not gates | ✅ HITL as a typed tool call |
| 8 | Make agents easy to debug | ✅ Markdown audit log |
| 9 | Make agents observable | ✅ Event log + OTel (v0.3) |
| 10 | Replayable from any point | ✅ Stateless reducer + checkpoint |
| 11 | Be a state machine, not a DAG | ⚠️ We are a DAG by design (declarative) |
| 12 | Deploy as a server, not a library | ✅ Native MCP server |

---

## Installation

```bash
pip install agentic-harness
```

Or install from source for development:

```bash
git clone https://github.com/frangelbarrera/agentic-harness.git
cd agentic-harness
pip install -e ".[dev]"
```

## Quickstart (60 seconds)

```bash
# 1. Install (see Installation above)

# 2. Create your first manual
arnes init --manual hello-world

# 3. Run it with the mock LLM (no network, $0 cost)
arnes run manuals/hello-world.yaml --mock

# 4. Stream a specialist's response token-by-token
arnes stream @planner --task "Plan a blog post about Agentic Harness" --mock

# 5. Run it with Ollama local (free, requires `ollama pull llama3.2`)
arnes run manuals/hello-world.yaml

# 6. Stream playbook step events as they complete
arnes run manuals/hello-world.yaml --mock --stream

# 7. Benchmark every playbook (multi-seed, p95, concurrent)
arnes benchmark --seeds 5 --concurrent 4
```

If you do not have Ollama installed, Agentic Harness detects it and guides you. To use
Anthropic/OpenAI, set the env var and Agentic Harness does the rest:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
arnes run manuals/audit-pr.yaml --model anthropic/claude-sonnet-4-20250514
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│   YOU (Claude Desktop / Cursor / CLI / Cline / Zed)            │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│   AGENTIC HARNESS MCP SERVER (1 install, 4 tools)                       │
│   run · list · events · resume                                │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│   PLAYBOOK RUNTIME                                            │
│   YAML → Pydantic → DAG → Executor (conditional/parallel/HITL)│
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│   SPECIALIST REGISTRY (12 pre-built agents)                    │
│   planner · coder · reviewer · tester · debugger ·            │
│   researcher · security-auditor · devops-engineer ·           │
│   data-scientist · product-manager · market-analyst ·         │
│   cost-estimator                                               │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│   CROSS-CUTTING MIDDLEWARE (all LLM calls pass through it)    │
│   🧠 Token Optimizer  🛡️ Verification  💰 Cost Guard          │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│   LLM PROVIDERS (vendor-neutral, default Ollama local)        │
│   ollama · openrouter · anthropic · openai · google · groq   │
│   mistral · cohere · azure · meta · deepseek · fireworks ·   │
│   together · perplexity · xai                                 │
└──────────────────────────────────────────────────────────────┘
```

---

## Benchmark

Agentic Harness ships a built-in benchmark runner that executes every playbook in
`manuals/` against a deterministic seeded mock LLM (no network, $0 spend)
and reports per-playbook success rate, avg/p95 duration, tokens, and cost.
Multi-seed runs give you statistical significance; concurrent runs let
you stress-test the executor's parallel-branch path.

```bash
# 1 seed, 1 concurrent (default — quick smoke test)
arnes benchmark

# 5 seeds per playbook (catch flaky playbooks)
arnes benchmark --seeds 5

# 4 playbooks at once (stress the asyncio.gather path)
arnes benchmark --concurrent 4

# Combined: 5 seeds × 4-way parallelism
arnes benchmark --seeds 5 --concurrent 4
```

Example output:

```
              Benchmark Results — basic suite
┏━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Playbook       ┃ Runs ┃ Success ┃ Avg dur   ┃ P95 dur   ┃ Avg tokens ┃ Avg cost  ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ hello-world    │    5 │    100% │    0.0089 │    0.0112 │        705 │ $0.000000 │
│ audit-pr       │    5 │    100% │    0.0241 │    0.0298 │       2104 │ $0.000000 │
│ debug-python   │    5 │    100% │    0.0187 │    0.0233 │       1583 │ $0.000000 │
│ write-feature  │    5 │    100% │    0.0312 │    0.0367 │       2431 │ $0.000000 │
└────────────────┴──────┴─────────┴───────────┴───────────┴────────────┴───────────┘

Overall: success=100%, avg_dur=0.0207s, avg_tokens=1706, avg_cost=$0.000000

Results saved to: benchmark-results.json
```

The JSON dump (default: `benchmark-results.json`, override with `--output`)
is suitable for diffing across commits or pasting into a PR description.

### Benchmark results (sample run)

The numbers below are from the bundled reference run
(`docs/benchmark-results.json`, captured 2026-07-30 on the v0.1.0a1
mock LLM, 2 seeds × 2-way concurrency, 10 playbooks, 20 total runs).
The mock LLM is deterministic, so re-running with the same seeds on
the same commit reproduces these numbers bit-for-bit.

| Playbook                  | Runs | Success | Avg dur (s) | P95 dur (s) | Avg tok in | Avg tok out |
|---------------------------|------|---------|-------------|-------------|------------|-------------|
| `audit-pr`                  | 2    | 100 %   | 0.00783     | 0.01040     | 1 172      | 78          |
| `code-review-security`      | 2    | 100 %   | 0.00209     | 0.00219     | 1 754      | 130         |
| `debug-python-issue`        | 2    | 100 %   | 0.00284     | 0.00343     | 1 329      | 150         |
| `hello-world`               | 2    | 100 %   | 0.00130     | 0.00133     | 705        | 74          |
| `incident-postmortem`       | 2    | 100 %   | 0.00320     | 0.00324     | 2 196      | 234         |
| `migrate-config`            | 2    | 100 %   | 0.00277     | 0.00287     | 1 481      | 157         |
| `refactor-extract-function` | 2    | 100 %   | 0.00266     | 0.00275     | 1 481      | 157         |
| `summarize-paper`           | 2    | 100 %   | 0.00148     | 0.00152     | 1 399      | 101         |
| `write-blog-post`           | 2    | 100 %   | 0.00261     | 0.00264     | 1 530      | 151         |
| `write-feature-tdd`         | 2    | 100 %   | 0.00363     | 0.00376     | 2 111      | 215         |
| **Overall**                 | 20   | **100 %** | **0.00304** | —           | **1 515**  | **144**     |

**Cost:** `$0.000000` across all 20 runs (mock LLM, no network).

Why the durations are tiny: the mock LLM has no network round-trip,
no model inference latency, no token streaming. Real-LLM runs (with
`--model openai/gpt-4o` etc.) will be orders of magnitude slower but
should preserve the relative ordering of playbooks (parallel branches
remain faster than sequential ones of equivalent work).

The full JSON (with per-seed, per-playbook, and per-step results) is
checked into the repo so any regression in playbook success rate,
token usage, or cost shows up in `git diff`.

---

## Reproducibility

Agentic Harness is built so that the *same* inputs produce the *same* outputs,
byte-for-byte, on every run. This is a hard requirement for both
production audit and scientific reproducibility.

**What is reproducible**

- **Mock-LLM runs.** The bundled `_SchemaValidMockLLMProvider` is fully
  deterministic: same input → same output, no time-of-day variation,
  no network calls, no API keys. `arnes run manuals/hello-world.yaml
  --mock` produces a bit-for-byte identical audit log across runs,
  machines, and OSes.
- **Benchmark results.** `arnes benchmark --seeds N` runs each
  playbook N times with deterministic seeds. The resulting
  `benchmark-results.json` is diffable across commits — a regression
  in playbook success rate, token count, or p95 duration is visible
  in `git diff`.
- **vcrpy cassettes.** Real-LLM HTTP traffic is recorded once with
  vcrpy and replayed on every test run. Tests that exercise
  `@planner`, `@coder`, and `@reviewer` against `openai/gpt-4o`
  replay the cassette — no API spend, no network, fully deterministic.
  See `docs/benchmarking.md` for the cassette inventory and the
  regeneration procedure.
- **Thread replay.** The stateless reducer pattern `(state, event) →
  state` means any Thread can be replayed from its event log. Given
  the same event sequence, the final state is identical. This is the
  primitive that v0.2 will use for HITL resume-after-pause and the
  primitive that v0.3 will use for episodic memory.

**What is NOT reproducible (yet)**

- **Real-LLM runs.** OpenAI / Anthropic / Ollama models are
  non-deterministic by design (temperature > 0, model-side sampling).
  Agentic Harness cannot make a non-deterministic model deterministic. What
  Agentic Harness *can* do is record every real-LLM call into the audit log
  so a non-deterministic run is at least *auditable* after the fact.
- **Real-time wall-clock durations.** Durations depend on machine
  load, network latency, and OS scheduling. The benchmark harness
  reports p95 *relative* durations (which are stable across runs on
  the same machine) but absolute durations are not portable.
- **Statistical significance.** v0.1 reports p95 only. Multi-seed
  runs give you the raw samples; running a Mann-Whitney U test or
  bootstrap CI on them is the caller's responsibility today. See
  `docs/statistics.md` for the recommended methodology and the
  v0.2 plan to ship a `arnes benchmark --stats` flag that does the
  analysis in-process.

**Citation**

If you use Agentic Harness in published research, cite the version you used
(see [CITATION.cff](CITATION.cff)) and include the run log +
`benchmark-results.json` from your experimental runs as supplementary
material. The run log is the auditable artifact that lets a reviewer
reproduce your agent's behaviour step-by-step.

---

## Roadmap

**v0.1 (now)** — 12 specialists, 13 domain templates, playbook DSL, MCP server, Cost Guard, review loops, 14 LLM vendors, 470+ tests.

**Next** — Retry execution, HITL resume, HTTP/SSE transport, context compaction, multi-agent crews.

---

## Community

- **Discussions:** [GitHub Discussions](https://github.com/frangelbarrera/agentic-harness/discussions)
- **Issues:** [Bug reports and feature requests](https://github.com/frangelbarrera/agentic-harness/issues)
- **Contributing:** read [CONTRIBUTING.md](CONTRIBUTING.md) — we accept PRs from day one.

---

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). TL;DR:

1. Fork + clone
2. `uv sync --all-extras` for dev setup
3. `pre-commit install`
4. Create your branch: `feat/my-feature`
5. Conventional commits: `feat: ...`, `fix: ...`, `docs: ...`
6. `pytest` must pass with >65% coverage
7. Open PR — review within 48h

**Good first issues:** look for issues labeled `good-first-issue`.

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

If you use Agentic Harness in academic research, please cite it. See [CITATION.cff](CITATION.cff) for the preferred citation format.

## Acknowledgments

Agentic Harness stands on the shoulders of:
- [LangGraph](https://github.com/langchain-ai/langgraph) — DAG engine inspiration
- [LiteLLM](https://github.com/BerriAI/litellm) — provider abstraction
- [MCP SDK](https://github.com/modelcontextprotocol/python-sdk) — protocol
- [12-factor-agents](https://github.com/humanlayer/12-factor-agents) — manifesto
- [Pydantic](https://github.com/pydantic/pydantic) — structured data

---

<div align="center">

**[⭐ Star the repo](https://github.com/frangelbarrera/agentic-harness)** if this resonates.

</div>

---

## Known Limitations in v0.1 (Alpha)

This is an **alpha release**. The following features are documented but have
known issues that will be fixed in v0.2:

- **HITL gates** auto-reject in non-interactive mode. Real interactive HITL
  (pausing execution and resuming on human input via the MCP transport)
  comes in v0.2. Until then, calling a HITL-gated tool without
  `interactive=True` returns a structured rejection rather than blocking.
- **LLM streaming** is implemented for all providers. `LLMProvider` declares
  `stream_complete()` (returns `AsyncIterator[LLMResponse]`). `MockLLMProvider`
  yields a single full-response chunk; `OllamaProvider` and `LiteLLMProvider`
  yield real token-by-token chunks. `CostGuard.stream_complete` tracks cost
  on the final chunk. Full per-chunk verification and semantic-cache
  population from streaming lands in v0.2.
- **MCP HTTP transport** is minimal (simple POST endpoint, no SSE). It
  *does* ship with bearer-token auth (`ARNES_MCP_TOKEN`), per-IP rate
  limiting (100 req/min), and a 1 MiB request size cap — but for production
  use the stdio transport is still recommended until full HTTP/SSE lands
  in v0.2.
- **Retry policy** schema is defined but execution is not yet implemented.
- **Context compaction** and **few-shot pruning** are not yet implemented.
- **Confidence gate** is not yet implemented (the actor-critic review loop IS implemented via `--loops`).

**What does work in v0.1:**
- ✅ Thread + stateless reducer pattern (append-only, O(1) per event)
- ✅ 12 specialists with ReAct tool-use loop
- ✅ Playbook Library with 13 domain templates + TaskRouter
- ✅ Actor-critic review loops (`--loops` flag)
- ✅ Playbook DSL with conditionals and template resolution
- ✅ Parallel branches (true `asyncio.gather` concurrency, isolated Threads)
- ✅ CostGuard with budget enforcement and circuit breaker
- ✅ VerificationLayer with structured outputs and refusal pattern
- ✅ TokenOptimizer with model routing and semantic cache
- ✅ MCP server (stdio transport + minimal HTTP transport with auth/rate limits)
- ✅ CLI (init, run, run --stream, stream, lint, eval, benchmark, list, mcp serve)
- ✅ Docker sandbox auto-detected when `docker` is on PATH (Tier 1 dev-local)
- ✅ SSRF protection with DNS resolution
- ✅ Path traversal + symlink escape detection
- ✅ Secret filtering from subprocess env
- ✅ argsFingerprint for HITL rug-pull detection
- ✅ `mypy --strict` enforced in CI and passing on all source files
- ✅ Test coverage above the 65% PR gate (unit + integration + stress)

## Security controls

See [docs/security-controls.md](docs/security-controls.md).
