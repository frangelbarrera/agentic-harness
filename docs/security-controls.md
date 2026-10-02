**Maintainer:** Frangel Raúl Crespo Barrera
**Last verified:** 2026-10-02
**Scope:** prompts, YAML playbooks, tools, MCP responses, cost controls, cancellation, logs, and isolation.

| Field | Current record |
|---|---|
| Status | Mature CI and security automation exist; control-to-test mapping remains a review task. |
| Evidence | `tests/unit/`, `tests/integration/`, `tests/stress/`, `docs/ethics.md`, `docs/mcp-server.md`, `.github/workflows/ci.yml`, `.github/workflows/codeql.yml`. |
| Verification | Run the existing unit/integration/stress suites and inspect the CI/CodeQL results. |
| Owner | Repository owner maintains policy and test evidence. |
| Limitations | Controls described here are objectives unless a linked test or workflow demonstrates them. |

Treat prompts, playbooks, tool output, MCP responses, and repository content as untrusted. Do not use real secrets or production data in examples, snapshots, or evaluation corpora.
