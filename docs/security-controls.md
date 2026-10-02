# Security controls

Agent runs should treat prompts, playbooks, tool output, MCP responses, and repository content as untrusted input. Controls should cover prompt injection, tool permissions, exfiltration, cost limits, cancellation, log redaction, and isolation.

This document describes control objectives; each implemented control should link to its test or CI evidence. Do not use real secrets or sensitive production data in examples, snapshots, or evaluation corpora.
