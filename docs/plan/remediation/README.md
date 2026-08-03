# Remediation planning documents

Long-form audits and execution prompts, kept out of the workspace root so that
the root holds only files a reader of the repository is expected to open first.
Nothing here is generated, and nothing here is authoritative about the current
state of the tree: for that, read `docs/PROJECT_STATUS.md` and the per-task
directories under `audit-evidence/`.

| Document | What it is | Still driving work? |
|---|---|---|
| [foundation-first-remediation-prompt.md](foundation-first-remediation-prompt.md) | The FND-000..FND-005 remediation brief the current sequence follows | Yes - FND-005 is active |
| [codebase-truth-audit.md](codebase-truth-audit.md) | Source audit that established the starting-state contradictions | Historical input to FND-000 |
| [next-steps-execution-prompt.md](next-steps-execution-prompt.md) | The execution prompt that preceded the foundation-first brief | Superseded |
| [master-implementation-plan.md](master-implementation-plan.md) | 0.1 to 1.0 plan across every subsystem | Deferred until FND-005 completes |
| [master-implementation-plan-security-audited.md](master-implementation-plan-security-audited.md) | The same plan with a security review folded in | Deferred until FND-005 completes |

The master plans describe work the foundation-first brief explicitly defers -
compiler optimization, accelerator breadth, distributed execution. Reading them
as a to-do list before FND-005 is finished inverts the dependency order the
brief exists to enforce.
