# Phase 17: CRD Schema Additive Update - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `17-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-05-06
**Phase:** 17-crd-schema-additive-update
**Areas discussed:** Variable key name enforcement, CRD description format, CRD validation methodology, Chart version bump policy

---

## Variable Key Name Enforcement at Admission

### Q1: Should the CRD itself reject invalid keys, or only the operator?

| Option | Description | Selected |
|--------|-------------|----------|
| CRD-level + operator-level (defense in depth) | `propertyNames: { pattern: ... }` rejects bad keys at apply-time. Phase 18 OP-10 still runs as fallback for cases where admission is bypassed. | ✓ |
| Operator-level only (per OP-10) | Trust Phase 18's `kopf.PermanentError`. CRD documents but doesn't enforce. | |
| CRD-level only (no Phase 18 check) | If admission rejects, OP-10 could be removed. Risk if admission bypassed. | |

**User's choice:** CRD-level + operator-level (defense in depth).

---

### Q2: What pattern for `propertyNames`?

| Option | Description | Selected |
|--------|-------------|----------|
| `^[_a-zA-Z][_a-zA-Z0-9]*$` — exact Python identifier | Matches `string.Template`'s internal rule. Parity at admission and render(). | ✓ |
| `^[a-zA-Z][a-zA-Z0-9]*$` — no underscore | Stricter; surprising to authors expecting `_my_var`. | |
| `^[a-zA-Z_][a-zA-Z0-9_-]*$` — also allow hyphens | Defeats the purpose; `${my-host}` breaks Template. | |

**User's choice:** `^[_a-zA-Z][_a-zA-Z0-9]*$`.

---

### Q3: Keep or relax Phase 18 OP-10 operator-level check?

| Option | Description | Selected |
|--------|-------------|----------|
| Keep OP-10 unchanged | Belt + suspenders. Catches the rare admission-bypass case. ~3 lines of operator code. | ✓ |
| Soften OP-10 to a warning log | Log warning, then proceed. Less defensive. | |
| Drop OP-10 from Phase 18 scope | Smallest Phase 18; admission is the only gate. Risk: cryptic Template ValueError if bypassed. | |

**User's choice:** Keep OP-10 unchanged.

---

## CRD Description Format

### Q1: What format for `description:`?

| Option | Description | Selected |
|--------|-------------|----------|
| Multi-line block scalar covering all 4 requirements | 6–8 lines. Reads well in `kubectl explain`. Matches PRD verbatim. | ✓ |
| Concise single-line + docs link | One-liner. Risk: CRD-02 SC#4 keyword grep fails. | |
| Multi-line block scalar but more concise (3–4 lines) | Compress 4 items into 3–4 lines. Still satisfies CRD-02 if every keyword present. | |

**User's choice:** Multi-line block scalar covering all 4 requirements.

---

### Q2: Exact description content?

| Option | Description | Selected |
|--------|-------------|----------|
| PRD's proposed text + identifier-name line | Use the PRD's exact text plus one explicit `Variable names must match Python identifier syntax: [_a-zA-Z][_a-zA-Z0-9]*.` line. | ✓ |
| Operator-authored fresh wording | Plain prose; risk of subtly missing a keyword. | |
| PRD text only (no identifier-name sentence) | CRD-02 SC#4 would fail. | |

**User's choice:** PRD's proposed text + identifier-name line.

---

## CRD Validation Methodology

### Q1: How to prove admission rules work?

| Option | Description | Selected |
|--------|-------------|----------|
| Live cluster `kubectl --dry-run=server` script | Verify-crd.sh: helm template + kubectl apply --dry-run=server + 4 test CRs. Uses existing EKS cluster. No new deps. | ✓ |
| Python unit test using openapi-schema-validator | Pure unit test, no cluster. Adds dep. | |
| Documented manual verification commands in SUMMARY.md | Lowest cost; relies on operator running them. | |
| Both: cluster script AND unit test | Most thorough. Two artifacts to maintain. | |

**User's choice:** Live cluster `kubectl --dry-run=server` script.

---

### Q2: Where does the script live? What fixtures?

| Option | Description | Selected |
|--------|-------------|----------|
| `weka-app-store-operator-chart/scripts/verify-crd.sh` + 4 inline fixtures | Co-located with chart. Heredoc CRs for: valid, integer-value, hyphenated-key, no-variables. | ✓ |
| Same script + external YAML fixtures | Cleaner per-fixture diffs; more files to maintain. | |
| Repo-root `scripts/verify-crd.sh` + heredocs | Less discoverable. | |

**User's choice:** `weka-app-store-operator-chart/scripts/verify-crd.sh` + 4 inline fixtures.

---

### Q3: Script also runs `kubectl explain` for CRD-02?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — grep `kubectl explain` for `${VAR}`, `$$`, `${namespace}`, `identifier` | Asserts CRD-02 SC#4 in code. | ✓ |
| No — description visually inspected | Lighter script; manual review for description. | |

**User's choice:** Yes — grep `kubectl explain` output for required keywords.

---

### Q4: Apply for-real or dry-run only?

| Option | Description | Selected |
|--------|-------------|----------|
| Dry-run-only by default; `--apply` flag for real install | Safe to re-run. `--apply` opt-in (required for `kubectl explain`). | ✓ |
| Always apply for-real | Simpler script. Risk: clobbers in-progress test state. | |
| Dry-run-only — skip kubectl explain | Pure dry-run. Drops CRD-02 SC#4 check. | |

**User's choice:** Dry-run-only by default; `--apply` flag for real install.

---

## Chart Version Bump Policy

### Q1: Bump chart version, and how much?

| Option | Description | Selected |
|--------|-------------|----------|
| Bump patch to 0.1.62 in this phase | Additive backward-compat change. Phase 18 will bump again. | ✓ |
| No bump in this phase; batch at end of v5.0 | Defer. Risk: version skew if user pulls early. | |
| Bump minor to 0.2.0 | Inflated for CRD-only change; users can't use variables until Phase 18. | |
| Bump patch AND publish .tgz to docs/ | 0.1.62 + helm package + helm repo index + commit docs/. | |

**User's choice:** Bump patch to 0.1.62 in this phase.

---

### Q2: Publish .tgz in this phase?

| Option | Description | Selected |
|--------|-------------|----------|
| Skip publish in this phase | No `helm package` / no `docs/` mutations. Publish at end of v5.0 or via manual workflow. | ✓ |
| Also publish (helm package + commit docs/) | Available to `helm repo update` users immediately. Adds Helm CLI prerequisite. | |

**User's choice:** Skip publish in this phase.

---

### Q3: CHANGELOG / release-note content?

| Option | Description | Selected |
|--------|-------------|----------|
| No CHANGELOG file in this phase | Defer to future docs/release-prep phase. Git commit message + version bump is the record. | ✓ |
| Add brief annotation in Chart.yaml description: field | Visible in `helm show chart`. Trivial change. | |
| Create new CHANGELOG.md | Bootstraps versioned release notes. Non-trivial scope. | |

**User's choice:** No CHANGELOG file in this phase.

---

## Claude's Discretion

- Exact placement of `variables:` block within rendered schema (Claude picks; goes right after `components:` block ends).
- Whitespace/indent matching existing crd.yaml (2-space indent).
- Script exit-code conventions (any non-zero is fail).
- Stderr-substring matches for dry-run failure cases (Kubernetes' admission error wording can vary slightly between server versions; pick robust substrings).

## Deferred Ideas

- Publishing `.tgz` to `docs/` — end-of-v5.0 milestone work.
- CHANGELOG.md — future docs/release-prep phase.
- README user-facing variable substitution docs — locked to Phase 18 (DOC-01..06).
- CI/automated CRD schema tests — future test-infra phase.
- `status.appStackVariables` observability field — already v51-02.
- Templating `targetNamespace` / operator-control fields — already v51-01.
- Default-value syntax (`${VAR:-default}`) — already v51-03.
