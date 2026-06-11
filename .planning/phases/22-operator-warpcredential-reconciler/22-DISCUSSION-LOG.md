# Phase 22: Operator WarpCredential Reconciler - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-11
**Phase:** 22-operator-warpcredential-reconciler
**Areas discussed:** Secret write mechanism, Handler registration pattern, Derivation logic structure

---

## Secret Write Mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| kr8s | Use kr8s.objects.Secret for create/patch — consistent with how Secrets are already read, pure Python, mockable in unit tests without subprocess | ✓ |
| kubectl apply via subprocess | Write Secret manifest to tempfile and call kubectl apply -f — consistent with kubernetesManifest path but harder to mock and briefly writes secret to disk | |
| kubernetes-client (k8s_client) | Use the kubernetes Python client already imported in main.py — more verbose | |

**User's choice:** kr8s

---

### Idempotency handling

| Option | Description | Selected |
|--------|-------------|----------|
| try create, except AlreadyExists → patch | Standard Kubernetes controller create-or-update pattern | ✓ |
| Always patch (server-side apply) | Cleaner but less tested in this codebase | |

**User's choice:** try create, except AlreadyExists → patch

---

## Handler Registration Pattern

| Option | Description | Selected |
|--------|-------------|----------|
| create + update + resume + delete | Adds @kopf.on.resume so operator restart restores derived secrets without a CR edit — full OPS-09 coverage | ✓ |
| create + update + delete only | Mirrors existing WekaAppStore pattern; OPS-09 only satisfied on CR mutation, not restart | |

**User's choice:** create + update + resume + delete

---

### Handler function structure

| Option | Description | Selected |
|--------|-------------|----------|
| Single reconcile function, three decorators | One reconcile_warpcredential() decorated with @kopf.on.create, @kopf.on.update, @kopf.on.resume — no code duplication | ✓ |
| Separate functions per event | Three separate kopf handler functions — more verbose | |

**User's choice:** Single reconcile function, three decorators

---

## Derivation Logic Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Pure helper functions per type | Extract _derive_ngc_payloads(), _derive_hf_payload(), _derive_weka_payload() — testable without kr8s mocking | ✓ |
| Inline in handler | Derivation logic inside reconcile_warpcredential() — requires full kr8s mock to reach derivation code in tests | |

**User's choice:** Pure helper functions per type

---

### NGC auth field encoding

| Option | Description | Selected |
|--------|-------------|----------|
| base64("$oauthtoken:<key>") standard encoding | Python base64.b64encode with padding — what nvcr.io expects per Docker config spec | ✓ |
| You decide | Leave to planner | |

**User's choice:** Standard base64 with padding

---

## Claude's Discretion

- Exact placement of `_derive_*` helpers in `operator_module/main.py`
- Error message wording (within Phase 18 format conventions)
- Weka-storage secretRef key structure — researcher to clarify from PRD whether `secretRef.key` provides the token only and username is a separate field

## Deferred Ideas

None.
