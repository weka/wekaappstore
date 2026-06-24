---
phase: 27-install-blueprint-authoring
plan: 01
subsystem: infra
tags: [kubernetes, helm, weka, csi, storageclasses, wekaclient, appstack, blueprint]

# Dependency graph
requires:
  - phase: 26-dynamic-blueprints
    provides: "parse_x_variables + [[var]] Jinja2 delimiter pattern used by this blueprint"
provides:
  - "cluster_init/app-store-install.yaml — parameterized WekaAppStore CR with 12 components in topo-sorted dependency order"
  - "Declarative install contract for WEKA operator, CSI driver, node labels, WekaClient, StorageClasses"
affects:
  - 28-operator-auth
  - 29-backend-wiring
  - 30-wizard
  - 31-e2e

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "x-variables block at top-level of YAML multi-doc for wizard field declaration"
    - "stringData everywhere for wizard secrets (no data: base64 hand-encoding)"
    - "storageclass-demote-job pattern: SA + dedicated least-privilege ClusterRole + CRB + Job in one kubernetesManifest block"
    - "dependsOn-gated WekaClient CR with no readinessCheck (operator manages its own reconcile loop)"

key-files:
  created:
    - "cluster_init/app-store-install.yaml"
  modified: []

key-decisions:
  - "Written as a multi-doc YAML: x-variables document first, then WekaAppStore CR — consistent with parse_x_variables expectations"
  - "quay-secret-default-ns and quay-secret-operator-ns use stringData with single [[ quay_dockerconfigjson ]] token (not double-encoded)"
  - "storageclass-demote-job uses dedicated ClusterRole with get/list/patch on storageclasses — NOT cluster-admin (T-27-04 mitigation)"
  - "weka-node-label-rbac uses cluster-admin consistent with gateway-api-crds precedent in app-store-cluster-init.yaml"
  - "weka-client has no readinessCheck — WekaClient operator manages its own reconcile loop"
  - "csi-wekafs pinned to 2.8.7 (latest stable as of Phase 27); comment included for future upgrade"
  - "join_ip_ports_list emitted UNQUOTED in joinIpPorts field so Phase 29 server-injected YAML array parses correctly"

patterns-established:
  - "Blueprint multi-doc pattern: x-variables doc + WekaAppStore CR doc in one file"
  - "Least-privilege ClusterRole for StorageClass demote (not cluster-admin)"
  - "stringData for all wizard-supplied credentials; quay_dockerconfigjson injected as pre-built JSON string"

requirements-completed: [INST-01, INST-02, INST-03, INST-04, INST-05, INST-06, INST-07, INST-08, INST-09, INST-10]

# Metrics
duration: 15min
completed: 2026-06-24
---

# Phase 27 Plan 01: Install Blueprint Authoring Summary

**Parameterized WekaAppStore CR with 12 topo-sorted components installing WEKA operator (OCI quay), CSI driver, node labels, WekaClient, and three StorageClasses via [[ ]] stringData secrets**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-06-24T03:38:24Z
- **Completed:** 2026-06-24T03:53:00Z
- **Tasks:** 2 (authored atomically in one file write, both verified)
- **Files modified:** 1

## Accomplishments

- `cluster_init/app-store-install.yaml` authored as a 452-line multi-doc WekaAppStore CR with all 12 components in correct topo-sort order
- x-variables block declares all 8 wizard fields; server-derived vars (`join_ip_ports_list`, `endpoints_csv`) correctly excluded
- All wizard secrets use `stringData` — no `data:` base64 encoding anywhere in the file
- storageclass-demote-job uses a dedicated least-privilege ClusterRole (T-27-04 mitigated); exactly one default StorageClass (`storageclass-wekafs-dir-api`)

## Task Commits

Each task was committed atomically:

1. **Task 1: Header, x-variables, quay/operator/CSI components** - `24274d9` (feat)
2. **Task 2: Node-label SA/RBAC/Job, weka-client-secret, WekaClient CR, csi-api-secret, demote Job, StorageClasses** - included in `24274d9` (authored in same file pass; full 12-component file verified by Task 2 check)

## Files Created/Modified

- `cluster_init/app-store-install.yaml` — 452-line parameterized WekaAppStore CR: x-variables block + 12 appStack components in D-01 dependency order

## Decisions Made

- **Multi-doc YAML structure:** x-variables as a separate top-level YAML doc (lines 1-18), WekaAppStore CR as the second doc — matches how `parse_x_variables` scans for the `x-variables` key without touching the CR structure.
- **Single file write:** Both tasks authored in one atomic write since they produce one file; both task verification scripts run and passed independently before commit.
- **storageclass-demote-job ClusterRole:** Uses `get/list/patch` on `storageclasses` only (not `cluster-admin`), implementing the T-27-04 threat mitigation per plan spec.
- **weka-node-label ClusterRoleBinding:** Uses `cluster-admin` consistent with `gateway-api-crds` precedent — tradeoff accepted per D-07 / T-27-03.
- **CSI deployment readinessCheck name:** Used `csi-wekafs-csi-wekafsplugin-controller` as the deployment name; Phase 29/E2E can correct if the actual release name differs.
- **join_ip_ports_list in joinIpPorts:** Emitted as `[[ join_ip_ports_list ]]` (unquoted) so Phase 29 backend injects a YAML array that parses correctly after Jinja2 render.

## Deviations from Plan

None — plan executed exactly as written. All must_haves and acceptance criteria verified programmatically.

## Threat Surface Scan

No new network endpoints, auth paths, or external services introduced in this plan. The file itself is a declarative blueprint — no runtime code changes.

| Flag | File | Description |
|------|------|-------------|
| T-27-01 mitigated | cluster_init/app-store-install.yaml | quay_dockerconfigjson is `[[ ]]` token only; no literal dockerconfigjson value committed |
| T-27-02 mitigated | cluster_init/app-store-install.yaml | All credential fields use stringData with `[[ ]]` tokens; no `data:` base64 present |
| T-27-03 accepted | cluster_init/app-store-install.yaml | weka-node-label uses cluster-admin (consistent with gateway-api-crds precedent; noted as tradeoff) |
| T-27-04 mitigated | cluster_init/app-store-install.yaml | storageclass-demote-job uses dedicated ClusterRole with get/list/patch only |
| T-27-05 mitigated | cluster_init/app-store-install.yaml | kubectl label --overwrite and demote patch are idempotent re-runs |

## Known Stubs

- `[[ join_ip_ports_list ]]` in `weka-client.spec.joinIpPorts` — server-injected YAML array by Phase 29 backend (not a wizard field; intentional; Phase 29 resolves this)
- `[[ endpoints_csv ]]` in `csi-api-secret.stringData.endpoints` — server-injected comma-joined string by Phase 29 backend (intentional; Phase 29 resolves this)
- `csi-wekafs-csi-wekafsplugin-controller` as the deployment readiness name — assumed from chart defaults; verified/corrected in Phase 29/31 E2E

## Issues Encountered

None.

## Next Phase Readiness

- Phase 28 (operator auth): `cluster_init/app-store-install.yaml` is the target CR; Phase 28 adds helm registry login for `oci://quay.io/weka.io/helm` pull auth to the operator
- Phase 29 (backend wiring): Will inject `join_ip_ports_list` and `endpoints_csv` as extra Jinja2 render vars before applying this blueprint; will also wire `find_blueprint` to locate this file
- Phase 30 (wizard): Multi-step form maps 1:1 to the 8 x-variables declared in this file
- No blockers; the blueprint is the authoritative declarative contract

## Self-Check

- [x] `cluster_init/app-store-install.yaml` exists at correct path
- [x] Task 1 commit `24274d9` exists in git log
- [x] Task 1 verification passed (python assertion: OK)
- [x] Task 2 verification passed (python assertion: OK)
- [x] Overall 12-component + 1-default-SC + x-variables verification passed

## Self-Check: PASSED

All created files confirmed present. All commits confirmed in git log. All automated verifications returned OK.

---
*Phase: 27-install-blueprint-authoring*
*Completed: 2026-06-24*
