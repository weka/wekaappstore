# Phase 27: Install Blueprint Authoring - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the analysis.

**Date:** 2026-06-24
**Phase:** 27-install-blueprint-authoring
**Mode:** assumptions
**Areas analyzed:** Component Ordering & Dependency Graph, StorageClass Default Handling, x-variables Block Design, Node Label Job Service Account, CSI Driver Helm Chart Reference

## Assumptions Presented

### Component Ordering & Dependency Graph

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Topo-sort: quay-secrets → weka-operator (readiness: deployment) → (node-label-sa/weka-client-secret/csi-wekafs) → (weka-client/csi-api-secret) → storageclass-demote-job → storageclasses | Confident | `operator_module/main.py` resolve_dependencies; ROADMAP SC1; `wekaClientCR-online.yaml` wekaSecretRef dependency |

### StorageClass Default Handling

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Pre-apply `storageclass-demote-job` Job (SA+CRB+Job pattern) checks and demotes existing default StorageClass before three-StorageClass manifest component | Likely | No `is-default-class` logic in operator_module/main.py; `gateway-api-crds-job` is canonical precedent; ROADMAP SC4 requires brownfield safe demote/skip |

### x-variables Block Design

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| 7 user-facing vars + `quay_dockerconfigjson` (validate: false); server-derived vars excluded | Likely | `parse_x_variables` reads x-variables dict; ROADMAP SC5 "GUI-derived vars excluded from validation"; WekaClient joinIpPorts is YAML-array, cannot be raw user string |

### Node Label Job Service Account

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Three-component SA+ClusterRoleBinding+Job sequence with cluster-admin binding; no reusable SA has node-patch | Confident | `weka-csi-config/rbac.yaml` confirms only get/list/watch on nodes; `app-store-cluster-init.yaml` lines 130-226 is canonical precedent |

### CSI Driver Helm Chart Reference

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| `https://weka.github.io/csi-wekafs`, chart `csi-wekafsplugin`, release `csi-wekafs`; OCI operator tag uses `v`-prefix | Confident (post-research) | `weka.github.io/csi-wekafs/index.yaml` live with 52 versions; `github.com/weka/weka-operator/releases` confirms v1.13.0 latest stable |

## Corrections Made

No corrections — all assumptions confirmed.

## External Research

- **CSI WekaFS chart URL:** `https://weka.github.io/csi-wekafs`, chart name `csi-wekafsplugin`, stable version `2.8.7` (Source: weka.github.io/csi-wekafs/index.yaml)
- **WEKA Operator OCI version tag format:** `v`-prefix confirmed; `v1.13.0` is current latest stable (Source: github.com/weka/weka-operator/releases)
