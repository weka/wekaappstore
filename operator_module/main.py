import logging
import kopf
import kr8s
import subprocess
import yaml
import tempfile
import os
import re
import time
import base64
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from functools import lru_cache

try:
    # Kubernetes client is optional at runtime but recommended
    from kubernetes import client as k8s_client, config as k8s_config
    from kubernetes.client import ApiException as K8sApiException
except Exception:  # pragma: no cover - fallback if lib not present
    k8s_client = None  # type: ignore
    k8s_config = None  # type: ignore
    K8sApiException = Exception  # type: ignore

# Creating a new class to stand up a K8s pods
# Note: This will use the lazy-loaded kr8s API
def get_warrpappstore_class():
    """Get WarrpAppStore class with lazy-loaded kr8s"""
    return kr8s.objects.new_class(kind='WekaAppStore', version='warp.io/v1alpha1', namespaced=True)


@lru_cache(maxsize=1)
def _get_custom_objects_api():
    """Return a configured CustomObjectsApi, or None if the kubernetes client is unavailable.

    Config is loaded once (in-cluster first, then local kubeconfig fallback).
    """
    if k8s_client is None:
        return None
    try:
        k8s_config.load_incluster_config()
    except Exception:
        try:
            k8s_config.load_kube_config()
        except Exception:
            pass
    return k8s_client.CustomObjectsApi()


def _patch_appstack_progress(namespace, name, component_statuses, overall_phase, logger):
    """Best-effort patch of a WekaAppStore .status so deploy progress is observable mid-reconcile.

    Writes via the /status subresource, so it does not bump metadata.generation and does
    not re-trigger the spec-filtered update handler. Failures are logged and ignored — the
    final kopf patch on handler return remains the source of truth.
    """
    api = _get_custom_objects_api()
    if api is None:
        return
    body = {'status': {'appStackPhase': overall_phase, 'componentStatus': component_statuses}}
    try:
        api.patch_namespaced_custom_object_status(
            group='warp.io', version='v1alpha1', namespace=namespace,
            plural='wekaappstores', name=name, body=body,
        )
    except Exception as e:
        logger.warning(f"Incremental status patch failed for {namespace}/{name}: {e}")


class HelmOperator:
    """Handles Helm chart operations for the WARRP operator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Helm command timeout in seconds (configurable via env HELM_CMD_TIMEOUT), default 900s
        try:
            self.helm_cmd_timeout = int(os.getenv("HELM_CMD_TIMEOUT", "900"))
        except Exception:
            self.helm_cmd_timeout = 900
    
    def install_or_upgrade(self, name: str, chart: str, values: Dict[str, Any], 
                          namespace: str, repository: Optional[str] = None,
                          version: Optional[str] = None,
                          skip_crds: Optional[bool] = None) -> tuple[bool, str]:
        """
        Install or upgrade a Helm chart
        
        Args:
            name: Release name
            chart: Chart name (can include repo prefix like 'stable/nginx')
            values: Dictionary of Helm values
            namespace: Target namespace
            repository: Helm repository URL (optional)
            version: Chart version (optional)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Add repository if provided
            if repository:
                repo_name = self._extract_repo_name(repository)
                self._add_repo(repo_name, repository)
            
            # Check if release exists
            if self._release_exists(name, namespace):
                return self._upgrade_chart(name, chart, values, namespace, version, skip_crds)
            else:
                return self._install_chart(name, chart, values, namespace, version, skip_crds)
        except Exception as e:
            error_msg = f"Helm operation failed: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def _add_repo(self, repo_name: str, repo_url: str) -> bool:
        """Add a Helm repository. For OCI registries (oci://...), skip repo add."""
        try:
            if repo_url.startswith("oci://"):
                # Helm doesn't support `helm repo add` for OCI. We'll use the full OCI ref at install time.
                self.logger.info(f"Skipping 'helm repo add' for OCI registry: {repo_url}")
                return True
            cmd = ["helm", "repo", "add", repo_name, repo_url, "--force-update"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                # Update repo after adding
                subprocess.run(["helm", "repo", "update"], capture_output=True, timeout=60)
                self.logger.info(f"Added Helm repository: {repo_name}")
                return True
            else:
                self.logger.warning(f"Failed to add repo {repo_name}: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Error adding Helm repo: {str(e)}")
            return False
    
    def _install_chart(self, name: str, chart: str, values: Dict[str, Any], 
                      namespace: str, version: Optional[str] = None,
                      skip_crds: Optional[bool] = None) -> tuple[bool, str]:
        """Install a new Helm chart"""
        cmd = [
            "helm", "install", name, chart,
            "--namespace", namespace,
            "--create-namespace",
        ]
        
        if version:
            cmd.extend(["--version", version])
        if skip_crds:
            cmd.append("--skip-crds")
        
        # Write values to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(values, f)
            values_file = f.name
        
        try:
            cmd.extend(["--values", values_file])
            self.logger.info(f"Running Helm install: release={name}, chart={chart}, ns={namespace}, version={version or 'latest'}")
            self.logger.debug(f"Helm command: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.helm_cmd_timeout)
            except subprocess.TimeoutExpired:
                msg = f"Helm install timed out after {self.helm_cmd_timeout}s for release '{name}' in ns '{namespace}'. Consider increasing HELM_CMD_TIMEOUT or reducing chart hooks."
                self.logger.error(msg)
                return False, msg
            
            if result.returncode == 0:
                msg = f"Successfully installed Helm release: {name}"
                self.logger.info(msg)
                return True, msg
            else:
                msg = f"Helm install failed (exit {result.returncode}): {result.stderr or result.stdout}"
                self.logger.error(msg)
                return False, msg
        finally:
            # Clean up temporary values file
            if os.path.exists(values_file):
                os.unlink(values_file)
    
    def _upgrade_chart(self, name: str, chart: str, values: Dict[str, Any], 
                      namespace: str, version: Optional[str] = None,
                      skip_crds: Optional[bool] = None) -> tuple[bool, str]:
        """Upgrade an existing Helm chart"""
        cmd = [
            "helm", "upgrade", name, chart,
            "--namespace", namespace,
        ]
        
        if version:
            cmd.extend(["--version", version])
        if skip_crds:
            cmd.append("--skip-crds")
        
        # Write values to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(values, f)
            values_file = f.name
        
        try:
            cmd.extend(["--values", values_file])
            self.logger.info(f"Running Helm upgrade: release={name}, chart={chart}, ns={namespace}, version={version or 'latest'}")
            self.logger.debug(f"Helm command: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.helm_cmd_timeout)
            except subprocess.TimeoutExpired:
                msg = f"Helm upgrade timed out after {self.helm_cmd_timeout}s for release '{name}' in ns '{namespace}'. Consider increasing HELM_CMD_TIMEOUT or reducing chart hooks."
                self.logger.error(msg)
                return False, msg
            
            if result.returncode == 0:
                msg = f"Successfully upgraded Helm release: {name}"
                self.logger.info(msg)
                return True, msg
            else:
                msg = f"Helm upgrade failed (exit {result.returncode}): {result.stderr or result.stdout}"
                self.logger.error(msg)
                return False, msg
        finally:
            # Clean up temporary values file
            if os.path.exists(values_file):
                os.unlink(values_file)
    
    def uninstall(self, name: str, namespace: str) -> tuple[bool, str]:
        """Uninstall a Helm release"""
        try:
            cmd = ["helm", "uninstall", name, "--namespace", namespace]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                msg = f"Successfully uninstalled Helm release: {name}"
                self.logger.info(msg)
                return True, msg
            else:
                msg = f"Helm uninstall failed: {result.stderr}"
                self.logger.error(msg)
                return False, msg
        except Exception as e:
            error_msg = f"Helm uninstall error: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def _release_exists(self, name: str, namespace: str) -> bool:
        """Check if a Helm release exists"""
        try:
            cmd = ["helm", "status", name, "--namespace", namespace]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except Exception:
            return False
    
    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract a repository name from URL"""
        # Simple extraction: use last part of path or domain
        parts = repo_url.rstrip('/').split('/')
        return parts[-1] if parts else "helm-repo"
    
    def get_release_info(self, name: str, namespace: str) -> Optional[Dict[str, Any]]:
        """Get information about a Helm release"""
        try:
            cmd = ["helm", "status", name, "--namespace", namespace, "--output", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                return json.loads(result.stdout)
            return None
        except Exception as e:
            self.logger.error(f"Error getting release info: {str(e)}")
            return None


def merge_values(base_values: Dict[str, Any], additional_values: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple values dictionaries, with later values taking precedence
    """
    result = base_values.copy()
    for values in additional_values:
        result = _deep_merge(result, values)
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def render(text: str, variables: Optional[Dict[str, str]]) -> str:
    """Allowlist ${VAR} substitution: replace only ${name} for names explicitly
    provided in `variables`; leave ALL other $-content untouched.

    Supersedes the original Phase 16 strict-string.Template contract (D-01..D-06,
    OP-02/OP-04). Rationale: kubernetesManifest and Helm-values blobs legitimately
    embed shell syntax — $(cmd), $VAR, ${SHELL_VAR}, ${} inside comments, and $$
    (the shell PID). string.Template.substitute() raised "Invalid placeholder" on
    the first such token, which broke every component whose manifest carries a
    bash script (e.g. aidp-bootstrap-ngc-secrets, keycloak-secret-sync — 46 such
    tokens across the AIDP appstack). Substituting only an explicit allowlist of
    known variable names removes the collision class entirely; the delimiter is
    not the problem, over-broad substitution is.

    Contract:
      - Returns text unchanged if variables is None or {}.
      - Replaces every braced occurrence ${name} -> variables[name], for each
        name in `variables` only. Bare $name is treated as shell content and left.
      - Every other $-sequence ($(, $VAR, ${unknown}, ${}, $$) is preserved
        byte-for-byte. No exception is raised for unknown/foreign placeholders.
        Undefined-variable detection is the caller's responsibility at the
        variable-resolution layer — it is not inferable from manifest text, where
        an "undefined" ${X} is indistinguishable from a legitimate shell ${X}.
    """
    if not variables:
        return text
    # Only the exact, known variable names — braced form — are substitution
    # targets. re.escape guards against regex-special characters in names.
    pattern = re.compile(r"\$\{(" + "|".join(re.escape(k) for k in variables) + r")\}")
    return pattern.sub(lambda m: variables[m.group(1)], text)


def _render_or_raise(
    text: str,
    variables: Optional[Dict[str, str]],
    *,
    source_desc: str,
) -> str:
    """Render text with variables; convert KeyError/ValueError to kopf.PermanentError.

    Wraps Phase 16 render() so each substitution call site can pass a
    caller-specific source_desc (component name, valuesFiles index, kind,
    namespace/name, key) without duplicating try/except boilerplate.

    Per CONTEXT.md D-15 (Phase 18, locked).
    """
    try:
        return render(text, variables)
    except (KeyError, ValueError) as e:
        raise kopf.PermanentError(f"{source_desc}: {e}") from e


# Cluster-scoped kinds must NOT receive a -n namespace flag when applied.
_CLUSTER_SCOPED_KINDS = {
    'Namespace', 'Node', 'PersistentVolume', 'StorageClass',
    'ClusterRole', 'ClusterRoleBinding', 'CustomResourceDefinition',
    'PriorityClass', 'IngressClass', 'RuntimeClass', 'APIService',
    'CSIDriver', 'CSINode', 'VolumeAttachment',
    'ValidatingWebhookConfiguration', 'MutatingWebhookConfiguration',
}


def _split_manifest_docs(manifest_yaml):
    """Parse a multi-doc manifest into [(kind, namespace_or_None, doc_text), ...].

    Uses the YAML parser (not raw '---' splitting) so document separators that
    appear inside block scalars — e.g. bash heredocs embedded in a Job command —
    are correctly ignored. Each doc is re-serialized for per-document kubectl
    invocation. Comments are dropped (irrelevant to kubectl); string values
    (including embedded scripts) round-trip intact.
    """
    docs = []
    for doc in yaml.safe_load_all(manifest_yaml):
        if not isinstance(doc, dict) or not doc:
            continue
        kind = doc.get('kind', '')
        ns = (doc.get('metadata') or {}).get('namespace')
        text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, allow_unicode=True)
        docs.append((kind, ns, text))
    return docs


def _namespace_args(kind, doc_namespace, default_namespace):
    """Return ['-n', <ns>] for namespaced kinds, [] for cluster-scoped kinds.

    A document's own metadata.namespace wins; otherwise the component's target
    namespace is used as the default. This lets one blueprint component span
    namespaces (e.g. App Store credential reads that create RBAC in the
    wekaappstore namespace) without the single-'-n' conflict that kubectl
    rejects ("namespace from the provided object does not match ...").
    """
    if kind in _CLUSTER_SCOPED_KINDS:
        return []
    return ['-n', doc_namespace or default_namespace]


def _apply_manifest_multi_ns(manifest_yaml, default_namespace):
    """kubectl apply a possibly multi-namespace manifest, one document at a time.

    Returns (ok: bool, message: str). Stops at the first failing document.
    """
    try:
        docs = _split_manifest_docs(manifest_yaml)
    except Exception as e:
        return False, f"Failed to parse manifest: {e}"
    for kind, doc_ns, doc_text in docs:
        cmd = ['kubectl', 'apply', '-f', '-'] + _namespace_args(kind, doc_ns, default_namespace)
        result = subprocess.run(cmd, input=doc_text, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return False, result.stderr
    return True, 'Manifest applied successfully'


def _delete_manifest_multi_ns(manifest_yaml, default_namespace, logger):
    """kubectl delete a possibly multi-namespace manifest, best-effort, per document.

    Deletes in reverse document order with --ignore-not-found; failures are logged
    and ignored so one missing/forbidden object never blocks finalizer removal.
    """
    try:
        docs = _split_manifest_docs(manifest_yaml)
    except Exception as e:
        logger.warning(f"Failed to parse manifest for delete: {e}")
        return
    for kind, doc_ns, doc_text in reversed(docs):
        cmd = ['kubectl', 'delete', '-f', '-', '--ignore-not-found'] + _namespace_args(kind, doc_ns, default_namespace)
        result = subprocess.run(cmd, input=doc_text, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning(f"Failed to delete {kind} from manifest: {result.stderr.strip()}")


# ===================== WarpCredential Pure Helpers =====================
# Decision-citing docstrings per Pattern S-3 from 22-PATTERNS.md.
# Helpers are placed in the "pure helper" zone immediately after _render_or_raise
# to mirror the _render_or_raise pattern established in Phase 18.

# Valid WarpCredential spec.type values — used by reconcile_warpcredential (Plan 02)
# for OPS-01 belt-and-suspenders type check (CRD admission already blocks unknown types).
_VALID_WARPCRED_TYPES = {'nvidia-ngc', 'huggingface', 'weka-storage'}


def _b64(s: str) -> str:
    """Standard base64 encode a UTF-8 string, returning ASCII-decoded str.

    Uses standard base64 WITH padding (not URL-safe, not stripped).
    Locked: D-12 — Docker and Kubernetes Secret APIs require standard padded base64.
    Do NOT switch to base64.urlsafe_b64encode or strip trailing '=' characters.
    """
    return base64.b64encode(s.encode('utf-8')).decode('ascii')


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with a trailing 'Z'.

    Match project pattern at main.py:727, 939, 972, 1109 — DO NOT migrate to
    datetime.now(timezone.utc) per RESEARCH.md Pitfall 5.  The project uses
    datetime.utcnow() throughout for consistency; a migration to the non-deprecated
    form is a separate project-wide refactor.
    """
    return datetime.utcnow().isoformat() + 'Z'


def _build_condition(type_: str, status: str, reason: str, message: str) -> dict:
    """Build a Kubernetes-style status condition dict.

    Output shape matches crd.yaml:330-358 for WarpCredential status.conditions[]:
      type (required), status (required, enum True/False/Unknown),
      reason (optional), message (optional), lastTransitionTime (optional ISO 8601).

    Locked: D-14; CRD lines 330-358.
    """
    return {
        'type': type_,
        'status': status,
        'reason': reason,
        'message': message,
        'lastTransitionTime': _now_iso(),
    }


def _derive_ngc_payloads(key: str) -> tuple[dict, dict]:
    """Return (apikey_data, docker_data) for the nvidia-ngc credential type.

    Both dicts contain already-base64-encoded values (D-13) ready to drop into
    the 'data' field of a kr8s Secret object.

    apikey_data  = {NGC_API_KEY, NGC_CLI_API_KEY, NVIDIA_API_KEY, apiKey}  (Opaque secret, OPS-04)
                   All four keys hold the same key value; this mirrors the canonical
                   ngc-api / ngc-api-key secrets that NVIDIA AIDP/NIM charts consume.
    docker_data  = {'.dockerconfigjson': _b64(json.dumps(...))}  (dockerconfigjson secret, OPS-04)

    The dockerconfigjson payload uses the nvcr.io convention:
      username = literal '$oauthtoken'
      auth     = base64('$oauthtoken:<key>') using standard padding (D-12)

    Locked: D-11 (per-type helper signature), D-12 (standard base64 with padding),
    D-13 (return already-encoded data), OPS-04 (NGC apikey + docker secrets).
    NEVER log or reference the `key` parameter value in any log call or docstring.
    """
    key_b64 = _b64(key)
    apikey_data = {
        'NGC_API_KEY': key_b64,
        'NGC_CLI_API_KEY': key_b64,
        'NVIDIA_API_KEY': key_b64,
        'apiKey': key_b64,
    }
    docker_auth_b64 = _b64(f'$oauthtoken:{key}')
    docker_config = {
        'auths': {
            'nvcr.io': {
                'username': '$oauthtoken',
                'password': key,
                'auth': docker_auth_b64,
            }
        }
    }
    docker_data = {'.dockerconfigjson': _b64(json.dumps(docker_config))}
    return apikey_data, docker_data


def _derive_hf_payload(key: str) -> dict:
    """Return the data dict for the huggingface credential type.

    Returns {'HF_API_KEY': _b64(key)} — a single-key Opaque Secret (OPS-05).
    Values are already base64-encoded (D-13); caller builds kr8s Secret directly.

    Locked: D-11 (per-type helper), D-13 (pre-encoded), OPS-05.
    NEVER log or reference the `key` parameter value.
    """
    return {'HF_API_KEY': _b64(key)}


def _derive_weka_payload(username: str, token: str, endpoint: str) -> dict:
    """Return the data dict for the weka-storage credential type.

    Returns exactly three keys, all base64-encoded (D-13):
      WEKA_API_USERNAME, WEKA_API_TOKEN, WEKA_API_ENDPOINT

    The caller has already extracted these values from the source Secret using the
    hard-coded key names resolved in RESEARCH.md §5: the source Secret for
    weka-storage contains all three keys with those literal names.  This helper
    assumes the caller has already selected and validated them.

    Locked: D-11 (per-type helper), D-13 (pre-encoded), OPS-06,
    RESEARCH.md §5 (resolution — source Secret carries all three hard-coded keys).
    NEVER log or reference the `username`, `token`, or `endpoint` parameter values.
    """
    return {
        'WEKA_API_USERNAME': _b64(username),
        'WEKA_API_TOKEN': _b64(token),
        'WEKA_API_ENDPOINT': _b64(endpoint),
    }


def _read_source_secret(name: str, namespace: str, *, ctx: str) -> dict:
    """Read the WarpCredential spec.secretRef Secret and return decoded bytes per key.

    Returns dict[key_name -> bytes] where values are the base64-decoded raw bytes
    from the source Secret's .data field.  Callers select required keys per credential
    type before passing to _derive_* helpers.

    Error dispatch mirrors load_values_from_reference at main.py:449-468 (canonical
    Phase 18 pattern) verbatim:
      kr8s.NotFoundError           -> kopf.TemporaryError(delay=30)  (OPS-02, D-07)
      kr8s.APITimeoutError         -> kopf.TemporaryError(delay=30)  (D-10)
      kr8s.ServerError(>=500)      -> kopf.TemporaryError(delay=30)  (D-10)
      kr8s.ServerError(4xx)        -> kopf.PermanentError            (D-10)

    Locked: D-07 (NotFoundError -> TemporaryError), D-10 (ServerError dispatch),
    OPS-02 (retry on missing secret); mirror of main.py:449-468.
    NEVER include any decoded value in exception messages (T-22-04).
    """
    try:
        secret = kr8s.objects.Secret.get(name=name, namespace=namespace)
    except kr8s.NotFoundError as e:
        raise kopf.TemporaryError(
            f'{ctx}: source Secret {namespace}/{name} not found (will retry in 30s)',
            delay=30,
        ) from e
    except kr8s.APITimeoutError as e:
        raise kopf.TemporaryError(
            f'{ctx}: timeout fetching source Secret {namespace}/{name} (will retry in 30s)',
            delay=30,
        ) from e
    except kr8s.ServerError as e:
        status = e.response.status_code if getattr(e, 'response', None) is not None else None
        if status is None:
            raise kopf.TemporaryError(
                f'{ctx}: unclassified API error fetching Secret {namespace}/{name} (no response; will retry in 30s)',
                delay=30,
            ) from e
        if status >= 500:
            raise kopf.TemporaryError(
                f'{ctx}: API server error {status} fetching Secret {namespace}/{name} (will retry in 30s)',
                delay=30,
            ) from e
        raise kopf.PermanentError(
            f'{ctx}: API error fetching Secret {namespace}/{name}: {e}'
        ) from e

    raw_data = secret.data or {}
    return {k: base64.b64decode(v) for k, v in raw_data.items()}


def _apply_secret_idempotent(secret_obj: kr8s.objects.Secret, *, ctx: str) -> None:
    """Create-or-patch a kr8s Secret.  Locked by D-02 (no delete-and-recreate).

    Idempotent: on 409 Conflict (already exists), issues a merge-patch
    with the new .data and .type values.  A single round-trip in steady
    state (create succeeds immediately).

    Error dispatch:
      409 Conflict (Secret already exists) -> patch({data, type}) and return (D-02)
      kr8s.ServerError(>=500)             -> kopf.TemporaryError(delay=30) (D-10)
      kr8s.ServerError(other 4xx)         -> kopf.PermanentError           (D-10)
      kr8s.APITimeoutError                -> kopf.TemporaryError(delay=30) (D-10)

    IMPORTANT — Pitfall 1 (RESEARCH.md §7.1): kr8s 0.20.10 does not have a
    dedicated AlreadyExists exception class.  The 409 path MUST be detected via
    e.response.status_code == 409 inside the except kr8s.ServerError block.
    Do not import a non-existent AlreadyExists class from kr8s — it will fail.

    The patch call passes exactly {'data': secret_obj.raw['data'], 'type': secret_obj.raw['type']}
    — two keys only (Plan 03 asserts on this exact dict shape).

    Locked: D-02 (create-or-patch idempotency), OPS-09 (derived secret restored on
    next reconcile); mirrors kr8s section in RESEARCH.md §1.
    NEVER log the contents of secret_obj.raw['data'] (T-22-04).
    """
    try:
        secret_obj.create()
    except kr8s.ServerError as e:
        status = e.response.status_code if getattr(e, 'response', None) is not None else None
        if status is None:
            raise kopf.TemporaryError(
                f'{ctx}: unclassified API error writing Secret (no response; will retry in 30s)',
                delay=30,
            ) from e
        if status == 409:
            # Already exists — merge-patch with new .data to converge to desired state.
            # Patch dict is exactly two keys; Plan 03 asserts on this shape.
            secret_obj.patch({'data': secret_obj.raw['data'], 'type': secret_obj.raw['type']})
            return
        if status >= 500:
            raise kopf.TemporaryError(
                f'{ctx}: API server error {status} writing Secret (will retry in 30s)',
                delay=30,
            ) from e
        raise kopf.PermanentError(f'{ctx}: API error writing Secret: {e}') from e
    except kr8s.APITimeoutError as e:
        raise kopf.TemporaryError(
            f'{ctx}: timeout writing Secret (will retry in 30s)',
            delay=30,
        ) from e


# ===================== CRD Strategy Helpers =====================

class HelmError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_kube_config_once() -> bool:
    """Attempt to load Kubernetes client config once.
    Returns True if some config was loaded, False otherwise.
    """
    if not k8s_config:
        return False
    try:
        k8s_config.load_incluster_config()
        return True
    except Exception:
        try:
            k8s_config.load_kube_config()
            return True
        except Exception:
            return False


@lru_cache(maxsize=128)
def discover_chart_crds(chart_ref: str, version: Optional[str] = None) -> set[str]:
    """Return the set of CRD names a chart would install using `helm show crds`.

    If helm cannot show CRDs (no crds/ or error), returns empty set.
    """
    cmd = ["helm", "show", "crds", chart_ref]
    if version:
        cmd += ["--version", str(version)]

    try:
        out = subprocess.check_output(cmd, text=True)
    except subprocess.CalledProcessError:
        return set()
    except Exception:
        return set()

    try:
        crd_docs = list(yaml.safe_load_all(out))
    except Exception:
        crd_docs = []
    names: set[str] = set()
    for doc in crd_docs:
        if not doc or not isinstance(doc, dict):
            continue
        if doc.get("kind") == "CustomResourceDefinition":
            meta = doc.get("metadata", {}) or {}
            name = meta.get("name")
            if name:
                names.add(str(name))
    return names


@lru_cache(maxsize=1)
def list_existing_crds() -> set[str]:
    """List CRD names currently installed in the cluster.
    If kubernetes client/config is unavailable, returns empty set.
    """
    if not k8s_client:
        return set()
    _load_kube_config_once()
    try:
        api = k8s_client.ApiextensionsV1Api()
        crds = api.list_custom_resource_definition().items
        return {crd.metadata.name for crd in crds if getattr(crd, "metadata", None)}
    except K8sApiException:
        return set()
    except Exception:
        return set()


def should_skip_crds_for_component(helm_cfg: Dict[str, Any], chart_ref: str, version: Optional[str]) -> bool:
    """Decide whether to pass --skip-crds for a Helm installation.

    Strategy options (case-insensitive):
      - Install: never skip (always let Helm install CRDs)
      - Skip: always skip (CRDs managed externally)
      - Auto (default): if any CRDs from the chart already exist in cluster, skip
    """
    strategy = (helm_cfg.get("crdsStrategy", "Auto") or "Auto").lower()

    if strategy == "install":
        return False
    if strategy == "skip":
        return True

    # Auto strategy
    chart_crds = discover_chart_crds(chart_ref, version)
    if not chart_crds:
        # No CRDs in chart → nothing to skip at Helm level
        return False

    existing = list_existing_crds()
    return bool(chart_crds & existing)


def load_values_from_reference(
    kind: str,
    name: str,
    key: str,
    namespace: str,
    variables: Optional[Dict[str, str]] = None,
    *,
    comp_name: Optional[str] = None,
    ref_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Load Helm values from a ConfigMap or Secret, optionally rendering ${VAR} substitution.

    New keyword-only params (Phase 18, OP-08/OP-09/OP-11):
      variables  -- if not None, render the raw value string with these vars
                    BEFORE yaml.safe_load (Phase 18 D-15 / OP-08).
      comp_name  -- component name for error context (Phase 18 D-03/D-04).
      ref_index  -- valuesFiles[] index for error context (Phase 18 D-03/D-04).

    Error dispatch (Phase 18 D-01):
      kr8s.NotFoundError, APITimeoutError, ServerError(>=500) -> kopf.TemporaryError(delay=30)
      kr8s.ServerError(4xx)                                   -> kopf.PermanentError
      yaml.YAMLError                                          -> kopf.PermanentError
      render() ValueError (via _render_or_raise)              -> kopf.PermanentError
    """
    # Build a stable resource-locator string for error messages.
    # Threat note (T-18-02): names ONLY metadata, never the rendered/decoded value.
    ctx = f"Component {comp_name!r} valuesFiles[{ref_index}]" if comp_name is not None else f"valuesFiles {kind} {namespace}/{name}"

    try:
        if kind == "ConfigMap":
            cm = kr8s.objects.ConfigMap.get(name=name, namespace=namespace)
            values_yaml = cm.data.get(key, "")
        elif kind == "Secret":
            secret = kr8s.objects.Secret.get(name=name, namespace=namespace)
            import base64
            values_yaml = base64.b64decode(secret.data.get(key, "")).decode('utf-8')
        else:
            raise kopf.PermanentError(f"{ctx}: unsupported valuesFiles kind: {kind}")
    except kr8s.NotFoundError as e:
        raise kopf.TemporaryError(
            f"{ctx}: {kind} {namespace}/{name} not found (will retry in 30s)",
            delay=30,
        ) from e
    except kr8s.APITimeoutError as e:
        raise kopf.TemporaryError(
            f"{ctx}: timeout fetching {kind} {namespace}/{name} (will retry in 30s)",
            delay=30,
        ) from e
    except kr8s.ServerError as e:
        status = e.response.status_code if getattr(e, "response", None) is not None else None
        if status is not None and status >= 500:
            raise kopf.TemporaryError(
                f"{ctx}: API server error {status} fetching {kind} {namespace}/{name} (will retry in 30s)",
                delay=30,
            ) from e
        raise kopf.PermanentError(
            f"{ctx}: API error fetching {kind} {namespace}/{name}: {e}"
        ) from e

    # Render BEFORE yaml.safe_load (Phase 18 OP-08). source_desc names only metadata (T-18-02).
    if variables is not None:
        values_yaml = _render_or_raise(
            values_yaml,
            variables,
            source_desc=f"{ctx} {kind} {namespace}/{name}[{key}]",
        )

    try:
        return yaml.safe_load(values_yaml) or {}
    except yaml.YAMLError as e:
        raise kopf.PermanentError(
            f"{ctx}: malformed YAML in {kind} {namespace}/{name}[{key}]: {e}"
        ) from e


def resolve_dependencies(components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Resolve component dependencies using topological sort.
    Returns components in deployment order.
    Raises ValueError if circular dependencies are detected.
    """
    # Build dependency graph
    component_map = {comp['name']: comp for comp in components}
    in_degree = {comp['name']: 0 for comp in components}
    adjacency = {comp['name']: [] for comp in components}
    
    # Count in-degrees and build adjacency list
    for comp in components:
        depends_on = comp.get('dependsOn', [])
        for dep in depends_on:
            if dep not in component_map:
                raise ValueError(f"Component '{comp['name']}' depends on unknown component '{dep}'")
            adjacency[dep].append(comp['name'])
            in_degree[comp['name']] += 1
    
    # Topological sort using Kahn's algorithm
    queue = [name for name, degree in in_degree.items() if degree == 0]
    sorted_names = []
    
    while queue:
        current = queue.pop(0)
        sorted_names.append(current)
        
        for neighbor in adjacency[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check for circular dependencies
    if len(sorted_names) != len(components):
        raise ValueError("Circular dependency detected in component dependencies")
    
    # Return components in sorted order
    return [component_map[name] for name in sorted_names]


def wait_for_component_ready(component: Dict[str, Any], namespace: str, timeout: int = 300) -> bool:
    """
    Wait for a component to be ready based on readiness check configuration.
    Returns True if ready, False if timeout or error.
    """
    readiness_check = component.get('readinessCheck', {})
    check_type = readiness_check.get('type', 'pod')

    # Build a smarter default selector:
    # Prefer Helm's standard labels and include a secondary name label if possible.
    default_selector = f"app={component['name']}"
    helm_cfg = component.get('helmChart') or {}
    release_name = helm_cfg.get('releaseName', component.get('name'))
    # Some charts set app.kubernetes.io/name to the chart/release name; use component name as a reasonable default
    chart_name = (helm_cfg.get('name') or component.get('chartName') or component.get('name'))
    if release_name:
        # Try a compound selector first (instance + name) which matches many Helm charts
        default_selector = f"app.kubernetes.io/instance={release_name},app.kubernetes.io/name={chart_name}"

    # Determine whether we're targeting by name or selector
    target_name = readiness_check.get('name')
    # Normalize selector: accept string, dict (matchLabels), or list
    raw_selector = readiness_check.get('selector')
    if raw_selector is None:
        # also accept Kubernetes-style matchLabels
        raw_selector = readiness_check.get('matchLabels', default_selector)

    if isinstance(raw_selector, dict):
        selector = ",".join([f"{k}={v}" for k, v in raw_selector.items()])
    elif isinstance(raw_selector, list):
        selector = ",".join([str(item) for item in raw_selector])
    else:
        selector = str(raw_selector)

    check_timeout = int(readiness_check.get('timeout', timeout))
    grace_period = int(readiness_check.get('gracePeriodSeconds', 5))

    # Allow readinessCheck.namespace to override component/CR namespace
    target_namespace = readiness_check.get('namespace') or component.get('targetNamespace', namespace)

    # Map check types to resources and conditions supported by kubectl wait
    type_to_wait = {
        'pod': ('pod', 'ready'),
        'deployment': ('deployment', 'available'),
        'statefulset': ('statefulset', 'ready'),
        'job': ('job', 'complete'),
    }

    # Unknown/custom types: fall back to pods
    resource, condition = type_to_wait.get(check_type, ('pod', 'ready'))

    try:
        logging.info(
            f"Waiting for component '{component['name']}' (type={check_type}, resource={resource}, selector='{selector}') to be ready (timeout: {check_timeout}s) in ns '{target_namespace}'...")

        # Small grace period to let resources appear after Helm returns
        if grace_period > 0:
            time.sleep(grace_period)

        # Build kubectl wait command. If a specific name is provided, prefer resource/name
        if target_name:
            resource_ref = f"{resource}/{target_name}"
            cmd = [
                'kubectl', 'wait', f"--for=condition={condition}", resource_ref,
                '-n', target_namespace,
                f"--timeout={check_timeout}s"
            ]
        else:
            cmd = [
                'kubectl', 'wait', f"--for=condition={condition}", resource,
                '-l', selector,
                '-n', target_namespace,
                f"--timeout={check_timeout}s"
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=check_timeout + 15)

        if result.returncode == 0:
            logging.info(f"Component '{component['name']}' is ready")
            return True

        stderr_full = result.stderr or ''
        stdout_full = result.stdout or ''
        stderr = stderr_full.lower()

        # If nothing matched: handle fallbacks only when we weren't explicitly given a name
        if 'no matching resources' in stderr and not target_name:
            # First try switching resource to deployment with the same selector
            if resource != 'deployment':
                logging.warning(
                    f"No resources found for selector '{selector}' on {resource}. Retrying once as 'deployment'.")
                cmd_deploy = [
                    'kubectl', 'wait', f"--for=condition=available", 'deployment',
                    '-l', selector,
                    '-n', target_namespace,
                    f"--timeout={min(90, check_timeout)}s"
                ]
                result_dep = subprocess.run(cmd_deploy, capture_output=True, text=True, timeout=min(105, check_timeout + 15))
                if result_dep.returncode == 0:
                    logging.info(f"Component '{component['name']}' is ready (deployment)")
                    return True
                # Fall through to label fallback below using pods

            # Then try a conservative fallback selector once on pods
            if selector != f"app={component['name']}":
                fallback_selector = f"app={component['name']}"
                logging.warning(
                    f"No resources found for selector '{selector}'. Retrying once with fallback selector '{fallback_selector}'.")
                cmd_fallback = [
                    'kubectl', 'wait', f"--for=condition={condition}", 'pod',
                    '-l', fallback_selector,
                    '-n', target_namespace,
                    f"--timeout={min(60, check_timeout)}s"
                ]
                result_fb = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=min(75, check_timeout + 15))
                if result_fb.returncode == 0:
                    logging.info(f"Component '{component['name']}' is ready (fallback selector)")
                    return True
                logging.warning(
                    f"Component '{component['name']}' not ready with fallback: {result_fb.stderr or result_fb.stdout}")
                return False

        # Explicitly surface Forbidden errors to hint at RBAC issues
        if 'forbidden' in stderr:
            logging.warning(
                f"RBAC forbidden while waiting for component '{component['name']}': {stderr_full or stdout_full}")
            return False

        logging.warning(
            f"Component '{component['name']}' not ready: {stderr_full or stdout_full}")
        return False

    except Exception as e:
        logging.error(f"Error waiting for component '{component['name']}': {str(e)}")
        return False


def handle_appstack_deployment(body, spec, name, namespace, status, **kwargs):
    """Handle AppStack multi-component deployment with dependencies"""
    logging.info(f"Deploying AppStack {name}")
    
    app_stack = spec['appStack']
    components = app_stack.get('components', [])
    
    if not components:
        raise kopf.PermanentError("appStack.components is required and cannot be empty")
    
    # Filter enabled components
    enabled_components = [comp for comp in components if comp.get('enabled', True)]
    
    if not enabled_components:
        logging.warning(f"No enabled components in AppStack {name}")
        return {
            'appStackPhase': 'Ready',
            'componentStatus': [],
            'conditions': [{
                'type': 'Ready',
                'status': 'True',
                'reason': 'NoComponentsEnabled',
                'message': 'No components are enabled',
                'lastTransitionTime': datetime.utcnow().isoformat() + 'Z'
            }]
        }

    # Build stack-scope variables dict with key/type validation (OP-06, OP-10).
    # Validation runs BEFORE any deployment work — invalid keys/values fail fast
    # without partial deployment. Defense-in-depth alongside Phase 17 CRD admission.
    raw_user_vars = app_stack.get('variables') or {}
    for key in raw_user_vars:
        if not re.fullmatch(r'^[_a-zA-Z][_a-zA-Z0-9]*$', key):
            raise kopf.PermanentError(
                f"Invalid variable key {key!r}: must match Python identifier syntax [_a-zA-Z][_a-zA-Z0-9]*"
            )
        if not isinstance(raw_user_vars[key], str):
            raise kopf.PermanentError(
                f"Invalid variable value for {key!r}: must be a string"
            )
    stack_vars = {'namespace': namespace, **raw_user_vars}

    # Resolve dependencies
    try:
        ordered_components = resolve_dependencies(enabled_components)
    except ValueError as e:
        raise kopf.PermanentError(f"Dependency resolution failed: {str(e)}")
    
    logging.info(f"Deployment order: {[comp['name'] for comp in ordered_components]}")
    
    # Deploy components in order
    helm_operator = HelmOperator()
    component_statuses = []
    failed = False
    
    # Update status to Installing
    if 'patch' in kwargs:
        kwargs['patch'].status['appStackPhase'] = 'Installing'
        kwargs['patch'].status['conditions'] = [{
            'type': 'Ready',
            'status': 'False',
            'reason': 'DeploymentStarted',
            'message': f'Installing {len(ordered_components)} components',
            'lastTransitionTime': datetime.utcnow().isoformat() + 'Z'
        }]
    
    for component in ordered_components:
        comp_name = component['name']
        logging.info(f"Deploying component: {comp_name}")
        
        # Initialize component status
        comp_status = {
            'name': comp_name,
            'phase': 'Installing',
            'message': f'Installing component {comp_name}',
            'lastTransitionTime': datetime.utcnow().isoformat() + 'Z'
        }

        # Publish "Installing" for this component so the GUI reflects real progress.
        _patch_appstack_progress(namespace, name, component_statuses + [comp_status], 'Installing', logging)

        try:
            # Check if it's a Helm chart or Kubernetes manifest
            if 'helmChart' in component and component['helmChart']:
                # Deploy via Helm
                helm_config = component['helmChart']
                chart_repo = helm_config.get('repository')
                chart_name = helm_config.get('name')
                chart_version = helm_config.get('version')
                release_name = helm_config.get('releaseName', comp_name)
                
                if not chart_name:
                    raise ValueError(f"helmChart.name is required for component {comp_name}")
                
                # Normalize/resolve chart reference
                # Heuristics:
                # - If repository is provided and is OCI (oci://), use full OCI ref: oci://.../<chart_name>
                # - If repository is provided (HTTP/S index), use <repo_name>/<chart_name>
                # - If name is a URL to a repo (http/https) and no repository set, assume misconfigured:
                #   treat it as repository URL and use releaseName as chart name.
                # - If name is a direct archive URL (.tgz), pass through as-is.
                chart_ref = None
                # Auto-correct common misconfiguration: name holds a repo URL
                if not chart_repo and isinstance(chart_name, str) and chart_name.startswith("http") and not chart_name.endswith(".tgz"):
                    logging.warning(
                        f"Component '{comp_name}' has helmChart.name set to a repository URL. "
                        f"Auto-correcting by treating it as 'repository' and using releaseName='{release_name}' as chart name. "
                        f"Prefer specifying helmChart.repository and helmChart.name explicitly.")
                    chart_repo = chart_name
                    chart_name = release_name
                if chart_repo and chart_repo.startswith("oci://"):
                    chart_ref = f"{chart_repo.rstrip('/')}/{chart_name}"
                elif chart_repo:
                    repo_name = chart_repo.rstrip('/').split('/')[-1]
                    chart_ref = f"{repo_name}/{chart_name}"
                else:
                    chart_ref = chart_name
                
                # Determine target namespace with flexible resolution order:
                # 1) component.targetNamespace
                # 2) component.namespace (alias)
                # 3) appStack.namespaces[componentName]
                # 4) appStack.defaultNamespace
                # 5) CR namespace
                namespaces_map = (app_stack.get('namespaces') or {}) if isinstance(app_stack, dict) else {}
                default_ns = app_stack.get('defaultNamespace', namespace) if isinstance(app_stack, dict) else namespace
                target_namespace = (
                    component.get('targetNamespace')
                    or component.get('namespace')
                    or namespaces_map.get(comp_name)
                    or default_ns
                    or namespace
                )

                # Merge values (inline + referenced files)
                merged_values = component.get('values', {}).copy()
                if 'valuesFiles' in component:
                    for idx, values_ref in enumerate(component['valuesFiles']):
                        ref_ns = values_ref.get('namespace', target_namespace)
                        ref_values = load_values_from_reference(
                            kind=values_ref['kind'],
                            name=values_ref['name'],
                            key=values_ref['key'],
                            namespace=ref_ns,
                            variables=stack_vars,
                            comp_name=comp_name,
                            ref_index=idx,
                        )
                        merged_values = _deep_merge(merged_values, ref_values)
                
                # Decide CRD handling strategy (default Auto)
                try:
                    # Ensure repo is added before CRD discovery so `helm show crds <repo/chart>` can resolve
                    if chart_repo and not chart_repo.startswith("oci://"):
                        try:
                            repo_name = helm_operator._extract_repo_name(chart_repo)
                            helm_operator._add_repo(repo_name, chart_repo)
                        except Exception as e:
                            logging.debug(f"Skipping repo add prior to CRD discovery for component '{comp_name}': {e}")
                    skip_crds = should_skip_crds_for_component(helm_config, chart_ref, chart_version)
                    logging.info(f"CRD strategy for component '{comp_name}': crdsStrategy={helm_config.get('crdsStrategy', 'Auto')} -> skip_crds={skip_crds}")
                except Exception as e:
                    # Be conservative: don't skip if we couldn't decide
                    skip_crds = False
                    logging.warning(f"Failed to evaluate CRD strategy for component '{comp_name}': {e}. Proceeding without --skip-crds.")

                # Install or upgrade
                success, message = helm_operator.install_or_upgrade(
                    name=release_name,
                    chart=chart_ref,
                    values=merged_values,
                    namespace=target_namespace,
                    repository=chart_repo,
                    version=chart_version,
                    skip_crds=skip_crds
                )
                
                if not success:
                    comp_status['phase'] = 'Failed'
                    comp_status['message'] = f"Failed to deploy: {message}"
                    failed = True
                else:
                    comp_status['releaseName'] = release_name
                    
                    # Wait for component to be ready if configured
                    if component.get('waitForReady', True):
                        if wait_for_component_ready(component, target_namespace):
                            comp_status['phase'] = 'Ready'
                            comp_status['message'] = 'Component deployed and ready'
                        else:
                            comp_status['phase'] = 'Failed'
                            comp_status['message'] = 'Component deployed but not ready within timeout'
                            failed = True
                    else:
                        comp_status['phase'] = 'Ready'
                        comp_status['message'] = 'Component deployed (readiness check skipped)'
                
            elif 'kubernetesManifest' in component and component['kubernetesManifest']:
                # Deploy raw Kubernetes manifest
                manifest_yaml = component['kubernetesManifest']
                target_namespace = component.get('targetNamespace', namespace)
                
                # Check if manifest is empty or contains only whitespace/comments
                manifest_stripped = manifest_yaml.strip()
                if not manifest_stripped or all(line.strip().startswith('#') or not line.strip() 
                                               for line in manifest_stripped.split('\n')):
                    logging.warning(f"Component {comp_name} has empty kubernetesManifest, skipping deployment")
                    comp_status['phase'] = 'Ready'
                    comp_status['message'] = 'Skipped: Empty manifest (placeholder component)'
                else:
                    # Render ${VAR} substitutions before kubectl apply (OP-07).
                    manifest_yaml = _render_or_raise(
                        manifest_yaml,
                        stack_vars,
                        source_desc=f"Component '{comp_name}'.kubernetesManifest",
                    )
                    # Apply per-document so a manifest spanning namespaces (e.g. App
                    # Store credential reads that grant RBAC in the wekaappstore
                    # namespace) is not rejected by a single conflicting -n flag.
                    ok, msg = _apply_manifest_multi_ns(manifest_yaml, target_namespace)
                    if ok:
                        comp_status['phase'] = 'Ready'
                        comp_status['message'] = 'Manifest applied successfully'
                    else:
                        comp_status['phase'] = 'Failed'
                        comp_status['message'] = f"Failed to apply manifest: {msg}"
                        failed = True
            else:
                raise ValueError(f"Component {comp_name} must specify either helmChart or kubernetesManifest")

        except (kopf.PermanentError, kopf.TemporaryError):
            # Phase 18 OP-07/OP-08/OP-11: kopf-typed errors must reach the reconcile
            # boundary so kopf can re-schedule (TemporaryError) or fail loudly
            # (PermanentError). Don't swallow into comp_status['message'] — that
            # hides transient cluster failures and undefined-variable bugs from
            # operators monitoring the CR's status conditions.
            raise
        except Exception as e:
            comp_status['phase'] = 'Failed'
            comp_status['message'] = f"Error deploying component: {str(e)}"
            failed = True
            logging.error(f"Error deploying component {comp_name}: {str(e)}")
        
        component_statuses.append(comp_status)

        # Publish this component's terminal phase (Ready/Failed) for live GUI progress.
        _patch_appstack_progress(namespace, name, component_statuses,
                                 'Failed' if failed else 'Installing', logging)

        # Stop if a component failed
        if failed:
            logging.error(f"Component {comp_name} failed, stopping deployment")
            break
    
    # Determine overall status
    if failed:
        overall_phase = 'Failed'
        condition_status = 'False'
        condition_reason = 'ComponentFailed'
        condition_message = 'One or more components failed to deploy'
    else:
        overall_phase = 'Ready'
        condition_status = 'True'
        condition_reason = 'AllComponentsReady'
        condition_message = f'Successfully deployed {len(component_statuses)} components'
    
    result = {
        'appStackPhase': overall_phase,
        'componentStatus': component_statuses,
        'conditions': [{
            'type': 'Ready',
            'status': condition_status,
            'reason': condition_reason,
            'message': condition_message,
            'lastTransitionTime': datetime.utcnow().isoformat() + 'Z'
        }]
    }
    
    # Explicitly update patch if provided
    if 'patch' in kwargs:
        kwargs['patch'].status.update(result)
        
    return result


# Function to handle the creation of a new WarrpAppStore
@kopf.on.create('warp.io', 'v1alpha1', 'wekaappstores')
def create_warrpappstore_function(body, spec, name, namespace, status, patch, **kwargs):
    logging.info(f"*** WarrpAppStore Created: {name}")
    
    # Check for AppStack deployment (multi-component)
    if 'appStack' in spec and spec['appStack']:
        return handle_appstack_deployment(body, spec, name, namespace, status, patch=patch)
    # Check if Helm chart is specified (single component)
    elif 'helmChart' in spec and spec['helmChart']:
        return handle_helm_deployment(body, spec, name, namespace, status, patch=patch)
    # Fall back to legacy pod-based deployment
    elif 'image' in spec and 'binary' in spec:
        return handle_pod_deployment(body, spec, name, namespace, patch=patch)
    else:
        error_msg = "Either appStack, helmChart, or image+binary must be specified"
        logging.error(error_msg)
        patch.status['conditions'] = [{
            'type': 'Ready',
            'status': 'False',
            'reason': 'InvalidSpec',
            'message': error_msg,
            'lastTransitionTime': datetime.utcnow().isoformat() + 'Z'
        }]
        raise kopf.PermanentError(error_msg)


def handle_helm_deployment(body, spec, name, namespace, status, **kwargs):
    """Handle Helm-based deployment"""
    logging.info(f"Deploying {name} via Helm")
    
    # Update status to Installing
    if 'patch' in kwargs:
        kwargs['patch'].status['releaseStatus'] = 'Installing'
        kwargs['patch'].status['conditions'] = [{
            'type': 'Ready',
            'status': 'False',
            'reason': 'DeploymentStarted',
            'message': f"Installing Helm chart for {name}",
            'lastTransitionTime': datetime.utcnow().isoformat() + 'Z'
        }]
    
    helm_chart_config = spec['helmChart']
    chart_repo = helm_chart_config.get('repository')
    chart_name = helm_chart_config.get('name')
    chart_version = helm_chart_config.get('version')
    release_name = helm_chart_config.get('releaseName', name)
    
    if not chart_name:
        raise kopf.PermanentError("helmChart.name is required")
    
    # Build full chart reference with heuristics
    # - If repository is OCI (oci://), use full ref oci://.../<chart>
    # - If repository is http(s), use <repo_name>/<chart>
    # - If name looks like a repo URL and repository missing, auto-correct by using it as repository and
    #   using releaseName as chart name.
    # - If name is a direct .tgz URL, pass through as-is.
    if not chart_repo and isinstance(chart_name, str) and chart_name.startswith("http") and not chart_name.endswith(".tgz"):
        logging.warning(
            f"helmChart.name appears to be a repository URL. Auto-correcting by treating it as 'repository' "
            f"and using releaseName='{release_name}' as chart name. Prefer specifying 'repository' and 'name' explicitly.")
        chart_repo = chart_name
        chart_name = release_name
    if chart_repo and chart_repo.startswith("oci://"):
        chart_ref = f"{chart_repo.rstrip('/')}/{chart_name}"
    elif chart_repo:
        # Extract repo name and use it as prefix
        repo_name = chart_repo.rstrip('/').split('/')[-1]
        chart_ref = f"{repo_name}/{chart_name}"
    else:
        chart_ref = chart_name
    
    # Start with inline values
    merged_values = spec.get('values', {}).copy()
    
    # Load and merge values from ConfigMaps/Secrets (support per-ref namespace)
    if 'valuesFiles' in spec:
        for values_ref in spec['valuesFiles']:
            ref_ns = values_ref.get('namespace', spec.get('targetNamespace', namespace))
            ref_values = load_values_from_reference(
                kind=values_ref['kind'],
                name=values_ref['name'],
                key=values_ref['key'],
                namespace=ref_ns
            )
            merged_values = _deep_merge(merged_values, ref_values)
    
    # Determine target namespace with alias and defaulting
    target_namespace = (
        spec.get('targetNamespace')
        or spec.get('namespace')
        or namespace
    )
    
    # Install or upgrade Helm chart
    helm_operator = HelmOperator()
    # Determine CRD strategy
    try:
        # Ensure repo is added before CRD discovery so `helm show crds <repo/chart>` can resolve
        if chart_repo and not chart_repo.startswith("oci://"):
            try:
                repo_name = helm_operator._extract_repo_name(chart_repo)
                helm_operator._add_repo(repo_name, chart_repo)
            except Exception as e:
                logging.debug(f"Skipping repo add prior to CRD discovery for release '{release_name}': {e}")
        skip_crds = should_skip_crds_for_component(helm_chart_config, chart_ref, chart_version)
        logging.info(f"CRD strategy for release '{release_name}': crdsStrategy={helm_chart_config.get('crdsStrategy', 'Auto')} -> skip_crds={skip_crds}")
    except Exception as e:
        skip_crds = False
        logging.warning(f"Failed to evaluate CRD strategy for release '{release_name}': {e}. Proceeding without --skip-crds.")
    success, message = helm_operator.install_or_upgrade(
        name=release_name,
        chart=chart_ref,
        values=merged_values,
        namespace=target_namespace,
        repository=chart_repo,
        version=chart_version,
        skip_crds=skip_crds
    )
    
    if not success:
        raise kopf.TemporaryError(message, delay=30)
    
    # Get release info and update status
    release_info = helm_operator.get_release_info(release_name, target_namespace)
    
    result = {
        'releaseName': release_name,
        'releaseStatus': release_info.get('info', {}).get('status') if release_info else 'deployed',
        'releaseVersion': release_info.get('version') if release_info else 1,
        'conditions': [
            {
                'type': 'Ready',
                'status': 'True',
                'reason': 'HelmReleaseDeployed',
                'message': message,
                'lastTransitionTime': datetime.utcnow().isoformat() + 'Z'
            }
        ]
    }
    
    # Explicitly update patch if provided
    if 'patch' in kwargs:
        kwargs['patch'].status.update(result)
        
    return result


def handle_pod_deployment(body, spec, name, namespace, **kwargs):
    """Handle legacy pod-based deployment"""
    logging.info(f"Deploying {name} via Pod (legacy mode)")
    
    # Update status to Installing
    if 'patch' in kwargs:
        kwargs['patch'].status['conditions'] = [{
            'type': 'Ready',
            'status': 'False',
            'reason': 'DeploymentStarted',
            'message': f"Creating pod for {name}",
            'lastTransitionTime': datetime.utcnow().isoformat() + 'Z'
        }]
    
    # Get the WarrpAppStore class with proper kr8s initialization
    Warrpappstore = get_warrpappstore_class()
    warrpappstore = Warrpappstore(body)
    
    # This is a hack to get the live object uid to set on pods so that the operator can delete pods it creates
    live_warrpappstore = Warrpappstore.get(name=name, namespace=namespace)
    
    pod = kr8s.objects.Pod({
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name + "-pod",
        },
        "spec": {
            "containers": [
                {
                    "name": "warrpappstore",
                    "image": spec['image'],
                    "command": [spec['binary'], "https://google.com"],
                }
            ]
        }
    })
    
    pod.create()
    pod.set_owner(live_warrpappstore)
    
    result = {
        'conditions': [
            {
                'type': 'Ready',
                'status': 'True',
                'reason': 'PodCreated',
                'message': f'Pod {name}-pod created successfully',
                'lastTransitionTime': datetime.utcnow().isoformat() + 'Z'
            }
        ]
    }
    
    # Explicitly update patch if provided
    if 'patch' in kwargs:
        kwargs['patch'].status.update(result)
        
    return result


# Handle updates to WarrpAppStore resources
@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores', field='spec')
def update_warrpappstore_function(body, spec, name, namespace, status, patch, **kwargs):
    logging.info(f"*** WarrpAppStore Updated: {name}")
    
    # For AppStack deployments, re-deploy the entire stack
    if 'appStack' in spec and spec['appStack']:
        return handle_appstack_deployment(body, spec, name, namespace, status, patch=patch)
    # For single Helm deployments, trigger upgrade
    elif 'helmChart' in spec and spec['helmChart']:
        return handle_helm_deployment(body, spec, name, namespace, status, patch=patch)
    else:
        logging.warning(f"Update not supported for pod-based deployments: {name}")
        return {}


# Handle deletion of WarrpAppStore resources
@kopf.on.delete('warp.io', 'v1alpha1', 'wekaappstores')
def delete_warrpappstore_function(spec, name, namespace, **kwargs):
    logging.info(f"*** WarrpAppStore Deleted: {name}")
    
    # If AppStack was used, uninstall all components in reverse order
    if 'appStack' in spec and spec['appStack']:
        app_stack = spec['appStack']
        components = app_stack.get('components', [])
        enabled_components = [comp for comp in components if comp.get('enabled', True)]

        # ${VAR} substitution variables, mirroring the deploy path
        # (handle_appstack_deployment). Without this, kubectl delete receives raw
        # ${namespace}/${VAR} tokens and rejects the manifest ("the namespace from the
        # provided object ${namespace} does not match ..."). render() is best-effort and
        # never raises, so deletion is not blocked.
        stack_vars = {'namespace': namespace, **(app_stack.get('variables') or {})}

        # Resolve dependencies to get proper order
        try:
            ordered_components = resolve_dependencies(enabled_components)
            # Reverse order for deletion (remove dependents before dependencies)
            ordered_components.reverse()
        except ValueError as e:
            logging.warning(f"Failed to resolve dependencies for deletion, deleting in original order: {str(e)}")
            ordered_components = enabled_components
        
        helm_operator = HelmOperator()
        
        for component in ordered_components:
            comp_name = component['name']
            
            if 'helmChart' in component and component['helmChart']:
                helm_config = component['helmChart']
                release_name = helm_config.get('releaseName', comp_name)
                target_namespace = component.get('targetNamespace', namespace)
                
                logging.info(f"Uninstalling component: {comp_name} (release: {release_name})")
                success, message = helm_operator.uninstall(release_name, target_namespace)
                
                if success:
                    logging.info(f"Component {comp_name} uninstalled successfully")
                else:
                    logging.warning(f"Failed to uninstall component {comp_name}: {message}")
            elif 'kubernetesManifest' in component and component['kubernetesManifest']:
                # For raw manifests, attempt to delete using kubectl.
                # Render ${VAR} substitutions first so the manifest namespace matches (OP-07).
                manifest_yaml = render(component['kubernetesManifest'], stack_vars)
                target_namespace = component.get('targetNamespace', namespace)

                # Delete per-document so cross-namespace manifests are honored
                # (each object goes to its own namespace, not a single -n).
                _delete_manifest_multi_ns(manifest_yaml, target_namespace, logging)
                logging.info(f"Component {comp_name} manifest delete attempted")
    
    # If single Helm chart was used, uninstall the release
    elif 'helmChart' in spec and spec['helmChart']:
        helm_chart_config = spec['helmChart']
        release_name = helm_chart_config.get('releaseName', name)
        target_namespace = spec.get('targetNamespace', namespace)
        
        helm_operator = HelmOperator()
        success, message = helm_operator.uninstall(release_name, target_namespace)
        
        if success:
            logging.info(f"Helm release {release_name} uninstalled successfully")
        else:
            logging.warning(f"Failed to uninstall Helm release {release_name}: {message}")
    else:
        logging.info(f"Pod-based deployment cleanup handled by owner reference")


# ===================== WarpCredential Handlers =====================

# D-05, OPS-08, RESEARCH.md §Pattern 3 — optional=True prevents kopf from adding
# a finalizer to the CR.  Without a finalizer the handler is best-effort (kopf may
# miss the event if Kubernetes removes the resource before kopf processes DELETED).
# Acceptable: this handler does no destructive work; logging a warning is best-effort
# by design (nolar/kopf#701).
@kopf.on.delete('warp.io', 'v1alpha1', 'warpcredentials', optional=True)
def delete_warpcredential(name, namespace, logger, **_):
    """OPS-08: log a warning only; do NOT delete derived secrets.

    optional=True prevents kopf from adding a finalizer (kopf maintainer comment
    in nolar/kopf#701: with optional=True, no finalizer is added; the handler is
    best-effort logging).  This is intentional — derived secrets must outlive the
    WarpCredential CR (OPS-08).  The inaction (preserving derived secrets) is the
    contract; cluster state is the same whether the warning was logged or not.

    Locked: D-05 (warning-only delete, no destructive work), OPS-08 (derived secrets
    must survive CR deletion), RESEARCH.md §Pattern 3 (optional=True rationale).
    """
    logger.warning(
        f'WarpCredential {namespace}/{name} deleted; derived secrets '
        f'warp-{name}-* are intentionally retained (OPS-08). '
        f'Administrator must delete them manually if no longer needed.'
    )


# D-04, OPS-01..OPS-09 — stacked decorators so create/update/resume all converge to the
# same reconcile function.  field='spec' on @kopf.on.update is REQUIRED (Pitfall 3, D-04):
# it prevents the operator's own status writes from re-triggering the handler, avoiding an
# infinite reconcile loop (mirrors WekaAppStore convention at main.py:1159).
@kopf.on.create('warp.io', 'v1alpha1', 'warpcredentials')
@kopf.on.update('warp.io', 'v1alpha1', 'warpcredentials', field='spec')
@kopf.on.resume('warp.io', 'v1alpha1', 'warpcredentials')
def reconcile_warpcredential(body, spec, name, namespace, patch, logger, **kwargs):
    """Reconcile a WarpCredential CR — derive type-appropriate Kubernetes Secrets and update status.

    Stacked decorators (D-04): create / update(field='spec') / resume all call this function.
    @kopf.on.resume fires when the operator pod restarts, re-checking every existing CR and
    restoring any deleted derived secrets (OPS-09 idempotency after restart).

    Error classification (D-07/D-08/D-09/D-10):
      - Unknown spec.type -> PermanentError, reason='UnknownType' (OPS-01, D-08)
      - Missing secretRef.name or .key -> PermanentError, reason='InvalidSpec' (belt-and-suspenders)
      - Source Secret not found -> TemporaryError(delay=30), reason='KeyMissing' (OPS-02, D-07)
      - Key absent from source Secret -> PermanentError, reason='KeyMissing' (OPS-02)
      - Empty/whitespace key value -> PermanentError, reason='EmptyKey' (OPS-03, D-09)
      - API/network error reading source Secret -> TemporaryError or PermanentError (D-10)

    Derivation helpers (D-11/D-12/D-13):
      - nvidia-ngc  -> _derive_ngc_payloads(key) -> apikey + dockerconfigjson Secrets (OPS-04)
      - huggingface -> _derive_hf_payload(key) -> token Secret (OPS-05)
      - weka-storage -> _derive_weka_payload(username, token, endpoint) -> token Secret (OPS-06)

    Status writes (D-14/D-15, S-4):
      - Failure paths patch status.conditions BEFORE raising (every single branch).
      - Success path writes conditions, derivedSecrets, lastSyncTime; weka-storage also wekaEndpoint.

    Security (API-08/D-03):
      - The decoded key/token/username/endpoint values are NEVER passed to any logger call.
      - All logger and exception messages use ctx (CR name+namespace+displayName), source Secret
        name, key NAME — never key VALUES.

    Requirements covered: OPS-01, OPS-02, OPS-03, OPS-04, OPS-05, OPS-06, OPS-07, OPS-09, API-08.
    """
    cred_type = spec.get('type')
    display_name = spec.get('displayName', name)
    ctx = f'WarpCredential {namespace}/{name}({display_name})'

    # OPS-01 (D-08) — unknown type belt-and-suspenders (CRD admission should block this first)
    if cred_type not in _VALID_WARPCRED_TYPES:
        patch.status['conditions'] = [_build_condition(
            'KeyReady', 'False', 'UnknownType',
            f'spec.type {cred_type!r} not recognized')]
        raise kopf.PermanentError(f'{ctx}: unknown spec.type {cred_type!r}')

    # Belt-and-suspenders secretRef validation (CRD required=[name,key] gates this at admission)
    secret_ref = spec.get('secretRef', {})
    src_name = secret_ref.get('name')
    src_key = secret_ref.get('key')
    if not src_name or not src_key:
        patch.status['conditions'] = [_build_condition(
            'KeyReady', 'False', 'InvalidSpec',
            'spec.secretRef.name and .key required')]
        raise kopf.PermanentError(f'{ctx}: spec.secretRef.name and .key required')

    # OPS-02 (D-07) — read source Secret; wrap to patch status BEFORE the kopf error escapes
    try:
        src_data = _read_source_secret(src_name, namespace, ctx=ctx)
    except kopf.TemporaryError:
        patch.status['conditions'] = [_build_condition(
            'KeyReady', 'False', 'KeyMissing',
            f'Source Secret {namespace}/{src_name} not found (retrying)')]
        raise
    except kopf.PermanentError:
        patch.status['conditions'] = [_build_condition(
            'KeyReady', 'False', 'KeyReadError',
            f'API error fetching Secret {namespace}/{src_name}')]
        raise

    # Type-specific key extraction and derivation
    derived_secrets_list = []

    if cred_type in ('nvidia-ngc', 'huggingface'):
        # For single-key types: verify the named key exists and is non-empty
        if src_key not in src_data:
            patch.status['conditions'] = [_build_condition(
                'KeyReady', 'False', 'KeyMissing',
                f'Source Secret {namespace}/{src_name} has no key {src_key!r}')]
            raise kopf.PermanentError(
                f'{ctx}: source Secret {src_name!r} missing key {src_key!r}')

        key = src_data[src_key].decode('utf-8')

        # OPS-03 (D-09) — empty/whitespace key -> PermanentError; key NAME in message, not value
        if not key.strip():
            patch.status['conditions'] = [_build_condition(
                'KeyReady', 'False', 'EmptyKey',
                f'Source Secret {namespace}/{src_name}[{src_key}] is empty')]
            raise kopf.PermanentError(
                f'{ctx}: source Secret key {src_key!r} is empty')

        if cred_type == 'nvidia-ngc':
            # OPS-04 — two derived Secrets: warp-{name}-apikey (Opaque) + warp-{name}-docker
            apikey_data, docker_data = _derive_ngc_payloads(key)
            apikey_secret = kr8s.objects.Secret({
                'apiVersion': 'v1',
                'kind': 'Secret',
                'metadata': {'name': f'warp-{name}-apikey', 'namespace': namespace},
                'type': 'Opaque',
                'data': apikey_data,
            })
            docker_secret = kr8s.objects.Secret({
                'apiVersion': 'v1',
                'kind': 'Secret',
                'metadata': {'name': f'warp-{name}-docker', 'namespace': namespace},
                'type': 'kubernetes.io/dockerconfigjson',
                'data': docker_data,
            })
            try:
                _apply_secret_idempotent(apikey_secret, ctx=f'{ctx}: apikey')
                _apply_secret_idempotent(docker_secret, ctx=f'{ctx}: docker')
            except (kopf.TemporaryError, kopf.PermanentError):
                patch.status['conditions'] = [_build_condition(
                    'KeyReady', 'False', 'SecretWriteError',
                    'Failed to write derived Secret(s) to the API server (see operator logs)')]
                raise
            derived_secrets_list = [
                {'name': f'warp-{name}-apikey', 'type': 'Opaque'},
                {'name': f'warp-{name}-docker', 'type': 'kubernetes.io/dockerconfigjson'},
            ]

        else:  # huggingface
            # OPS-05 — single derived Secret: warp-{name}-token (Opaque, HF_API_KEY)
            hf_data = _derive_hf_payload(key)
            hf_secret = kr8s.objects.Secret({
                'apiVersion': 'v1',
                'kind': 'Secret',
                'metadata': {'name': f'warp-{name}-token', 'namespace': namespace},
                'type': 'Opaque',
                'data': hf_data,
            })
            try:
                _apply_secret_idempotent(hf_secret, ctx=f'{ctx}: token')
            except (kopf.TemporaryError, kopf.PermanentError):
                patch.status['conditions'] = [_build_condition(
                    'KeyReady', 'False', 'SecretWriteError',
                    'Failed to write derived Secret(s) to the API server (see operator logs)')]
                raise
            derived_secrets_list = [{'name': f'warp-{name}-token', 'type': 'Opaque'}]

    else:  # weka-storage
        # OPS-06, RESEARCH §5 — source Secret contains three hard-coded keys.
        # spec.secretRef.key is read for schema symmetry only (set to WEKA_API_TOKEN by convention);
        # the operator reads all three keys directly by their literal names.
        username_bytes = src_data.get('WEKA_API_USERNAME')
        token_bytes = src_data.get('WEKA_API_TOKEN')
        endpoint_bytes = src_data.get('WEKA_API_ENDPOINT')

        # Validate each required key individually (names in messages, not values — T-22-04)
        for key_name, key_bytes in [
            ('WEKA_API_USERNAME', username_bytes),
            ('WEKA_API_TOKEN', token_bytes),
            ('WEKA_API_ENDPOINT', endpoint_bytes),
        ]:
            if key_bytes is None:
                patch.status['conditions'] = [_build_condition(
                    'KeyReady', 'False', 'KeyMissing',
                    f'weka-storage source Secret {namespace}/{src_name} missing key {key_name}')]
                raise kopf.PermanentError(
                    f'{ctx}: weka-storage source Secret {src_name!r} missing key {key_name!r}')
            if not key_bytes.decode('utf-8').strip():
                patch.status['conditions'] = [_build_condition(
                    'KeyReady', 'False', 'EmptyKey',
                    f'weka-storage source Secret {namespace}/{src_name} key {key_name} is empty')]
                raise kopf.PermanentError(
                    f'{ctx}: weka-storage source Secret key {key_name!r} is empty')

        username = username_bytes.decode('utf-8')
        token = token_bytes.decode('utf-8')
        endpoint_from_src = endpoint_bytes.decode('utf-8')
        # spec.endpoint takes precedence; fall back to source Secret value (defense-in-depth)
        endpoint = spec.get('endpoint') or endpoint_from_src

        weka_data = _derive_weka_payload(username, token, endpoint)
        weka_secret = kr8s.objects.Secret({
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {'name': f'warp-{name}-token', 'namespace': namespace},
            'type': 'Opaque',
            'data': weka_data,
        })
        try:
            _apply_secret_idempotent(weka_secret, ctx=f'{ctx}: token')
        except (kopf.TemporaryError, kopf.PermanentError):
            patch.status['conditions'] = [_build_condition(
                'KeyReady', 'False', 'SecretWriteError',
                'Failed to write derived Secret(s) to the API server (see operator logs)')]
            raise
        derived_secrets_list = [{'name': f'warp-{name}-token', 'type': 'Opaque'}]
        patch.status['wekaEndpoint'] = endpoint  # resolved value (spec.endpoint or source Secret)

    # OPS-07 (D-14) — success status write; all failure paths above already patched status
    conditions = [_build_condition('KeyReady', 'True', 'KeyPresent', 'Derived secrets reconciled')]
    if cred_type == 'nvidia-ngc':
        conditions.append(_build_condition(
            'DockerSecretReady', 'True', 'DockerSecretPresent', 'NGC docker Secret reconciled'))

    patch.status['conditions'] = conditions
    patch.status['derivedSecrets'] = derived_secrets_list
    patch.status['lastSyncTime'] = _now_iso()

    # Log metadata only — key values, token values, username, endpoint are NEVER included (API-08, D-03)
    logger.info(f'{ctx}: reconciled {len(derived_secrets_list)} derived Secret(s)')
