from fastapi import FastAPI, Request, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict, Any, List, Tuple
import os
import re
import yaml
import base64
import json
import time
import asyncio
import copy
import subprocess
import shutil
import platform
import logging
import ssl
import datetime
import urllib.request
import urllib.error
import urllib.parse
import ipaddress

from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException
from jinja2 import Environment
from webapp.inspection import collect_cluster_inspection, collect_weka_inspection, flatten_cluster_status
from webapp.inspection.cluster import _cpu_to_millicores, _memory_to_bytes, _safe_int
from webapp.planning import ApplyGateway

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WEKA App Store")

class ClusterInitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Paths to exclude from blocking
        exempt_paths = ["/healthz", "/readyz", "/static", "/welcome", "/init-cluster", "/init-logs", "/cluster-status", "/cluster-info","/deploy-stream"]
        
        # Allow exempt paths
        if any(request.url.path.startswith(p) for p in exempt_paths):
            return await call_next(request)
        
        # Check initialization status
        # We check in all namespaces to find any initialization CR
        initialized = await is_cluster_initialized_anywhere()
        
        # If it's the root path, we'll check and redirect if needed
        if request.url.path == "/":
            if not initialized:
                return RedirectResponse(url="/welcome")
            return await call_next(request)

        # For other paths, return 503 if not initialized
        if not initialized:
            return JSONResponse(
                status_code=503,
                content={"detail": "Cluster initialization required", "init_required": True}
            )
        
        return await call_next(request)

async def is_cluster_initialized_anywhere():
    """Check if the cluster is initialized by looking for the CR in any namespace."""
    try:
        load_kube_config()
        custom_api = client.CustomObjectsApi()
        # list_cluster_custom_object is for cluster-scoped resources, 
        # but wekaappstores are namespaced. So we use list_namespaced_custom_object across all namespaces if possible,
        # or just list_cluster_custom_object if it works for namespaced too? 
        # Actually, list_cluster_custom_object works for namespaced resources if you want to see all of them.
        crs = custom_api.list_cluster_custom_object(
            group="warp.io",
            version="v1alpha1",
            plural="wekaappstores"
        )
        for cr in crs.get("items", []):
            if cr.get("metadata", {}).get("name") == "app-store-cluster-init":
                status = cr.get("status", {})
                phase = status.get("appStackPhase")
                if phase == "Ready":
                    return True
        return False
    except Exception as e:
        logger.error(f"Error checking cluster initialization across all namespaces: {e}")
        # Fallback to checking default namespace as a safety measure
        return await is_cluster_initialized(namespace="default")

async def is_cluster_initialized(namespace: str = "default"):
    try:
        load_kube_config()
        custom_api = client.CustomObjectsApi()
        cr = custom_api.get_namespaced_custom_object(
            group="warp.io",
            version="v1alpha1",
            namespace=namespace,
            plural="wekaappstores",
            name="app-store-cluster-init"
        )
        status = cr.get("status", {})
        phase = status.get("appStackPhase")
        
        # Consider the cluster initialized if the phase is 'Ready'
        # We also log this for troubleshooting purposes
        logger.info(f"Checking cluster initialization status: phase={phase}")
        
        if phase == "Ready":
            return True
            
        # In some cases, if the operator is still processing but core components are up,
        # we might want to allow access. However, for robustness, we stick to 'Ready'.
        return False
    except ApiException as e:
        if e.status == 404:
            logger.info("Cluster initialization CR 'app-store-cluster-init' not found. Welcome screen will be shown.")
            return False
        logger.error(f"Kubernetes API error checking initialization: {e}")
        # On other API errors, we default to not-initialized to be safe
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking cluster initialization: {e}")
        return False

app.add_middleware(ClusterInitMiddleware)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
# Resolve BLUEPRINTS_DIR robustly with sane fallbacks
# Priority:
# 1) Explicit env BLUEPRINTS_DIR
# 2) /app/manifests (project image default, subPath mounted to the repo root)
# 3) git-sync root+link (GIT_SYNC_ROOT/GIT_SYNC_LINK)
# 4) Project-relative manifests for local dev
_default_git_sync_root = os.getenv("GIT_SYNC_ROOT", "/tmp/git-sync-root")
_default_git_sync_link = os.getenv("GIT_SYNC_LINK", "../../manifests")
_default_blueprints_dir = os.path.join(_default_git_sync_root, _default_git_sync_link)

_candidates = []
_env_bp = os.getenv("BLUEPRINTS_DIR")
if _env_bp:
    _c = _env_bp.rstrip("/")
    _candidates.append(_c)
# git-sync derived path
## Prefer the image default mount first to avoid symlink/permission quirks
# image default path
_candidates.append("/app/manifests")
# git-sync derived path (may be a symlink)
_candidates.append(_default_blueprints_dir)
# project-relative manifests (for local dev)
_candidates.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../manifests"))

BLUEPRINTS_DIR = None
for _p in _candidates:
    try:
        if _p and os.path.isdir(_p):
            BLUEPRINTS_DIR = _p
            break
    except Exception:
        pass
if not BLUEPRINTS_DIR:
    # Fall back to first candidate even if it doesn't exist to keep behavior predictable
    BLUEPRINTS_DIR = _candidates[0] if _candidates else _default_blueprints_dir

# If the selected directory itself contains a top-level 'manifests' directory
# (common when the git repo root has a manifests/ folder), descend into it so
# callers can consistently reference BLUEPRINTS_DIR/<blueprint>/...
try:
    _nested = os.path.join(BLUEPRINTS_DIR, "manifests")
    if os.path.isdir(_nested):
        BLUEPRINTS_DIR = _nested
except Exception:
    pass

# Mount static if present (not strictly required since we use Tailwind CDN)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)
PLANNING_APPLY_GATEWAY = ApplyGateway(project_root=PROJECT_ROOT)

# Apps whose blueprints carry fixed per-component targetNamespace values that must
# NOT be overwritten by a user-selected namespace.  Any new blueprint that uses
# hard-coded targetNamespace values should be added here.
NAMESPACE_PRESERVING_APPS = {"cluster-init", "app-store-install"}

# Load logo as base64 once for reuse in templates
LOGO_B64 = None
try:
    logo_path = os.path.join(BASE_DIR, "app_store_logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as _lf:
            LOGO_B64 = base64.b64encode(_lf.read()).decode('ascii')
except Exception:
    LOGO_B64 = None

# Load Glocomp partner logo (used on home-page chip + blueprint detail page)
GLOCOMP_LOGO_B64 = None
try:
    glocomp_logo_path = os.path.join(TEMPLATES_DIR, "glocomp_logo.png")
    if os.path.exists(glocomp_logo_path):
        with open(glocomp_logo_path, 'rb') as _gf:
            GLOCOMP_LOGO_B64 = base64.b64encode(_gf.read()).decode('ascii')
except Exception:
    GLOCOMP_LOGO_B64 = None

# Load TokenVisor partner logo + architecture diagram
TOKENVISOR_LOGO_B64 = None
try:
    tv_logo_path = os.path.join(TEMPLATES_DIR, "tokenvisor_logo.png")
    if os.path.exists(tv_logo_path):
        with open(tv_logo_path, 'rb') as _tf:
            TOKENVISOR_LOGO_B64 = base64.b64encode(_tf.read()).decode('ascii')
except Exception:
    TOKENVISOR_LOGO_B64 = None

TOKENVISOR_ARCH_B64 = None
try:
    tv_arch_path = os.path.join(TEMPLATES_DIR, "tokenvisor_architecture.png")
    if os.path.exists(tv_arch_path):
        with open(tv_arch_path, 'rb') as _tf:
            TOKENVISOR_ARCH_B64 = base64.b64encode(_tf.read()).decode('ascii')
except Exception:
    TOKENVISOR_ARCH_B64 = None

# Helper: load kube config
# _config_loaded is intentionally a process-level singleton: once the kube
# config is loaded for this process there is no need (or safe way) to reload
# it dynamically.  Tests that need a fresh config should restart the process
# or mock load_kube_config() directly.
_config_loaded = False


def load_kube_config() -> None:
    """Load Kubernetes client configuration.

    Mode selection:
    - KUBERNETES_AUTH_MODE=incluster → force in-cluster ServiceAccount auth
    - KUBERNETES_AUTH_MODE=kubeconfig → force local kubeconfig
    - KUBERNETES_AUTH_MODE=auto or unset → try in-cluster, then fall back to kubeconfig
    """
    global _config_loaded
    if _config_loaded:
        return

    mode = os.getenv("KUBERNETES_AUTH_MODE", "auto").lower().strip()

    def try_incluster() -> None:
        config.load_incluster_config()

    def try_kubeconfig() -> None:
        # Respect KUBECONFIG env var if provided; kubernetes client handles it natively
        config.load_kube_config()

    try:
        if mode == "incluster":
            try_incluster()
        elif mode == "kubeconfig":
            try_kubeconfig()
        else:
            # auto
            try:
                try_incluster()
            except Exception as ic_err:
                # Fall back to kubeconfig
                try_kubeconfig()
        _config_loaded = True
        return
    except Exception as e:
        # Construct a helpful error message with guidance
        msg = (
            "Unable to load Kubernetes configuration. "
            f"Mode={mode}. "
            "If running inside a cluster, ensure the Pod has a ServiceAccount "
            "with the necessary RBAC and that /var/run/secrets/kubernetes.io/serviceaccount exists. "
            "If running locally, ensure a valid kubeconfig exists and KUBECONFIG is set if needed. "
            f"Underlying error: {e}"
        )
        raise RuntimeError(msg)


def get_cluster_status() -> Dict[str, Any]:
    """Collect bounded cluster status through the shared inspection service."""
    snapshot = collect_cluster_inspection(load_kube_config=load_kube_config)
    return flatten_cluster_status(snapshot)


def ensure_namespace_exists(namespace: Optional[str]) -> None:
    """Ensure the given namespace exists; create it if missing.
    Safe to call repeatedly. No-op if namespace is falsy.
    """
    if not namespace:
        return
    load_kube_config()
    core = client.CoreV1Api()
    try:
        core.read_namespace(name=namespace)
        return
    except ApiException as ae:
        if ae.status != 404:
            # Unexpected error
            raise
    # Create the namespace
    body = client.V1Namespace(metadata=client.V1ObjectMeta(name=namespace))
    try:
        core.create_namespace(body=body)
    except ApiException as ae:
        if ae.status == 409:
            # Already exists (race condition)
            return
        raise


def _with_last_applied_annotation(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of doc with kubectl.kubernetes.io/last-applied-configuration set.

    The annotation value is a compact JSON serialization of the manifest without the
    annotation itself (to avoid self-referential growth). This helps prevent
    kubectl apply warnings later by mimicking kubectl's stored configuration behavior.
    """
    try:
        # Work on a deep copy to avoid mutating caller prior to annotation value generation
        doc_copy = copy.deepcopy(doc)
        md = doc_copy.setdefault("metadata", {}) if isinstance(doc_copy, dict) else {}
        anns = md.setdefault("annotations", {}) if isinstance(md, dict) else {}
        # Remove any existing last-applied to avoid nesting
        anns.pop("kubectl.kubernetes.io/last-applied-configuration", None)
        # Serialize the copy
        serialized = json.dumps(doc_copy, separators=(",", ":"), sort_keys=True)
        # Now ensure annotation also exists in the real document and assign
        md_real = doc.setdefault("metadata", {}) if isinstance(doc, dict) else {}
        anns_real = md_real.setdefault("annotations", {}) if isinstance(md_real, dict) else {}
        anns_real["kubectl.kubernetes.io/last-applied-configuration"] = serialized
    except Exception:
        # Best-effort: if anything goes wrong, just return original doc
        return doc
    return doc


def apply_yaml(file_path: str, namespace: Optional[str] = None) -> Dict[str, Any]:
    """Apply a YAML manifest using the Kubernetes Python client."""
    load_kube_config()

    if not os.path.isabs(file_path):
        # Resolve relative to project root
        file_path = os.path.join(PROJECT_ROOT, file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    k8s_client = client.ApiClient()
    created = utils.create_from_yaml(k8s_client, file_path, namespace=namespace, verbose=False)

    # Normalize the return to a list of (obj, resp) tuples
    def _to_tuples(created_any):
        tuples = []
        if created_any is None:
            return []
        if isinstance(created_any, list):
            for item in created_any:
                if isinstance(item, tuple) and len(item) == 2:
                    tuples.append(item)
                else:
                    tuples.append((item, None))
        elif isinstance(created_any, tuple) and len(created_any) == 2:
            tuples = [created_any]
        else:
            tuples = [(created_any, None)]
        return tuples

    tuples = _to_tuples(created)

    applied_kinds = []
    for obj, _ in tuples:
        try:
            applied_kinds.append(obj.kind)
        except Exception:
            applied_kinds.append(str(obj))

    return {"applied": applied_kinds}


from functools import lru_cache

# A pragmatic list for built-in cluster-scoped kinds you’re likely to have in init bundles.
CLUSTER_SCOPED_KINDS = {
    "Namespace",
    "Node",
    "PersistentVolume",
    "CustomResourceDefinition",
    "ClusterRole",
    "ClusterRoleBinding",
    "StorageClass",
    "MutatingWebhookConfiguration",
    "ValidatingWebhookConfiguration",
    "APIService",
    "PriorityClass",
    "PodSecurityPolicy",  # legacy clusters
    "RuntimeClass",
    "CSIDriver",
    "CSINode",
    "CertificateSigningRequest",
}

def _kind(doc: dict) -> str:
    return str((doc or {}).get("kind") or "")

def _api_version(doc: dict) -> str:
    return str((doc or {}).get("apiVersion") or "")

def _metadata(doc: dict) -> dict:
    md = (doc or {}).get("metadata")
    return md if isinstance(md, dict) else {}

def is_builtin_cluster_scoped(doc: dict) -> bool:
    return _kind(doc) in CLUSTER_SCOPED_KINDS

@lru_cache(maxsize=256)
def crd_scope_for(group: str, plural: str) -> str:
    """
    Returns 'Namespaced' or 'Cluster' based on the CRD spec.scope.
    """
    try:
        load_kube_config()
        ext = client.ApiextensionsV1Api()
        crd_name = f"{plural}.{group}"
        crd = ext.read_custom_resource_definition(crd_name)
        return str(crd.spec.scope or "Namespaced")
    except Exception:
        # Fallback to Namespaced if CRD not found or other error
        return "Namespaced"

def is_cluster_scoped(doc: dict) -> bool:
    """
    Check if a resource is cluster-scoped.
    Built-in kinds are checked against CLUSTER_SCOPED_KINDS.
    For CRDs, we check the scope dynamically via Apiextensions API.
    """
    if _kind(doc) in CLUSTER_SCOPED_KINDS:
        return True
    
    api_version = _api_version(doc)
    if "/" in api_version:
        group = api_version.split("/", 1)[0]
        kind = _kind(doc)
        # Naive pluralization for CRD lookup
        plural = kind.lower() + "s"
        if lower_kind := kind.lower():
             if lower_kind == "warrpappstore":
                 plural = "warrpappstores"
             elif lower_kind == "wekaappstore":
                 plural = "wekaappstores"
        
        if crd_scope_for(group, plural) == "Cluster":
            return True
            
    return False

def _doc_id(doc: dict) -> str:
    md = (doc or {}).get("metadata") or {}
    return f"{doc.get('apiVersion')} {doc.get('kind')} {md.get('namespace','<cluster>')}/{md.get('name')}"

def apply_blueprint_with_namespace(file_path: str, namespace: str) -> Dict[str, Any]:
    return PLANNING_APPLY_GATEWAY.apply_file(file_path, namespace)


def apply_blueprint_content_with_namespace(content: str, namespace: str) -> Dict[str, Any]:
    return PLANNING_APPLY_GATEWAY.apply_content(content, namespace)


def apply_blueprint_documents_with_namespace(documents, namespace: str) -> Dict[str, Any]:
    return PLANNING_APPLY_GATEWAY.apply_documents(documents, namespace)


def install_warrp_crd() -> Dict[str, Any]:
    crd_path = os.path.join(PROJECT_ROOT, "warrp-crd.yaml")
    return apply_yaml(crd_path, namespace=None)


# ---------------------------------------------------------------------------
# Blueprint resource requirements (declared-first, inferred-fallback)
# ---------------------------------------------------------------------------

_GPU_RESOURCE_KEYS = ("nvidia.com/gpu", "amd.com/gpu", "gpu.intel.com/i915")
_WORKLOAD_KINDS = {
    "Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob",
    "ReplicaSet", "ReplicationController", "Pod",
}


def _empty_requirements() -> Dict[str, Any]:
    return {
        "cpu_cores": None,
        "memory_gib": None,
        "gpu_devices": None,
        "gpu_required": False,
        "gpu_count_known": False,
        "gpu_model": None,
        "source": "none",
    }


def _requirements_from_declaration(x_req: Any) -> Optional[Dict[str, Any]]:
    """Map an x-requirements block to the normalized requirements dict, or None."""
    if not isinstance(x_req, dict):
        return None
    req = _empty_requirements()
    req["source"] = "declared"

    cpu = x_req.get("cpu")
    if isinstance(cpu, dict) and cpu.get("cores") is not None:
        try:
            req["cpu_cores"] = float(cpu["cores"])
        except (TypeError, ValueError):
            pass

    mem = x_req.get("memory")
    if isinstance(mem, dict) and mem.get("gib") is not None:
        try:
            req["memory_gib"] = float(mem["gib"])
        except (TypeError, ValueError):
            pass

    gpu = x_req.get("gpu")
    if isinstance(gpu, dict):
        req["gpu_required"] = True
        if gpu.get("model"):
            req["gpu_model"] = str(gpu["model"])
        if gpu.get("count") is not None:
            try:
                req["gpu_devices"] = int(gpu["count"])
                req["gpu_count_known"] = True
                if req["gpu_devices"] <= 0:
                    req["gpu_required"] = False
            except (TypeError, ValueError):
                pass
    return req


def _pod_specs_in_doc(doc: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    """Return (pod_specs, replicas) for a k8s workload doc.

    Handles the nesting differences between Deployment/StatefulSet/Job/CronJob/Pod.
    replicas defaults to 1 (DaemonSet/CronJob are treated as 1 — node fan-out is unknown).
    """
    spec = doc.get("spec") or {}
    kind = doc.get("kind")
    if kind == "Pod":
        return ([spec], 1)
    if kind == "CronJob":
        job_spec = ((spec.get("jobTemplate") or {}).get("spec")) or {}
        pod_spec = ((job_spec.get("template") or {}).get("spec")) or {}
        return ([pod_spec] if pod_spec else [], 1)
    # Deployment / StatefulSet / DaemonSet / Job / ReplicaSet / ReplicationController
    pod_spec = ((spec.get("template") or {}).get("spec")) or {}
    replicas = spec.get("replicas")
    try:
        replicas = int(replicas) if replicas is not None else 1
    except (TypeError, ValueError):
        replicas = 1
    if kind in ("DaemonSet", "Job", "CronJob"):
        replicas = 1
    return ([pod_spec] if pod_spec else [], replicas)


def _accumulate_workload(doc: Dict[str, Any], acc: Dict[str, Any]) -> None:
    """Sum container resource requests/limits from a workload doc into acc."""
    pod_specs, replicas = _pod_specs_in_doc(doc)
    for pod_spec in pod_specs:
        if not isinstance(pod_spec, dict):
            continue
        if str(pod_spec.get("runtimeClassName") or "").lower() == "nvidia":
            acc["gpu_required"] = True
        containers = (pod_spec.get("containers") or []) + (pod_spec.get("initContainers") or [])
        for c in containers:
            if not isinstance(c, dict):
                continue
            res = c.get("resources") or {}
            requests = res.get("requests") or {}
            limits = res.get("limits") or {}
            # CPU/memory: prefer requests, fall back to limits.
            cpu_q = requests.get("cpu", limits.get("cpu"))
            mem_q = requests.get("memory", limits.get("memory"))
            if cpu_q is not None:
                acc["cpu_milli"] += _cpu_to_millicores(cpu_q) * replicas
            if mem_q is not None:
                acc["mem_bytes"] += _memory_to_bytes(mem_q) * replicas
            # GPU: only ever expressed as a limit.
            for key in _GPU_RESOURCE_KEYS:
                if key in limits or key in requests:
                    count = _safe_int(limits.get(key, requests.get(key)))
                    if count > 0:
                        acc["gpu_devices"] += count * replicas
                        acc["gpu_required"] = True
                        acc["gpu_count_known"] = True


def _scavenge_gpu(obj: Any, acc: Dict[str, Any]) -> None:
    """Best-effort recursive scan for GPU signals in arbitrary structures (e.g. helm values).

    Sets gpu_required when a GPU resource key appears; sums counts when they are numeric.
    Does not attempt CPU/memory summation here (no reliable replica/container context).
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _GPU_RESOURCE_KEYS:
                acc["gpu_required"] = True
                count = _safe_int(v)
                if count > 0:
                    acc["gpu_devices"] += count
                    acc["gpu_count_known"] = True
            else:
                _scavenge_gpu(v, acc)
    elif isinstance(obj, list):
        for item in obj:
            _scavenge_gpu(item, acc)


def _parse_embedded_docs(text: Any) -> List[Dict[str, Any]]:
    """Parse a YAML string (inline kubernetesManifest / valuesContent) into doc dicts."""
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        return [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]
    except yaml.YAMLError:
        return []


# Last-resort markers that a blueprint needs GPUs even when no countable
# nvidia.com/gpu resource is visible (e.g. GPU workloads packaged in external
# Helm charts referenced via valuesFiles). Low-false-positive, NVIDIA/NIM-centric.
_GPU_NAME_MARKERS = ("nvidia", "nim", "triton", "tensorrt", "gpu-operator", "-gpu", "cuda")
_GPU_TEXT_MARKERS = ("weka.io/nim-role", "nvidia.com/gpu", "runtimeclassname: nvidia", "gpu.nvidia.com")


def _heuristic_gpu_required(docs: List[Dict[str, Any]], raw: str) -> bool:
    """Conservative detection of GPU need from names/charts/labels when no
    countable GPU resource was found."""
    low = (raw or "").lower()
    if any(marker in low for marker in _GPU_TEXT_MARKERS):
        return True
    for top in docs:
        if not isinstance(top, dict):
            continue
        components = (((top.get("spec") or {}).get("appStack")) or {}).get("components") or []
        for comp in components:
            if not isinstance(comp, dict):
                continue
            name = str(comp.get("name") or "").lower()
            helm = comp.get("helmChart") or {}
            chart = " ".join(str(helm.get(k) or "") for k in ("name", "chart", "repository")).lower()
            if any(mk in name or mk in chart for mk in _GPU_NAME_MARKERS):
                return True
    return False


def _infer_requirements(docs: List[Dict[str, Any]], raw: str = "") -> Dict[str, Any]:
    """Deep-infer resource requirements by walking appStack components, embedded
    manifests, and helm valuesContent. Returns a normalized requirements dict."""
    acc = {"cpu_milli": 0, "mem_bytes": 0, "gpu_devices": 0,
           "gpu_required": False, "gpu_count_known": False}

    def handle_workload_docs(workload_docs: List[Dict[str, Any]]) -> None:
        for d in workload_docs:
            if d.get("kind") in _WORKLOAD_KINDS:
                _accumulate_workload(d, acc)

    for top in docs:
        if not isinstance(top, dict):
            continue
        # Top-level doc may itself be a workload (legacy/standalone manifests).
        if top.get("kind") in _WORKLOAD_KINDS:
            _accumulate_workload(top, acc)
        app_stack = ((top.get("spec") or {}).get("appStack")) or {}
        components = app_stack.get("components") or []
        for comp in components:
            if not isinstance(comp, dict):
                continue
            handle_workload_docs(_parse_embedded_docs(comp.get("kubernetesManifest")))
            helm = comp.get("helmChart") or {}
            if isinstance(helm, dict):
                for vc_doc in _parse_embedded_docs(helm.get("valuesContent")):
                    _scavenge_gpu(vc_doc, acc)
        # Single helmChart mode at spec level.
        helm = (top.get("spec") or {}).get("helmChart") or {}
        if isinstance(helm, dict):
            for vc_doc in _parse_embedded_docs(helm.get("valuesContent")):
                _scavenge_gpu(vc_doc, acc)

    req = _empty_requirements()
    req["source"] = "inferred"
    req["cpu_cores"] = round(acc["cpu_milli"] / 1000.0, 2) if acc["cpu_milli"] else None
    req["memory_gib"] = round(acc["mem_bytes"] / float(1024 ** 3), 2) if acc["mem_bytes"] else None
    req["gpu_required"] = acc["gpu_required"]
    req["gpu_count_known"] = acc["gpu_count_known"]
    req["gpu_devices"] = acc["gpu_devices"] if acc["gpu_count_known"] else None
    # Last resort: flag GPU need from names/charts/labels (count stays unknown).
    if not req["gpu_required"] and _heuristic_gpu_required(docs, raw):
        req["gpu_required"] = True
        req["gpu_count_known"] = False
    return req


def compute_blueprint_requirements(yaml_path: Optional[str]) -> Dict[str, Any]:
    """Determine a blueprint's resource needs.

    Strategy: an explicit `x-requirements` block is authoritative; otherwise infer
    from container resources across all docs, embedded manifests, and helm values.
    When GPU is clearly required but the count can't be determined, gpu_count_known
    is False so the UI can show an honest "count unknown".
    """
    if not yaml_path:
        return _empty_requirements()
    path = yaml_path if os.path.isabs(yaml_path) else os.path.join(PROJECT_ROOT, yaml_path)
    try:
        with open(path, "r") as f:
            raw = f.read()
    except Exception:
        return _empty_requirements()

    try:
        docs = [d for d in yaml.safe_load_all(raw) if isinstance(d, dict)]
    except yaml.YAMLError:
        docs = []

    # 1) Declared block wins.
    for d in docs:
        declared = _requirements_from_declaration(d.get("x-requirements"))
        if declared:
            return declared

    # 2) Inference fallback.
    return _infer_requirements(docs, raw)


def _compute_requirement_meets(reqs: Dict[str, Any], status: Dict[str, Any]) -> Dict[str, Any]:
    """Compare requirements against the cluster's FREE capacity.

    Each value is True (fits), False (insufficient), or None (unknown — either the
    requirement or the cluster figure is unavailable).
    """
    def fits(need, free):
        if need is None or free is None:
            return None
        try:
            return float(free) >= float(need)
        except (TypeError, ValueError):
            return None

    cpu = fits(reqs.get("cpu_cores"), status.get("cpu_cores_free"))
    memory = fits(reqs.get("memory_gib"), status.get("memory_gib_free"))
    if not reqs.get("gpu_required"):
        gpu = True  # blueprint needs no GPU
    elif not reqs.get("gpu_count_known"):
        gpu = None  # GPU needed but count unknown
    else:
        gpu = fits(reqs.get("gpu_devices"), status.get("gpu_devices_free"))
    return {"cpu": cpu, "memory": memory, "gpu": gpu}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    status = await asyncio.to_thread(get_cluster_status)
    auth = await asyncio.to_thread(get_auth_status)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "status": status,
            "auth": auth,
            "logo_b64": LOGO_B64,
            "glocomp_logo_b64": GLOCOMP_LOGO_B64,
            "tokenvisor_logo_b64": TOKENVISOR_LOGO_B64,
        },
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    auth = await asyncio.to_thread(get_auth_status)
    status = await asyncio.to_thread(get_cluster_status)
    # Use detected namespace if available, else default
    detected_ns = (auth.get("details", {}) or {}).get("namespace") if isinstance(auth, dict) else None
    ns = detected_ns or "default"

    credentials_by_type = await _get_credentials_by_type(ns)
    weka_storage_credentials = [c for c in credentials_by_type["weka-storage"] if c.get("ready")]

    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "request": request,
            "auth": auth,
            "status": status,
            "detected_namespace": detected_ns or "default",
            "logo_b64": LOGO_B64,
            "credentials_by_type": credentials_by_type,
            "weka_storage_credentials": weka_storage_credentials,
        },
    )


@app.get("/status")
async def status_endpoint():
    return JSONResponse(get_cluster_status())


def create_or_update_secret(name: str, namespace: str, string_data: Dict[str, str]) -> Dict[str, Any]:
    """Create or update an Opaque secret with given string_data.

    Returns a dict with name/namespace and operation performed.
    """
    load_kube_config()
    ensure_namespace_exists(namespace)
    core = client.CoreV1Api()
    metadata = client.V1ObjectMeta(name=name, namespace=namespace)
    secret_body = client.V1Secret(metadata=metadata, type="Opaque", string_data=string_data)
    try:
        # Try create
        core.create_namespaced_secret(namespace=namespace, body=secret_body)
        return {"name": name, "namespace": namespace, "action": "created"}
    except ApiException as ae:
        if ae.status == 409:
            # Exists – patch to update data without needing resourceVersion
            patched = client.V1Secret(metadata=metadata, type="Opaque", string_data=string_data)
            core.patch_namespaced_secret(name=name, namespace=namespace, body=patched)
            return {"name": name, "namespace": namespace, "action": "updated"}
        raise


@app.get("/api/blueprints")
async def list_blueprints(namespace: str = Query("all", description="Namespace to list WekaAppStore CRs from; use 'all' for cluster-wide")):
    """List Weka App Store custom resources (CRs).

    CRD: group=warp.io, version=v1alpha1, plural=wekaappstores
    Returns safe metadata only: name, namespace, creationTimestamp.
    """
    try:
        load_kube_config()
        co_api = client.CustomObjectsApi()
        items = []
        if (namespace or "").strip().lower() in ("all", "*"):
            resp = co_api.list_cluster_custom_object(group="warp.io", version="v1alpha1", plural="wekaappstores")
        else:
            resp = co_api.list_namespaced_custom_object(group="warp.io", version="v1alpha1", plural="wekaappstores", namespace=namespace.strip())
        for it in (resp or {}).get("items", []) or []:
            md = (it or {}).get("metadata", {}) or {}
            items.append({
                "name": md.get("name"),
                "namespace": md.get("namespace") or "default",
                "creationTimestamp": md.get("creationTimestamp"),
            })
        return JSONResponse({"ok": True, "items": items})
    except ApiException as ae:
        # 404 if CRD not installed or no permission
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.delete("/api/blueprints/{namespace}/{name}")
async def delete_blueprint(namespace: str, name: str):
    """Delete a Weka App Store CR instance by namespace and name."""
    try:
        load_kube_config()
        co_api = client.CustomObjectsApi()
        # Foreground deletion to ensure child resources are cleaned up by owner refs if any
        body = client.V1DeleteOptions(propagation_policy="Foreground")
        co_api.delete_namespaced_custom_object(
            group="warp.io",
            version="v1alpha1",
            namespace=namespace,
            plural="wekaappstores",
            name=name,
            body=body,
        )
        return JSONResponse({"ok": True})
    except ApiException as ae:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/secrets")
async def list_secrets(namespace: str = Query("all", description="Namespace to list secrets from; use 'all' for all namespaces")):
    """List Kubernetes Secrets and return safe metadata only.

    Returns items with: name, namespace, type, creationTimestamp.
    """
    try:
        load_kube_config()
        core = client.CoreV1Api()
        items = []
        if (namespace or "").strip().lower() in ("all", "*"):
            sec_list = core.list_secret_for_all_namespaces(_preload_content=True)
        else:
            ns = namespace.strip()
            sec_list = core.list_namespaced_secret(namespace=ns, _preload_content=True)
        for s in (sec_list.items or []):
            md = getattr(s, 'metadata', None)
            items.append({
                "name": md.name if md else None,
                "namespace": md.namespace if md else None,
                "type": getattr(s, 'type', None),
                "creationTimestamp": (md.creation_timestamp.isoformat() if (md and md.creation_timestamp) else None),
            })
        return JSONResponse({"ok": True, "items": items})
    except ApiException as ae:
        # Permission errors or others
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Credentials API — WarpCredential CRD (group=warp.io, version=v1alpha1)
# ---------------------------------------------------------------------------

_CREDENTIAL_TYPE_KEYS: Dict[str, Dict[str, Any]] = {
    "nvidia-ngc": {
        "secret_ref_key": "NGC_API_KEY",
        "secret_data_keys": ["NGC_API_KEY"],
    },
    "huggingface": {
        "secret_ref_key": "HF_API_KEY",
        "secret_data_keys": ["HF_API_KEY"],
    },
    "weka-storage": {
        "secret_ref_key": "WEKA_API_TOKEN",
        "secret_data_keys": ["WEKA_API_USERNAME", "WEKA_API_TOKEN", "WEKA_API_ENDPOINT"],
    },
}

_VALID_CREDENTIAL_TYPES: tuple = tuple(_CREDENTIAL_TYPE_KEYS.keys())


def _make_credential_slug(display_name: str) -> str:
    """Convert a human-readable displayName into a DNS-1123-compatible slug.

    Rules (D-11): lowercase, replace non-alphanumeric runs with '-',
    strip leading/trailing hyphens, truncate to 48 characters.
    Truncating to 48 (not 52) leaves room for the -99 suffix added by
    _allocate_unique_credential_slug while staying within 51 characters total.
    Raises ValueError if the result is empty.
    """
    slug = display_name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    slug = slug[:48]
    if not slug:
        raise ValueError("displayName produced an empty slug")
    return slug


def _allocate_unique_credential_slug(co_api: Any, namespace: str, base_slug: str) -> str:
    """Return a slug that does not collide with any existing WarpCredential in the namespace.

    Appends -2, -3, ... up to -99 until a unique name is found (D-12).
    Raises RuntimeError if exhausted.
    """
    resp = co_api.list_namespaced_custom_object(
        group="warp.io",
        version="v1alpha1",
        plural="warpcredentials",
        namespace=namespace,
    )
    existing_names = {
        (it.get("metadata") or {}).get("name")
        for it in (resp or {}).get("items", []) or []
    }
    if base_slug not in existing_names:
        return base_slug
    for suffix in range(2, 100):
        candidate = f"{base_slug}-{suffix}"
        if candidate not in existing_names:
            return candidate
    raise RuntimeError("could not allocate unique credential slug after 99 attempts")


def _build_credential_response_item(cr: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a WarpCredential CR dict into the API response item shape.

    Never includes raw credential values — only safe metadata and status fields.
    """
    md = cr.get("metadata") or {}
    spec = cr.get("spec") or {}
    status = cr.get("status") or {}
    conditions = status.get("conditions") or []

    cred_type = spec.get("type", "")

    # Derive ready flag from KeyReady condition
    key_ready_condition = next((c for c in conditions if c.get("type") == "KeyReady"), None)
    ready = (key_ready_condition.get("status") == "True") if key_ready_condition else False

    item: Dict[str, Any] = {
        "name": md.get("name"),
        "namespace": md.get("namespace") or "default",
        "type": cred_type,
        "displayName": spec.get("displayName"),
        "ready": ready,
        "lastSyncTime": status.get("lastSyncTime"),
        "derivedSecrets": status.get("derivedSecrets") or [],
    }

    # Add error field when not ready and a message is available
    if not ready and key_ready_condition and key_ready_condition.get("message"):
        item["error"] = key_ready_condition["message"]

    # nvidia-ngc: add dockerSecretReady
    if cred_type == "nvidia-ngc":
        docker_condition = next((c for c in conditions if c.get("type") == "DockerSecretReady"), None)
        item["dockerSecretReady"] = (docker_condition.get("status") == "True") if docker_condition else False

    # weka-storage: add endpoint from status (never from raw Secret)
    if cred_type == "weka-storage":
        item["endpoint"] = status.get("wekaEndpoint")

    return item


async def _get_credentials_by_type(ns: str) -> dict:
    """Return ready credentials grouped by type for a namespace.

    Returns {"nvidia-ngc": [...], "huggingface": [...], "weka-storage": [...]}.
    Falls back to empty dict-of-lists on ApiException | ConnectionError | TimeoutError.
    Only credentials with ready=True are included (single source of truth for SDK-02's
    ready contract — macros can use simple truthiness checks with no selectattr needed).
    """
    credentials_by_type: dict = {"nvidia-ngc": [], "huggingface": [], "weka-storage": []}

    def _list():
        return client.CustomObjectsApi().list_namespaced_custom_object(
            group="warp.io", version="v1alpha1",
            plural="warpcredentials", namespace=ns,
        )
    try:
        load_kube_config()
        resp = await asyncio.to_thread(_list)
        for cr in (resp or {}).get("items", []) or []:
            item = _build_credential_response_item(cr)
            t = item.get("type")
            if t in credentials_by_type and item.get("ready"):
                credentials_by_type[t].append(item)
    except (ApiException, ConnectionError, TimeoutError):
        pass
    return credentials_by_type


@app.get("/api/credentials")
async def list_credentials(
    namespace: str = Query("default", description="Namespace to list WarpCredential CRs from; use 'all' for cluster-wide"),
    cred_type: Optional[str] = Query(None, alias="type", description="Filter by credential type; if set, only items of this type with ready=true are returned"),
):
    """List WarpCredential CRs with safe status shape.

    Returns safe metadata and status only — never raw credential values.
    CRD: group=warp.io, version=v1alpha1, plural=warpcredentials
    """
    try:
        load_kube_config()

        def _list() -> Dict[str, Any]:
            co_api = client.CustomObjectsApi()
            ns = (namespace or "").strip()
            if ns.lower() in ("all", "*"):
                return co_api.list_cluster_custom_object(
                    group="warp.io",
                    version="v1alpha1",
                    plural="warpcredentials",
                )
            return co_api.list_namespaced_custom_object(
                group="warp.io",
                version="v1alpha1",
                plural="warpcredentials",
                namespace=ns,
            )

        resp = await asyncio.to_thread(_list)
        items = [
            _build_credential_response_item(cr)
            for cr in (resp or {}).get("items", []) or []
        ]

        # Apply type filter: only ready items of the requested type (API-02)
        if cred_type is not None:
            items = [it for it in items if it["type"] == cred_type and it["ready"] is True]

        return JSONResponse({"ok": True, "items": items})
    except ApiException as ae:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/credentials")
async def create_credential(
    display_name: str = Form(...),
    cred_type: str = Form(..., alias="type"),
    namespace: str = Form("default"),
    key: str = Form(...),
    username: Optional[str] = Form(None),
    endpoint: Optional[str] = Form(None),
):
    """Create a new WarpCredential CR and its raw warp-cred-<slug> Secret.

    Handles slug collision by appending -2, -3, ... (D-12).
    Never echoes raw credential values in the response (API-08).
    """
    try:
        # --- Validation ---
        if not display_name.strip():
            return JSONResponse({"ok": False, "error": "displayName is required"}, status_code=400)

        if cred_type not in _VALID_CREDENTIAL_TYPES:
            return JSONResponse(
                {"ok": False, "error": f"invalid credential type: {cred_type}; must be one of {list(_VALID_CREDENTIAL_TYPES)}"},
                status_code=400,
            )

        if cred_type == "weka-storage":
            if not username or not username.strip():
                return JSONResponse({"ok": False, "error": "username is required for weka-storage credentials"}, status_code=400)
            if not endpoint or not endpoint.strip():
                return JSONResponse({"ok": False, "error": "endpoint is required for weka-storage credentials"}, status_code=400)

        load_kube_config()
        ns = namespace.strip() or "default"

        try:
            base_slug = _make_credential_slug(display_name)
        except ValueError as ve:
            return JSONResponse({"ok": False, "error": str(ve)}, status_code=400)

        co_api = client.CustomObjectsApi()
        slug = await asyncio.to_thread(_allocate_unique_credential_slug, co_api, ns, base_slug)

        # Build string_data for the raw Secret per D-08/D-09/D-10
        if cred_type == "nvidia-ngc":
            string_data = {"NGC_API_KEY": key}
        elif cred_type == "huggingface":
            string_data = {"HF_API_KEY": key}
        else:  # weka-storage
            string_data = {
                "WEKA_API_USERNAME": username.strip(),
                "WEKA_API_TOKEN": key,
                "WEKA_API_ENDPOINT": endpoint.strip(),
            }

        # Create the raw Secret (warp-cred-<slug>)
        core = client.CoreV1Api()
        await asyncio.to_thread(create_or_update_secret, f"warp-cred-{slug}", ns, string_data)

        # Build the WarpCredential CR body
        body: Dict[str, Any] = {
            "apiVersion": "warp.io/v1alpha1",
            "kind": "WarpCredential",
            "metadata": {"name": slug, "namespace": ns},
            "spec": {
                "type": cred_type,
                "displayName": display_name.strip(),
                "secretRef": {
                    "name": f"warp-cred-{slug}",
                    "key": _CREDENTIAL_TYPE_KEYS[cred_type]["secret_ref_key"],
                },
            },
        }
        if cred_type == "weka-storage":
            body["spec"]["endpoint"] = endpoint.strip()

        # Create the WarpCredential CR
        try:
            await asyncio.to_thread(
                co_api.create_namespaced_custom_object,
                group="warp.io",
                version="v1alpha1",
                namespace=ns,
                plural="warpcredentials",
                body=body,
            )
        except ApiException as ae:
            if ae.status == 409:
                # Roll back the Secret we just created
                try:
                    await asyncio.to_thread(
                        core.delete_namespaced_secret,
                        name=f"warp-cred-{slug}",
                        namespace=ns,
                    )
                except Exception:
                    pass
                return JSONResponse({"ok": False, "error": f"slug {slug} already taken; retry"}, status_code=409)
            raise

        logger.info("Created WarpCredential: name=%s namespace=%s type=%s", slug, ns, cred_type)

        # Return metadata only — never echo key, username, or endpoint values (API-08)
        return JSONResponse({
            "ok": True,
            "name": slug,
            "namespace": ns,
            "type": cred_type,
            "displayName": display_name.strip(),
        })
    except ApiException as ae:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


_CREDENTIAL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,251}[a-z0-9])?$")


@app.delete("/api/credentials/{name}")
async def delete_credential(
    name: str,
    namespace: str = Query("default", description="Namespace of the WarpCredential CR"),
):
    """Delete a WarpCredential CR and its raw warp-cred-<name> Secret.

    Deletes the CR first (stops operator reconcile loop), then the raw Secret.
    Derived secrets (warp-<name>-*) are never touched (API-04, OPS-08).
    """
    try:
        # Validate first — no I/O on invalid input (matches get_weka_overview pattern)
        if not _CREDENTIAL_NAME_RE.match(name):
            return JSONResponse({"ok": False, "error": "invalid credential name"}, status_code=400)

        load_kube_config()
        ns = namespace.strip() or "default"
        co_api = client.CustomObjectsApi()
        core = client.CoreV1Api()

        # Step 1: Delete the WarpCredential CR first (stops operator reconcile loop)
        delete_opts = client.V1DeleteOptions(propagation_policy="Foreground")
        try:
            await asyncio.to_thread(
                co_api.delete_namespaced_custom_object,
                group="warp.io",
                version="v1alpha1",
                namespace=ns,
                plural="warpcredentials",
                name=name,
                body=delete_opts,
            )
        except ApiException as ae:
            if ae.status != 404:
                raise
            # 404 means CR already gone — still attempt raw Secret cleanup

        # Step 2: Delete the raw warp-cred-<name> Secret
        # Derived warp-<name>-* secrets are intentionally NOT deleted here (API-04, OPS-08)
        try:
            await asyncio.to_thread(
                core.delete_namespaced_secret,
                name=f"warp-cred-{name}",
                namespace=ns,
            )
        except ApiException as ae:
            if ae.status != 404:
                raise
            # 404 means Secret already gone — idempotent delete

        logger.info("Deleted WarpCredential: name=%s namespace=%s", name, ns)
        return JSONResponse({"ok": True})
    except ApiException as ae:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/weka/overview")
async def get_weka_overview(
    credential: str = Query(...),
    namespace: str = Query("default", description="Namespace of the WarpCredential CR"),
    bust: int = Query(0, description="Set bust=1 to bypass the 60s cache and refetch"),
):
    """Proxy WEKA REST API overview data for a named weka-storage credential.

    Exchanges WEKA_API_USERNAME + WEKA_API_TOKEN for a Bearer token, then calls
    fileSystems, cluster, and containers endpoints in parallel.
    Results are cached per credential for 60 seconds; ?bust=1 bypasses the cache.
    API-05, API-06, API-08.
    """
    # Step 1: Validate credential name (T-23-03-03 — DNS-1123, no path traversal)
    if not _CREDENTIAL_NAME_RE.match(credential):
        return JSONResponse({"ok": False, "error": "invalid credential name"}, status_code=400)

    # Step 2: Normalize namespace
    ns = namespace.strip() or "default"

    # Step 3: Namespace-scoped cache key (T-23-03-04 — prevents cross-namespace cache reads)
    cache_key = f"{ns}/{credential}"

    # Step 4: Cache check (skip when bust=1)
    if bust != 1:
        entry = _weka_overview_cache.get(cache_key)
        if entry and (time.time() - entry["ts"]) < _WEKA_CACHE_TTL_SECONDS:
            logger.info("weka overview: credential=%s namespace=%s cache_hit=True", credential, ns)
            return JSONResponse({"ok": True, "data": entry["data"], "cached": True})

    # Step 5: Cache miss or bust=1 — fetch from WEKA
    try:
        load_kube_config()

        # 5b: Resolve credential Secret
        try:
            endpoint, _username, token = await asyncio.to_thread(
                _resolve_weka_credential_secret, credential, ns
            )
        except ApiException as ae:
            status_code = ae.status if ae.status in (400, 401, 403, 404) else 500
            return JSONResponse(
                {"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"},
                status_code=status_code,
            )
        except RuntimeError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

        # Validate endpoint before using it (SSRF guard)
        try:
            _validate_weka_endpoint(endpoint)
        except RuntimeError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

        # 5c: Resolve the Bearer token. WEKA API tokens (weka user generate-api-token)
        # are long-lived JWTs used directly as Bearer; refresh-style tokens are first
        # exchanged via /api/v2/login/refresh. _weka_resolve_bearer_token handles both.
        bearer = await asyncio.to_thread(_weka_resolve_bearer_token, endpoint, token)

        # 5d-e: Build request headers; the token is never logged
        headers = {"Authorization": f"Bearer {bearer}", "Accept": "application/json"}
        base = endpoint.rstrip("/")

        # 5f: Three parallel WEKA data calls (D-05)
        # return_exceptions=True lets one endpoint fail without cancelling the others
        results = await asyncio.gather(
            asyncio.to_thread(_weka_get_json, f"{base}/api/v2/fileSystems", headers),
            asyncio.to_thread(_weka_get_json, f"{base}/api/v2/cluster", headers),
            asyncio.to_thread(_weka_get_json, f"{base}/api/v2/containers", headers),
            return_exceptions=True,
        )

        # The cluster call is essential. If it failed, the token was rejected or the
        # cluster is unreachable — surface a clear 502 instead of an empty overview.
        if isinstance(results[1], Exception):
            return JSONResponse(
                {"ok": False, "error": "WEKA authentication failed or cluster unreachable"},
                status_code=502,
            )

        fs_resp = results[0] if not isinstance(results[0], Exception) else []
        cluster_resp = results[1]
        containers_resp = results[2] if not isinstance(results[2], Exception) else []

        # 5g-h: Assemble response
        fetched_at_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        data = _assemble_weka_overview(fs_resp, cluster_resp, containers_resp, fetched_at_iso)

        # 5i: Update cache (D-06)
        _weka_overview_cache[cache_key] = {"ts": time.time(), "data": data}

        logger.info("weka overview: credential=%s namespace=%s cache_hit=False", credential, ns)

        # 5j: Return response
        return JSONResponse({"ok": True, "data": data, "cached": False})

    except ApiException as ae:
        status_code = ae.status if ae.status in (400, 401, 403, 404) else 500
        return JSONResponse(
            {"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"},
            status_code=status_code,
        )
    except Exception as e:
        # WEKA-side failures (helpers raise RuntimeError with "WEKA " prefix) → 502
        # Other unexpected errors → 500
        err_str = str(e)
        if err_str.startswith("WEKA "):
            return JSONResponse({"ok": False, "error": err_str}, status_code=502)
        return JSONResponse({"ok": False, "error": err_str}, status_code=500)


@app.get("/api/namespaces")
async def list_namespaces():
    """Return a simple list of namespace names available in the cluster.
    Example response: {"ok": true, "items": ["default", "kube-system", ...]}
    """
    try:
        load_kube_config()
        core = client.CoreV1Api()
        ns_list = core.list_namespace(_request_timeout=(5, 10))
        names = []
        for ns in (ns_list.items or []):
            md = getattr(ns, 'metadata', None)
            if md and md.name:
                names.append(md.name)
        names.sort()
        return JSONResponse({"ok": True, "items": names})
    except ApiException as ae:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


def get_auth_status() -> Dict[str, Any]:
    """Return whether the app can authenticate to the Kubernetes API, with connection details.

    Semantics:
    - authenticated=True (green) if we can load config and successfully call the Version API.
    - 403 Forbidden is treated as authenticated (identity recognized) but unauthorized (still green for auth light).
    - 401 or other failures → authenticated=False (red).

    Also returns `details` describing how we connected.
    """
    mode = os.getenv("KUBERNETES_AUTH_MODE", "auto").lower().strip()

    # Default details
    details: Dict[str, Any] = {
        "source": "unknown",   # incluster | kubeconfig | unknown
        "server": None,
        "context": None,
        "namespace": None,
        "user": None,
    }

    def detect_details_after_load() -> None:
        """Populate `details` by inspecting the active client configuration and environment."""
        try:
            # Host (API server) from the active client config
            cfg = client.Configuration.get_default_copy() if hasattr(client.Configuration, 'get_default_copy') else client.Configuration()
            host = getattr(cfg, 'host', None)
            if not host:
                # Fallback via ApiClient
                try:
                    api_client = client.ApiClient()
                    host = getattr(api_client.configuration, 'host', None)
                except Exception:
                    host = None
            details["server"] = host
        except Exception:
            pass

        # Heuristics to determine source/context/namespace/user
        sa_token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        sa_ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
        svc_host_env = os.getenv("KUBERNETES_SERVICE_HOST")
        try:
            # Prefer in-cluster if SA files/env present and host resembles in-cluster
            incluster_hint = (os.path.exists(sa_token_path) and (svc_host_env is not None or (details.get("server") or "").endswith(".svc")))
            if incluster_hint:
                details["source"] = "incluster"
                # Namespace from SA file if available
                try:
                    if os.path.exists(sa_ns_path):
                        with open(sa_ns_path, 'r') as f:
                            details["namespace"] = f.read().strip() or None
                except Exception:
                    pass
                details["user"] = "ServiceAccount (in-cluster)"
                return
        except Exception:
            pass

        # Otherwise, attempt to obtain kubeconfig context info
        try:
            contexts, current = config.list_kube_config_contexts()
            cur_ctx_name = (current or {}).get('name')
            cur_ctx = (current or {}).get('context') or {}
            ns = cur_ctx.get('namespace') or 'default'
            user = cur_ctx.get('user')
            details.update({
                "source": "kubeconfig",
                "context": cur_ctx_name,
                "namespace": ns,
                "user": user,
            })
        except Exception:
            # Leave as unknown if not available
            pass

    try:
        # Try to load configuration (in-cluster or kubeconfig depending on env)
        load_kube_config()
        detect_details_after_load()
    except Exception as e:
        return {"authenticated": False, "mode": mode, "message": str(e), "details": details}

    try:
        version_api = client.VersionApi()
        ver = version_api.get_code()
        return {
            "authenticated": True,
            "mode": mode,
            "message": f"Connected to Kubernetes {getattr(ver, 'git_version', 'unknown')}",
            "details": details,
        }
    except ApiException as e:
        # 403 generally means the token is valid (authenticated) but lacks permission
        if e.status == 403:
            return {"authenticated": True, "mode": mode, "message": f"Authenticated but forbidden (RBAC): {e.reason}", "details": details}
        if e.status == 401:
            return {"authenticated": False, "mode": mode, "message": f"Unauthorized: {e.reason}", "details": details}
        return {"authenticated": False, "mode": mode, "message": f"API error: {e.reason}", "details": details}
    except Exception as e:
        return {"authenticated": False, "mode": mode, "message": str(e), "details": details}


@app.get("/auth-status")
async def auth_status_endpoint():
    return JSONResponse(get_auth_status())


@app.get("/storage-classes")
async def list_storage_classes():
    """Return available StorageClass names in the cluster."""
    try:
        load_kube_config()
        storage_api = client.StorageV1Api()
        sc_list = storage_api.list_storage_class()
        names = sorted([sc.metadata.name for sc in (sc_list.items or []) if sc.metadata and sc.metadata.name])
        return JSONResponse({"ok": True, "items": names})
    except ApiException as e:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {e.reason}", "status": e.status}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


_QTY_FACTORS = {
    "Ki": 1024, "Mi": 1024 ** 2, "Gi": 1024 ** 3, "Ti": 1024 ** 4, "Pi": 1024 ** 5, "Ei": 1024 ** 6,
    "k": 1000, "M": 1000 ** 2, "G": 1000 ** 3, "T": 1000 ** 4, "P": 1000 ** 5, "E": 1000 ** 6,
}


def _parse_k8s_quantity_bytes(qty: str) -> int:
    """Best-effort parse of a Kubernetes storage quantity (e.g. '10Gi') to bytes. Returns 0 on failure."""
    if not qty:
        return 0
    qty = qty.strip()
    for suffix in ("Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "k", "M", "G", "T", "P", "E"):
        if qty.endswith(suffix):
            try:
                return int(float(qty[:-len(suffix)]) * _QTY_FACTORS[suffix])
            except ValueError:
                return 0
    try:
        return int(float(qty))
    except ValueError:
        return 0


@app.get("/api/weka/storage-classes")
async def list_weka_storage_classes():
    """Return WEKA CSI StorageClasses (provisioner csi.weka.io) with admin detail and live PVC usage.

    Used by the Settings admin page. Distinct from /storage-classes, which returns a flat
    name list for blueprint deploy forms — that endpoint's shape is left unchanged.
    """
    try:
        load_kube_config()
        storage_api = client.StorageV1Api()
        core_api = client.CoreV1Api()
        sc_list = await asyncio.to_thread(storage_api.list_storage_class)
        pvc_list = await asyncio.to_thread(core_api.list_persistent_volume_claim_for_all_namespaces)

        # Aggregate live PVC usage per storage class (count + bound capacity).
        usage: Dict[str, Dict[str, int]] = {}
        for pvc in (pvc_list.items or []):
            sc_name = pvc.spec.storage_class_name if pvc.spec else None
            if not sc_name:
                continue
            entry = usage.setdefault(sc_name, {"count": 0, "bytes": 0})
            entry["count"] += 1
            cap = (pvc.status.capacity or {}).get("storage") if pvc.status else None
            if not cap and pvc.spec and pvc.spec.resources and pvc.spec.resources.requests:
                cap = pvc.spec.resources.requests.get("storage")
            entry["bytes"] += _parse_k8s_quantity_bytes(cap or "")

        items = []
        for sc in (sc_list.items or []):
            if (sc.provisioner or "") != "csi.weka.io":
                continue
            md = sc.metadata
            ann = md.annotations or {}
            params = sc.parameters or {}
            u = usage.get(md.name, {"count": 0, "bytes": 0})
            items.append({
                "name": md.name,
                "isDefault": ann.get("storageclass.kubernetes.io/is-default-class") == "true",
                "provisioner": sc.provisioner,
                "reclaimPolicy": sc.reclaim_policy,
                "volumeBindingMode": sc.volume_binding_mode,
                "allowVolumeExpansion": bool(sc.allow_volume_expansion),
                "filesystemName": params.get("filesystemName"),
                "filesystemGroupName": params.get("filesystemGroupName"),
                "volumeType": params.get("volumeType"),
                "capacityEnforcement": params.get("capacityEnforcement"),
                "pvcCount": u["count"],
                "boundBytes": u["bytes"],
            })
        # Default class first, then alphabetical.
        items.sort(key=lambda x: (not x["isDefault"], x["name"]))
        return JSONResponse({"ok": True, "items": items})
    except ApiException as e:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {e.reason}", "status": e.status}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/wekaappstore-exists")
async def wekaappstore_exists(
    name: str = Query(..., description="WekaAppStore CR name to check"),
    namespace: str = Query("default", description="Namespace to look in"),
):
    """Return whether a named WekaAppStore CR exists in a namespace, and its phase.

    Used for blueprint dependency gating — e.g. the semantic-search blueprint requires the
    AIDP blueprint (CR 'weka-aidp') to already be present in the target namespace.
    """
    ns = (namespace or "default").strip() or "default"
    if not _CREDENTIAL_NAME_RE.match(name):
        return JSONResponse({"ok": False, "error": "invalid name"}, status_code=400)
    try:
        load_kube_config()
        custom_api = client.CustomObjectsApi()
        cr = await asyncio.to_thread(
            custom_api.get_namespaced_custom_object,
            group="warp.io", version="v1alpha1",
            namespace=ns, plural="wekaappstores", name=name,
        )
        phase = (cr.get("status", {}) or {}).get("appStackPhase")
        return JSONResponse({"ok": True, "exists": True, "phase": phase, "variables": _cr_gui_variables(cr)})
    except ApiException as e:
        if e.status == 404:
            return JSONResponse({"ok": True, "exists": False, "phase": None})
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {e.reason}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/refresh-blueprints")
async def refresh_blueprints():
    """Re-sync the warp-blueprints checkout that BLUEPRINTS_DIR is served from.

    git-sync runs once as an init container; this re-pulls the current worktree in place
    (fetch + reset --hard) so newly published blueprints appear without restarting the pod.
    BLUEPRINTS_DIR is a read-only bind mount pinned to this worktree, so updating files here
    is reflected immediately for blueprint listing.
    """
    root = os.getenv("GIT_SYNC_ROOT", "/manifests")
    link = os.getenv("GIT_SYNC_LINK", "manifests")
    branch = os.getenv("GIT_SYNC_BRANCH", "main") or "main"
    worktree = os.path.join(root, link)

    # The git-sync worktree has a `.git` file (gitdir pointer). Absence means this
    # environment is not git-sync-managed (e.g. local dev) — nothing to refresh.
    if not os.path.exists(os.path.join(worktree, ".git")):
        return JSONResponse(
            {"ok": False, "error": "Blueprints are not managed by git-sync in this environment"},
            status_code=400,
        )

    def _run(args):
        return subprocess.run(["git", "-C", worktree, *args], capture_output=True, text=True, timeout=60)

    def _rev():
        r = _run(["rev-parse", "--short", "HEAD"])
        return r.stdout.strip() if r.returncode == 0 else None

    try:
        before = await asyncio.to_thread(_rev)
        fetch = await asyncio.to_thread(_run, ["fetch", "--depth=1", "origin", branch])
        if fetch.returncode != 0:
            logger.warning("refresh-blueprints fetch failed: %s", fetch.stderr.strip())
            return JSONResponse(
                {"ok": False, "error": "git fetch failed; check cluster network access to the blueprints repo"},
                status_code=502,
            )
        reset = await asyncio.to_thread(_run, ["reset", "--hard", "FETCH_HEAD"])
        if reset.returncode != 0:
            logger.warning("refresh-blueprints reset failed: %s", reset.stderr.strip())
            return JSONResponse({"ok": False, "error": "git reset failed"}, status_code=500)
        after = await asyncio.to_thread(_rev)
        logger.info("refresh-blueprints: %s -> %s (branch=%s)", before, after, branch)
        return JSONResponse({"ok": True, "changed": before != after, "before": before, "after": after})
    except subprocess.TimeoutExpired:
        return JSONResponse({"ok": False, "error": "git operation timed out"}, status_code=504)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


def parse_x_variables(yaml_text: str) -> dict:
    """Extract the x-variables schema block from raw blueprint YAML text. Returns {} on any parse failure."""
    if not yaml_text:
        return {}
    try:
        data = next(yaml.safe_load_all(yaml_text), None)
        if not isinstance(data, dict):
            return {}
        x_vars = data.get("x-variables")
        if not isinstance(x_vars, dict):
            return {}
        return x_vars
    except Exception:
        return {}


# Raised default deploy timeout (seconds).  Used when a blueprint has no x-deploy-timeout
# key or the key is malformed.  Value is within the 1800–2400s band (35 minutes).
DEFAULT_DEPLOY_TIMEOUT_SECONDS = 2100


def parse_deploy_timeout(yaml_text: str) -> int:
    """Read the top-level x-deploy-timeout key (seconds) from raw blueprint YAML text.

    Returns the blueprint value when present and valid (positive integer).
    Falls back to DEFAULT_DEPLOY_TIMEOUT_SECONDS on any parse failure, missing key,
    non-positive value, or non-integer value — so a malformed blueprint cannot set an
    unbounded or zero deadline (T-29-02 mitigate).
    """
    if not yaml_text:
        return DEFAULT_DEPLOY_TIMEOUT_SECONDS
    try:
        data = next(yaml.safe_load_all(yaml_text), None)
        if not isinstance(data, dict):
            return DEFAULT_DEPLOY_TIMEOUT_SECONDS
        raw = data.get("x-deploy-timeout")
        if raw is None:
            return DEFAULT_DEPLOY_TIMEOUT_SECONDS
        val = int(raw)
        if val <= 0:
            return DEFAULT_DEPLOY_TIMEOUT_SECONDS
        return val
    except Exception:
        return DEFAULT_DEPLOY_TIMEOUT_SECONDS


def build_quay_dockerconfigjson(user: str, password: str) -> str:
    """Build the .dockerconfigjson value for a quay.io pull secret.

    Returns a compact JSON string suitable for a Kubernetes Secret stringData field.
    The auth value uses base64.b64encode which produces no trailing newlines —
    auths["quay.io"]["auth"] base64-decodes to exactly "user:password" with no
    trailing bytes (D-04, T-29-04 mitigate).
    """
    auth = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
    return json.dumps({"auths": {"quay.io": {"auth": auth}}}, separators=(",", ":"))


def split_endpoints(join_ip_ports: str) -> dict:
    """Split a comma-delimited host:port string into both forms needed by the blueprint.

    Returns a dict with two keys:
    - join_ip_ports_list: a json.dumps'd string of the list (double-quoted JSON array),
      suitable for `joinIpPorts: [[ join_ip_ports_list ]]` in the WekaClient CR.
    - endpoints_csv: the comma-joined normalized string for the CSI API secret.

    Whitespace around each entry is stripped; empty entries are dropped (D-05, T-29-05).
    """
    entries = [e.strip() for e in (join_ip_ports or "").split(",") if e.strip()]
    return {
        "join_ip_ports_list": json.dumps(entries),
        "endpoints_csv": ",".join(entries),
    }


def _blueprint_cr_identity(yaml_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (cr_name, cr_namespace) of the WekaAppStore doc in a blueprint file, else (None, None)."""
    if not yaml_path:
        return (None, None)
    path = yaml_path if os.path.isabs(yaml_path) else os.path.join(PROJECT_ROOT, yaml_path)
    try:
        with open(path, "r") as f:
            for doc in yaml.safe_load_all(f):
                if isinstance(doc, dict) and doc.get("kind") == "WekaAppStore":
                    md = doc.get("metadata") or {}
                    return (md.get("name"), md.get("namespace"))
    except Exception:
        return (None, None)
    return (None, None)


def _cr_gui_variables(cr: dict) -> dict:
    """Recall the variables submitted when a WekaAppStore CR was deployed.

    Prefers the warp.io/gui-variables annotation stamped at deploy time; falls back
    to spec.appStack.variables for CRs deployed before stamping existed.
    """
    md = (cr or {}).get("metadata", {}) or {}
    raw = (md.get("annotations", {}) or {}).get("warp.io/gui-variables")
    if raw:
        try:
            v = json.loads(raw)
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    spec_vars = (((cr.get("spec") or {}).get("appStack") or {}).get("variables")) or {}
    return spec_vars if isinstance(spec_vars, dict) else {}


# ---------------------------------------------------------------------------
# Secret-key predicate and redaction helpers (SEC-01, D-09, D-10)
#
# _is_secret_key is the SINGLE source of truth used by BOTH _safe_gui_variables
# (annotation allowlist) and _redact_secrets (SSE message redactor).
# ---------------------------------------------------------------------------

_SECRET_KEY_SUBSTRINGS = ("password", "token", "secret")
_SECRET_KEY_EXACT = "quay_dockerconfigjson"


def _is_secret_key(name: str) -> bool:
    """Return True if ``name`` identifies a secret variable.

    A key is secret when its lowercase form contains any of the substrings
    'password', 'token', or 'secret', OR when it equals exactly
    'quay_dockerconfigjson' (case-sensitive exact match, as it is a
    well-known derived key that does not contain the above substrings).
    """
    lower = name.lower()
    if any(sub in lower for sub in _SECRET_KEY_SUBSTRINGS):
        return True
    return name == _SECRET_KEY_EXACT


def _safe_gui_variables(user_vars: dict) -> dict:
    """Return a copy of ``user_vars`` with all secret keys removed.

    Used to stamp the warp.io/gui-variables CR annotation so that secret
    values (passwords, tokens, dockerconfigjson) are never persisted in etcd
    or visible via ``kubectl get wekaappstore -o yaml`` (T-29-07 mitigate).
    The original dict is not mutated.
    """
    return {k: v for k, v in user_vars.items() if not _is_secret_key(k)}


def _redact_secrets(message: str, user_vars: dict) -> str:
    """Replace occurrences of secret VALUES in ``message`` with '***'.

    Builds the secret-value set from the same _is_secret_key predicate, so
    the annotation allowlist and the SSE redactor share one definition
    (T-29-09 mitigate). Only non-empty string values are replaced; empty
    values are skipped to avoid replacing every empty substring. Replacement
    is longest-first to prevent partial-overlap artifacts (T-29-08 mitigate).
    """
    secret_values = sorted(
        (str(v) for k, v in user_vars.items() if _is_secret_key(k) and v),
        key=len,
        reverse=True,
    )
    for secret in secret_values:
        message = message.replace(secret, "***")
    return message


# Variable names that should hold a URL/endpoint even when the blueprint only
# declares them as a plain string (most blueprints predate a dedicated url type).
_URL_FIELD_RE = re.compile(r"(?:^|_)(url|uri|endpoint|host|server|address)(?:_|$)", re.I)


def _validation_disabled(meta: dict) -> bool:
    """A blueprint can opt a variable out of format validation with `validate: false`."""
    return isinstance(meta, dict) and meta.get("validate") is False


def _is_hostname_field(meta: dict) -> bool:
    """True when a blueprint declares `format: hostname` — a bare host/FQDN/IP,
    not a URL (e.g. keycloak_fqdn, which the AIDP chart prefixes with a scheme)."""
    return isinstance(meta, dict) and (meta.get("format") or "").lower() == "hostname"


def _is_url_field(var_name: str, meta: dict) -> bool:
    """True if this variable is expected to contain a URL/endpoint."""
    if not isinstance(meta, dict):
        meta = {}
    if _validation_disabled(meta) or _is_hostname_field(meta):
        return False
    if (meta.get("type") or "").lower() == "url" or (meta.get("format") or "").lower() == "url":
        return True
    return bool(_URL_FIELD_RE.search(var_name or ""))


def _validate_variable_value(var_name: str, meta: dict, value: str) -> Optional[str]:
    """Return a human-readable error if value is malformed for this variable, else None.

    Empty values are accepted here — required-ness is enforced separately. A
    blueprint may set `validate: false` to skip checks, or `format: hostname` to
    require a bare host/FQDN/IP (rejecting an http:// scheme, a path, or spaces).
    """
    if _validation_disabled(meta):
        return None
    value = (value or "").strip()
    if not value:
        return None
    if _is_hostname_field(meta):
        if "://" in value or value.lower().startswith(("http:", "https:")):
            return f"{var_name}: enter a hostname only — remove the http:// or https:// prefix"
        if "/" in value:
            return f"{var_name}: enter a hostname only — remove the path or trailing slash"
        if any(ch.isspace() for ch in value):
            return f"{var_name}: hostname must not contain spaces"
        return None
    if _is_url_field(var_name, meta):
        if any(ch.isspace() for ch in value):
            return f"{var_name}: URL must not contain spaces"
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return f"{var_name}: must be a valid URL (e.g. https://host.example.com)"
    return None


def find_blueprint(app_name: str, blueprints_dir: str = None) -> Optional[str]:
    """Scan BLUEPRINTS_DIR for a blueprint YAML file with an x-variables block matching app_name. Returns absolute path or None."""
    if blueprints_dir is None:
        blueprints_dir = BLUEPRINTS_DIR
    # Special case: cluster-init always maps to its fixed path
    if app_name == "cluster-init":
        return os.path.abspath(os.path.join(blueprints_dir, "cluster_init", "app-store-cluster-init.yaml"))
    if not blueprints_dir or not os.path.isdir(blueprints_dir):
        return None
    try:
        for root, _dirs, files in os.walk(blueprints_dir):
            for filename in files:
                if not (filename.endswith(".yaml") or filename.endswith(".yml")):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r") as f:
                        text = f.read()
                except Exception:
                    continue
                schema = parse_x_variables(text)
                if not schema:
                    continue
                stem = os.path.splitext(filename)[0]
                parent_dir = os.path.basename(root)
                if stem == app_name or parent_dir == app_name:
                    return os.path.abspath(filepath)
    except Exception:
        return None
    return None


@app.get("/blueprint/{name}", response_class=HTMLResponse)
async def blueprint_detail(request: Request, name: str):
    yaml_path = find_blueprint(name)
    status = await asyncio.to_thread(get_cluster_status)
    # Determine the blueprint's resource needs (declared x-requirements first, else
    # deep inference) and compare against the cluster's free capacity.
    reqs = compute_blueprint_requirements(yaml_path)
    meets = _compute_requirement_meets(reqs, status)
    # Prepare embedded image for OSS RAG blueprint
    oss_img_b64 = None
    if name == 'oss-rag':
        img_path = os.path.join(TEMPLATES_DIR, 'oss_rag.png')
        if os.path.exists(img_path):
            try:
                with open(img_path, 'rb') as f:
                    oss_img_b64 = base64.b64encode(f.read()).decode('ascii')
            except Exception:
                oss_img_b64 = None
    # Prepare embedded diagram for NeuralMesh AIDP blueprint
    aidp_img_b64 = None
    if name == 'neuralmesh-aidp':
        img_path = os.path.join(TEMPLATES_DIR, 'aidp_diagram.png')
        if os.path.exists(img_path):
            try:
                with open(img_path, 'rb') as f:
                    aidp_img_b64 = base64.b64encode(f.read()).decode('ascii')
            except Exception:
                aidp_img_b64 = None
    # Choose a specific template if present (except for RAG pages which keep generic)
    preferred = f"blueprint_{name}.html"
    use_template = "blueprint.html"
    try:
        if name not in {"oss-rag", "nvidia-rag"}:
            if os.path.exists(os.path.join(TEMPLATES_DIR, preferred)):
                use_template = preferred
    except Exception:
        use_template = "blueprint.html"

    # Resolve namespace for credential lookup (same pattern as settings_page:524-528)
    auth = await asyncio.to_thread(get_auth_status)
    detected_ns = (auth.get("details", {}) or {}).get("namespace") if isinstance(auth, dict) else None
    ns = detected_ns or "default"
    credentials_by_type = await _get_credentials_by_type(ns)

    variable_schema: dict = {}
    if yaml_path and os.path.isfile(yaml_path):
        try:
            with open(yaml_path, "r") as _f:
                variable_schema = parse_x_variables(_f.read())
        except Exception:
            variable_schema = {}
    # Annotate URL/hostname fields so the template can give the browser an early
    # hint (HTML5 url input, or a hostname pattern that rejects a scheme/path).
    for _vname, _vmeta in (variable_schema or {}).items():
        if not isinstance(_vmeta, dict) or _vmeta.get("type") == "credential":
            continue
        if _is_hostname_field(_vmeta):
            _vmeta["_input_type"] = "text"
            _vmeta["_pattern"] = r"[^\s/]+(\.[^\s/]+)*"
            _vmeta["_title"] = "Hostname only, e.g. keycloak.example.com (no http:// and no path)"
        elif _is_url_field(_vname, _vmeta):
            _vmeta["_input_type"] = "url"

    # CR identity for install-state detection (Deploy↔Uninstall toggle, read-only fields).
    cr_name, cr_namespace = _blueprint_cr_identity(yaml_path)

    return templates.TemplateResponse(
        request,
        use_template,
        {
            "request": request,
            "name": name,
            "yaml_path": yaml_path,
            "status": status,
            "requirements": reqs,
            "meets": meets,
            "oss_img_b64": oss_img_b64,
            "aidp_img_b64": aidp_img_b64,
            "logo_b64": LOGO_B64,
            "glocomp_logo_b64": GLOCOMP_LOGO_B64,
            "tokenvisor_logo_b64": TOKENVISOR_LOGO_B64,
            "tokenvisor_arch_b64": TOKENVISOR_ARCH_B64,
            "credentials_by_type": credentials_by_type,
            "variable_schema": variable_schema,
            "available_creds": credentials_by_type,
            "cr_name": cr_name,
            "cr_namespace": cr_namespace or "default",
        },
    )


@app.post("/deploy")
async def deploy(app_name: str = Form(...), namespace: str = Form("default")):
    yaml_path = find_blueprint(app_name)
    if not yaml_path:
        return JSONResponse({"ok": False, "error": "Unknown app"}, status_code=400)

    # For namespace-preserving apps, keep the namespaces defined in the YAML (do not override)
    effective_ns = "" if app_name in NAMESPACE_PRESERVING_APPS else namespace

    try:
        # Apply with explicit namespace overrides so the blueprint deploys into the chosen namespace
        result = apply_blueprint_with_namespace(yaml_path, namespace=effective_ns)
        return JSONResponse({"ok": True, "result": result})
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    except ApiException as e:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {e.reason}", "status": e.status}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/install-crd")
async def install_crd():
    try:
        result = install_warrp_crd()
        return JSONResponse({"ok": True, "result": result})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/uninit-cluster")
async def uninit_cluster():
    """Un-initialize the cluster by deleting the CR defined in the cluster-init blueprint YAML.

    This mimics: kubectl delete -f cluster_init/app-store-cluster-init.yaml
    using the Kubernetes Python client. We parse the YAML to get name/namespace
    so we delete the correct object even if the namespace was customized in the file.
    """
    try:
        # Resolve the blueprint path (same mapping used by the deploy/init button)
        yaml_path = os.path.join(BLUEPRINTS_DIR, "cluster_init", "app-store-cluster-init.yaml")
        if not os.path.isabs(yaml_path):
            yaml_path = os.path.join(PROJECT_ROOT, yaml_path)
        if not os.path.exists(yaml_path):
            return JSONResponse({"ok": False, "error": f"YAML file not found: {yaml_path}"}, status_code=404)

        load_kube_config()
        k8s_client = client.ApiClient()
        co_api = client.CustomObjectsApi(k8s_client)

        # Load all docs and find the WekaAppStore CR to delete (defensive for multi-doc files)
        with open(yaml_path, "r") as f:
            docs = list(yaml.safe_load_all(f))

        deleted: List[Dict[str, Any]] = []
        found_any = False
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            api_version = str(doc.get("apiVersion") or "")
            kind = str(doc.get("kind") or "")
            if not api_version or not kind:
                continue
            md = doc.get("metadata") or {}
            name = md.get("name")
            ns = md.get("namespace") or "default"

            # We only expect a WekaAppStore CR here, but guard for either warp.io or warrp.io variants
            if api_version.startswith("warp.io/") and kind == "WekaAppStore":
                found_any = True
                try:
                    resp = co_api.delete_namespaced_custom_object(
                        group="warp.io",
                        version=api_version.split("/", 1)[1],
                        namespace=ns,
                        plural="wekaappstores",
                        name=name,
                    )
                    # resp can be a dict-like already; ensure serializable
                    deleted.append({"group": "warp.io", "version": api_version.split("/", 1)[1], "kind": kind, "name": name, "namespace": ns})
                except ApiException as ae:
                    if ae.status == 404:
                        deleted.append({"group": "warp.io", "version": api_version.split("/", 1)[1], "kind": kind, "name": name, "namespace": ns, "status": "Not present"})
                    else:
                        raise

        if not found_any:
            # Fallback: attempt delete by the conventional name if YAML didn't include the expected doc
            try:
                resp = co_api.delete_namespaced_custom_object(
                    group="warp.io",
                    version="v1alpha1",
                    namespace="default",
                    plural="wekaappstores",
                    name="app-store-cluster-init",
                )
                deleted.append({"group": "warp.io", "version": "v1alpha1", "kind": "WekaAppStore", "name": "app-store-cluster-init", "namespace": "default"})
            except ApiException as ae:
                if ae.status == 404:
                    deleted.append({"group": "warp.io", "version": "v1alpha1", "kind": "WekaAppStore", "name": "app-store-cluster-init", "namespace": "default", "status": "Not present"})
                else:
                    raise

        return JSONResponse({"ok": True, "deleted": deleted})
    except ApiException as e:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {e.reason}", "status": e.status}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# Convenience roots for health and readiness
# Liveness: process is up and able to serve HTTP
@app.get("/healthz")
async def healthz():
    return {"ok": True}

# ---------------------------------------------------------------------------
# WEKA Overview Proxy — helpers, cache, and route
# ---------------------------------------------------------------------------

_weka_overview_cache: Dict[str, Dict[str, Any]] = {}
_WEKA_CACHE_TTL_SECONDS = 60


def _weka_ssl_context() -> ssl.SSLContext:
    """Return an SSL context for WEKA API calls.

    By default uses a verifying context (ssl.create_default_context()).
    Set WEKA_OVERVIEW_INSECURE_TLS=true to disable certificate verification
    (needed for self-signed certs on typical WEKA production clusters).
    """
    ctx = ssl.create_default_context()
    if os.getenv("WEKA_OVERVIEW_INSECURE_TLS", "false").lower() in ("1", "true", "yes"):
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _validate_weka_endpoint(endpoint: str) -> None:
    """Raise RuntimeError if endpoint is not a safe https:// or http:// URL."""
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in ("https", "http"):
        raise RuntimeError(f"WEKA endpoint must use https:// or http:// scheme, got: {parsed.scheme!r}")
    host = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_loopback or addr.is_link_local or addr.is_reserved:
            raise RuntimeError(f"WEKA endpoint resolves to a forbidden address: {host!r}")
    except ValueError:
        pass  # hostname (not a bare IP) — prefix check still applies
    forbidden_prefixes = ("127.", "169.254.", "0.", "::1", "fc", "fe80")
    if any(host.startswith(p) for p in forbidden_prefixes) or host in ("localhost",):
        raise RuntimeError(f"WEKA endpoint resolves to a forbidden host: {host!r}")


def _weka_get_json(url: str, headers: Dict[str, str], timeout: float = 15.0) -> Dict[str, Any]:
    """Issue a GET request to the WEKA REST API and return the parsed JSON response.

    Never includes header values in error messages (T-23-03-01, T-23-03-08).
    """
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_weka_ssl_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"WEKA API call failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"WEKA API call failed: connection error") from exc


def _weka_post_json(url: str, payload: Dict[str, Any], timeout: float = 15.0) -> Dict[str, Any]:
    """Issue a POST request to the WEKA REST API and return the parsed JSON response.

    Never logs or includes payload contents in error messages (T-23-03-01, T-23-03-08).
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_weka_ssl_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"WEKA login failed: {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"WEKA login failed: {exc.reason}") from exc


def _weka_resolve_bearer_token(endpoint: str, token: str) -> str:
    """Resolve the Bearer token to use against the WEKA REST API from a stored API token.

    WEKA `weka user generate-api-token` produces a long-lived JWT that is used
    directly as a Bearer token — no login exchange. Older refresh-style tokens must
    first be exchanged via POST /api/v2/login/refresh for a short-lived access token.

    Strategy: attempt the refresh exchange; if WEKA rejects the token there (it is not
    a refresh token) the stored token is used directly as the Bearer token. This is the
    common case for generated API tokens. Never logs the token.

    The token is NOT validated here — the caller surfaces auth failures from the
    subsequent data calls. Never logs the token or the returned access token.
    """
    endpoint = endpoint.rstrip("/")
    try:
        resp = _weka_post_json(f"{endpoint}/api/v2/login/refresh", {"refresh_token": token})
    except RuntimeError:
        # Not a refresh token (or refresh endpoint unreachable) — long-lived API
        # tokens are used directly as the Bearer token.
        return token

    access_token = resp.get("access_token")
    if access_token is None and isinstance(resp.get("data"), dict):
        access_token = resp["data"].get("access_token")
    if access_token is None:
        access_token = resp.get("token")
    return str(access_token) if access_token is not None else token


def _resolve_weka_credential_secret(credential_name: str, namespace: str) -> Tuple[str, str, str]:
    """Look up a weka-storage WarpCredential CR and return (endpoint, username, token).

    Validates the CR type is weka-storage, then reads the corresponding raw Secret
    (warp-cred-<credential_name> by default, or the name stored in spec.secretRef.name).
    Raises RuntimeError if the credential is the wrong type or if the Secret is missing
    any required key.  Never returns or logs credential values.
    """
    co_api = client.CustomObjectsApi()
    cr = co_api.get_namespaced_custom_object(
        group="warp.io",
        version="v1alpha1",
        namespace=namespace,
        plural="warpcredentials",
        name=credential_name,
    )

    cr_type = (cr.get("spec") or {}).get("type")
    if cr_type != "weka-storage":
        raise RuntimeError(f"credential {credential_name} is not of type weka-storage")

    secret_name = ((cr.get("spec") or {}).get("secretRef") or {}).get("name") or f"warp-cred-{credential_name}"

    core = client.CoreV1Api()
    secret = core.read_namespaced_secret(name=secret_name, namespace=namespace)
    data = secret.data or {}

    required_keys = _CREDENTIAL_TYPE_KEYS["weka-storage"]["secret_data_keys"]
    decoded: Dict[str, str] = {}
    for key in required_keys:
        if key not in data:
            raise RuntimeError(f"weka-storage Secret {secret_name} missing key {key}")
        decoded[key] = base64.b64decode(data[key]).decode("utf-8")

    return decoded["WEKA_API_ENDPOINT"], decoded["WEKA_API_USERNAME"], decoded["WEKA_API_TOKEN"]


def _assemble_weka_overview(
    filesystems_resp: Any,
    cluster_resp: Any,
    containers_resp: Any,
    fetched_at_iso: str,
) -> Dict[str, Any]:
    """Assemble the API-06 WEKA overview response shape from raw WEKA API responses.

    Pure function (no I/O) — tolerates multiple field-name variants per the PRD
    "verify against Swagger" note and T-23-03-07 (field name drift across WEKA versions).
    Never copies the input dicts verbatim; only whitelisted fields appear in the output
    (T-23-03-01).
    """
    # --- Filesystems ---
    # Response may be a list directly or {"data": [...]}
    if isinstance(filesystems_resp, list):
        fs_items = filesystems_resp
    elif isinstance(filesystems_resp, dict):
        fs_items = filesystems_resp.get("data") or []
        if not isinstance(fs_items, list):
            fs_items = []
    else:
        fs_items = []

    filesystems = []
    for fs in fs_items:
        name = fs.get("name")
        total = fs.get("total_budget") or fs.get("size") or fs.get("totalBytes") or 0
        used = fs.get("used_total") or fs.get("used_size") or fs.get("usedBytes") or 0
        total = int(total)
        used = int(used)
        used_percent = round((used / total) * 100, 2) if total else 0
        filesystems.append({
            "name": name,
            "totalBytes": total,
            "usedBytes": used,
            "usedPercent": used_percent,
        })

    # --- Cluster capacity ---
    # Tolerate {"data": {...}} wrapper
    if isinstance(cluster_resp, dict) and "data" in cluster_resp and isinstance(cluster_resp["data"], dict):
        cluster_data = cluster_resp["data"]
    elif isinstance(cluster_resp, dict):
        cluster_data = cluster_resp
    else:
        cluster_data = {}

    capacity_block = cluster_data.get("capacity")
    capacity: Dict[str, Any] = {}
    if isinstance(capacity_block, dict):
        c_total = capacity_block.get("total_bytes") or capacity_block.get("total") or capacity_block.get("totalBytes") or 0
        c_used = capacity_block.get("used_bytes") or capacity_block.get("used") or capacity_block.get("usedBytes") or 0
        c_total = int(c_total)
        c_used = int(c_used)
        c_avail = c_total - c_used
        c_pct = round((c_used / c_total) * 100, 2) if c_total else 0
        capacity = {
            "totalBytes": c_total,
            "usedBytes": c_used,
            "availableBytes": c_avail,
            "usedPercent": c_pct,
        }
    else:
        # Fallback: sum filesystem totals/used when cluster capacity dict is absent
        fallback_total = sum(fs["totalBytes"] for fs in filesystems)
        fallback_used = sum(fs["usedBytes"] for fs in filesystems)
        fallback_avail = fallback_total - fallback_used
        fallback_pct = round((fallback_used / fallback_total) * 100, 2) if fallback_total else 0
        capacity = {
            "totalBytes": fallback_total,
            "usedBytes": fallback_used,
            "availableBytes": fallback_avail,
            "usedPercent": fallback_pct,
            "capacity_source": "fallback-sum",
        }

    # --- Backend nodes ---
    if isinstance(containers_resp, list):
        container_items = containers_resp
    elif isinstance(containers_resp, dict):
        container_items = containers_resp.get("data") or []
        if not isinstance(container_items, list):
            container_items = []
    else:
        container_items = []

    backend_nodes = []
    for c in container_items:
        # Determine if this is a BACKEND role
        role = c.get("role") or c.get("mode")
        roles = c.get("roles")
        is_backend = (
            role == "BACKEND"
            or (isinstance(roles, list) and "BACKEND" in roles)
        )
        if not is_backend:
            continue

        # Resolve IP
        ip = c.get("ip") or c.get("ip_address") or c.get("management_ip")
        if ip is None:
            ips = c.get("ips")
            if isinstance(ips, list):
                for candidate in ips:
                    if not candidate.startswith("127.") and not candidate.startswith("169.254."):
                        ip = candidate
                        break
        if ip is None:
            continue
        # Skip loopback and link-local
        if ip.startswith("127.") or ip.startswith("169.254."):
            continue
        backend_nodes.append({"ip": ip})

    return {
        "capacity": capacity,
        "filesystems": filesystems,
        "backendNodes": backend_nodes,
        "fetchedAt": fetched_at_iso,
    }


# Readiness: dependent systems are reachable so we can actually do useful work
# This differs from liveness: a pod can be alive but not ready if, for example,
# the Kubernetes API or required templates/assets are unavailable yet.
_last_ready_cache: dict = {"ts": 0.0, "resp": {"ok": False, "ready": False}}

@app.get("/readyz")
async def readyz():
    try:
        # Simple cache to avoid hammering APIs during frequent probes
        ttl = float(os.getenv("READINESS_TTL_SECONDS", "5"))
        now = time.time()
        if ttl > 0 and (now - _last_ready_cache["ts"]) < ttl:
            return JSONResponse(_last_ready_cache["resp"], status_code=200 if _last_ready_cache["resp"].get("ready") else 503)

        problems = []

        # 1) Templates check (required by the GUI)
        idx_path = os.path.join(TEMPLATES_DIR, "index.html")
        if not os.path.isfile(idx_path):
            problems.append(f"missing template: {idx_path}")
        else:
            try:
                # ensure readable
                with open(idx_path, "r"):
                    pass
            except Exception as e:
                problems.append(f"template unreadable: {e}")

        # 2) Kubernetes API check (optional; can be disabled)
        if os.getenv("READINESS_SKIP_K8S", "false").lower() not in ("1", "true", "yes"):            
            try:
                load_kube_config()
                core = client.CoreV1Api()
                # very lightweight call with short timeouts
                core.list_namespace(limit=1, _request_timeout=(2, 3))
            except Exception as e:
                problems.append(f"k8s api: {str(e)}")

        ready = len(problems) == 0
        resp = {"ok": ready, "ready": ready}
        if not ready:
            resp["problems"] = problems

        # Cache result
        _last_ready_cache["ts"] = now
        _last_ready_cache["resp"] = resp

        return JSONResponse(resp, status_code=200 if ready else 503)
    except Exception as e:
        # Any unexpected error means not ready
        resp = {"ok": False, "ready": False, "error": str(e)}
        _last_ready_cache["ts"] = time.time()
        _last_ready_cache["resp"] = resp
        return JSONResponse(resp, status_code=503)


@app.get("/cluster-info")
async def get_cluster_info():
    """Get information about the current Kubernetes cluster and required components."""
    try:
        load_kube_config()
        # Try to get cluster name from the current context first
        try:
            contexts, active_context = config.list_kube_config_contexts()
            cluster_name = active_context.get('context', {}).get('cluster', 'Unknown Cluster')
        except Exception:
            # Fallback for in-cluster: try to get it from something like a well-known configmap or just 'Kubernetes Cluster'
            cluster_name = os.getenv("KUBERNETES_CLUSTER_NAME", "Kubernetes Cluster")

        # Check for WEKA Operator CRDs
        operator_crds = [
            "wekapolicies.weka.weka.io",
            "wekamanualoperations.weka.weka.io",
            "wekacontainers.weka.weka.io",
            "wekaclusters.weka.weka.io",
            "wekaclients.weka.weka.io",
            "driveclaims.weka.weka.io"
        ]
        
        crd_status = True
        try:
            api_extensions = client.ApiextensionsV1Api()
            crds = api_extensions.list_custom_resource_definition()
            installed_crds = [crd.metadata.name for crd in crds.items]
            for target_crd in operator_crds:
                if target_crd not in installed_crds:
                    crd_status = False
                    break
        except Exception:
            crd_status = False

        # Check for WEKA CSI Driver deployment
        csi_status = False
        try:
            apps_v1 = client.AppsV1Api()
            deployments = apps_v1.list_deployment_for_all_namespaces(field_selector="metadata.name=csi-wekafs-controller")
            if deployments.items:
                csi_status = True
        except Exception:
            csi_status = False
            
        return {
            "cluster_name": cluster_name,
            "weka_operator_installed": crd_status,
            "weka_csi_installed": csi_status
        }
    except Exception as e:
        return {
            "cluster_name": "Kubernetes Cluster",
            "weka_operator_installed": False,
            "weka_csi_installed": False,
            "error": str(e)
        }

@app.get("/welcome", response_class=HTMLResponse)
async def welcome_screen(request: Request):
    """Serve the welcome/initialization screen."""
    return templates.TemplateResponse(
        request,
        "welcome.html",
        {
            "request": request,
            "logo_b64": LOGO_B64,
            "title": "Welcome to WEKA App Store",
        },
    )

@app.get("/cluster-status")
async def get_cluster_status_endpoint(namespace: str = "default"):
    """Get the current initialization status."""
    try:
        load_kube_config()
        custom_api = client.CustomObjectsApi()
        
        # If we don't know the namespace, we might need to find where the CR is
        # But for now, we'll respect the namespace parameter, defaulting to 'default'
        cr = custom_api.get_namespaced_custom_object(
            group="warp.io",
            version="v1alpha1",
            namespace=namespace,
            plural="wekaappstores",
            name="app-store-cluster-init"
        )
        status = cr.get("status", {})
        phase = status.get("appStackPhase", "Pending")
        
        # Get progress message from the first condition or most recent event
        message = ""
        conditions = status.get("conditions", [])
        if conditions:
            # Sort conditions by last transition time if available
            try:
                sorted_conditions = sorted(conditions, key=lambda x: x.get('lastTransitionTime', ''), reverse=True)
                message = sorted_conditions[0].get("message", "")
            except Exception:
                message = conditions[0].get("message", "")
        
        # If any component is failed, report it in the message
        component_statuses = status.get("componentStatus", [])
        has_failure = False
        for comp in component_statuses:
            if comp.get("phase") in ["Failed", "Error"]:
                message = f"Error: {comp.get('name')} failed: {comp.get('message', 'Check logs')}"
                phase = "Failed"
                has_failure = True
                break
        
        # Also check for Error state in conditions
        if not has_failure:
            for cond in conditions:
                if (
                    cond.get("type") in ["Error", "Failed"]
                    or (cond.get("status") == "False" and cond.get("type") in ["Ready", "Initialized"])
                ):
                    if cond.get("reason") in ["Error", "Failed", "Degraded"]:
                         phase = "Failed"
                         message = f"Error: {cond.get('message', 'Initialization failed')}"
                         break
        
        logger.info(f"Cluster status poll: phase={phase}, message={message}")
        
        # If not failed, check for installing
        if phase != "Failed":
            for comp in component_statuses:
                if comp.get("phase") == "Installing":
                    message = f"Installing {comp.get('name')}..."
                    break

        # Extract redirect URL from HTTPRoute in the manifest if Ready
        redirect_url = None
        if phase == "Ready":
            try:
                # We can find the hostname in the app-store-cluster-init.yaml manifest
                init_manifest_path = os.path.join(BLUEPRINTS_DIR, "cluster_init", "app-store-cluster-init.yaml")
                
                if os.path.exists(init_manifest_path):
                    with open(init_manifest_path, 'r') as f:
                        docs = list(yaml.safe_load_all(f))
                        manifest = next((d for d in docs if isinstance(d, dict) and d.get("kind") == "WekaAppStore"), {})
                        app_stack = manifest.get("spec", {}).get("appStack", [])
                        for item in app_stack:
                            if item.get("name") == "envoy-route-appstore-gui":
                                route_manifest = yaml.safe_load(item.get("kubernetesManifest", ""))
                                hostnames = route_manifest.get("spec", {}).get("hostnames", [])
                                if hostnames:
                                    redirect_url = f"http://{hostnames[0]}"
                                    break
            except Exception as e:
                logging.error(f"Error extracting redirect URL: {e}")

        return {
            "exists": True,
            "phase": phase,
            "message": message,
            "redirect_url": redirect_url
        }
    except ApiException as e:
        if e.status == 404:
            return {"exists": False, "phase": "None"}
        return {"exists": False, "error": str(e)}
    except Exception as e:
        return {"exists": False, "error": str(e)}

@app.post("/init-cluster")
async def initialize_cluster(namespace: str = "default"):
    """Trigger cluster initialization by applying the init CR."""
    try:
        # Load the init manifest path
        init_manifest_path = os.path.join(BLUEPRINTS_DIR, "cluster_init", "app-store-cluster-init.yaml")
        
        if not os.path.exists(init_manifest_path):
            return JSONResponse({"ok": False, "error": f"Init manifest not found at {init_manifest_path}"}, status_code=404)

        # Use the standard blueprint application method which handles multi-doc YAMLs and CRDs
        # We pass the namespace to override the default 'default' namespace if requested
        result = apply_blueprint_with_namespace(init_manifest_path, namespace=namespace)
        return JSONResponse({"ok": True, "message": "Cluster initialization started.", "result": result})
            
    except Exception as e:
        logger.error(f"Error in initialize_cluster: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/init-logs")
async def stream_init_logs():
    """Stream operator logs filtered for the initialization process."""
    def log_generator():
        try:
            load_kube_config()
            core_api = client.CoreV1Api()
            
            # Find the operator pod
            # Use the chart name from the label (default is weka-app-store-operator-chart)
            retry_count = 0
            max_retries = 5
            pods = None
            
            while retry_count < max_retries:
                # Search in all namespaces for the operator pod
                pods = core_api.list_pod_for_all_namespaces(
                    label_selector="app.kubernetes.io/name=weka-app-store-operator-chart"
                )
                
                if not pods.items:
                    # Fallback to a broader search across all namespaces if the exact chart name changed
                    pods = core_api.list_pod_for_all_namespaces(
                        label_selector="app.kubernetes.io/managed-by=Helm"
                    )
                    # Filter for something that looks like our operator if many things are managed by Helm
                    pods.items = [p for p in pods.items if "operator" in p.metadata.name]

                if pods.items:
                    break
                
                retry_count += 1
                if retry_count < max_retries:
                    yield f"data: Error: Operator pod not found in any namespace (attempt {retry_count}/{max_retries})\n\n"
                    time.sleep(2)
            
            if not pods or not pods.items:
                yield "data: Error: Operator pod not found after 5 attempts. Cancelling initialization monitoring.\n\n"
                return

            pod_obj = pods.items[0]
            pod_name = pod_obj.metadata.name
            pod_namespace = pod_obj.metadata.namespace
            
            # Stream logs using the kubernetes client
            # We'll use read_namespaced_pod_log with follow=True for streaming
            try:
                # Use follow=True and _preload_content=False to get a streaming response
                log_stream = core_api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=pod_namespace,
                    follow=True,
                    _preload_content=False,
                    tail_lines=100
                )
                
                # Filter for relevant logs: 
                # - logs containing "app-store-cluster-init"
                # - logs from handle_appstack_deployment
                # - logs from HelmOperator
                for line_bytes in log_stream:
                    line = line_bytes.decode('utf-8', errors='replace')
                    if any(x in line for x in ["app-store-cluster-init", "Deploying", "Installing", "Ready", "Failed", "Helm"]):
                        yield f"data: {line}\n\n"
                    
                    # Small sleep to be nice to the event loop if needed, 
                    # but log_stream is a generator that blocks/waits
            except Exception as stream_err:
                logger.error(f"Error streaming logs from pod {pod_name}: {stream_err}")
                yield f"data: Error streaming logs: {str(stream_err)}\n\n"
            finally:
                if 'log_stream' in locals():
                    try:
                        log_stream.release_conn()
                    except Exception:
                        pass
                    
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(log_generator(), media_type="text/event-stream")

# Uvicorn entry point: `uvicorn webapp.main:app --reload`



# On-demand git sync endpoint to update manifests from the configured repo
@app.post("/sync")
async def sync_blueprints(request: Request):
    try:
        # Optional bearer token check
        token_required = os.environ.get("SYNC_TOKEN", "")
        if token_required:
            auth = request.headers.get("Authorization") or ""
            if not auth.startswith("Bearer ") or auth.split(" ", 1)[1] != token_required:
                return JSONResponse({"ok": False, "error": "Unauthorized"}, status_code=401)

        # Config from env with sensible defaults
        repo = os.environ.get("GIT_SYNC_REPO")
        branch = os.environ.get("GIT_SYNC_BRANCH", "main")
        # Default root to a writable path to avoid permission issues when running as non-root
        root = os.environ.get("GIT_SYNC_ROOT", "/tmp/git-sync-root")
        link = os.environ.get("GIT_SYNC_LINK", "../../manifests")
        if not repo:
            return JSONResponse({"ok": False, "error": "GIT_SYNC_REPO not set"}, status_code=400)

        # Ensure the root directory exists and is writable
        try:
            os.makedirs(root, exist_ok=True)
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Failed to create GIT_SYNC_ROOT '{root}': {e}"}, status_code=500)

        # Ensure git-sync binary exists (download if needed)
        bin_path = await _ensure_git_sync_binary()

        # Serialize concurrent syncs using a lock file in /tmp (always writable)
        lock_path = "/tmp/git-sync.lock"
        rc = 1
        stdout = ""
        stderr = ""
        try:
            import fcntl  # type: ignore
            with open(lock_path, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                cmd = [
                    bin_path,
                    f"--repo={repo}",
                    f"--branch={branch}",
                    f"--root={root}",
                    "--one-time",
                    f"--link={link}",
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except Exception as e:
            # If locking fails for any reason, still attempt a one-off run
            cmd = [
                bin_path,
                f"--repo={repo}",
                f"--branch={branch}",
                f"--root={root}",
                "--one-time",
                f"--link={link}",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr

        return JSONResponse({"ok": rc == 0, "stdout": stdout, "stderr": stderr}, status_code=200 if rc == 0 else 500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _ensure_git_sync_binary() -> str:
    # Allow overriding binary path or download URL/version via environment
    explicit_path = os.environ.get("GIT_SYNC_BIN") or os.environ.get("GIT_SYNC_PATH")
    def _is_valid_binary(path: str) -> bool:
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            return False
        try:
            # Prefer --version; many releases support it. Fallback to --help if needed.
            proc = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
            if proc.returncode == 0:
                return True
            # Some older builds may not support --version; try --help
            proc = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=5)
            return proc.returncode == 0
        except OSError as oe:
            # e.g., Exec format error (wrong architecture)
            return False
        except Exception:
            return False

    if explicit_path and _is_valid_binary(explicit_path):
        return explicit_path

    # Use a cached location inside the container FS and common install paths, but validate
    path_candidates = ["/tmp/git-sync", "/usr/local/bin/git-sync", "/usr/bin/git-sync"]
    for p in path_candidates:
        if _is_valid_binary(p):
            return p

    # Determine asset name based on linux arch
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"):
        asset = "git-sync-linux-amd64"
        legacy_asset = "git-sync-amd64"
    elif arch in ("aarch64", "arm64"):
        asset = "git-sync-linux-arm64"
        legacy_asset = "git-sync-arm64"
    else:
        # Fallback to amd64
        asset = "git-sync-linux-amd64"
        legacy_asset = "git-sync-amd64"

    # Build list of URLs to try, in order
    dest = "/tmp/git-sync"
    urls_to_try = []

    # If an explicit URL is provided, try it first
    override_url = os.environ.get("GIT_SYNC_DOWNLOAD_URL")
    if override_url:
        urls_to_try.append(override_url)

    # Version candidates: env provided version first, then known good fallbacks
    env_version = os.environ.get("GIT_SYNC_VERSION")
    version_candidates = [v for v in [env_version] if v]
    # Add a few known release versions as fallbacks (keep list short to avoid long delays)
    version_candidates += [
        "v4.4.0",
        "v4.3.0",
        "v4.2.3",
        "v4.2.0",
        "v4.1.0",
    ]

    for ver in version_candidates:
        urls_to_try.append(f"https://github.com/kubernetes/git-sync/releases/download/{ver}/{asset}")
        # Also try legacy asset naming used by some older releases
        urls_to_try.append(f"https://github.com/kubernetes/git-sync/releases/download/{ver}/{legacy_asset}")

    errors: List[str] = []

    # Try to download from the list
    for url in urls_to_try:
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=30) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            os.chmod(dest, 0o755)
            # Validate the downloaded binary to guard against HTML error pages, etc.
            if _is_valid_binary(dest):
                return dest
            else:
                errors.append(f"{url} -> downloaded but failed to execute")
        except Exception as e:
            errors.append(f"{url} -> {e}")
            continue

    # If we reach here, all attempts failed
    detail = "; ".join(errors[-5:])  # keep message concise
    raise RuntimeError(
        "Failed to obtain a working git-sync binary. Tried URLs: " + ", ".join(urls_to_try[:6]) +
        (" ..." if len(urls_to_try) > 6 else "") +
        f". Last errors: {detail}"
    )


def get_blueprint_components(file_path: str) -> List[str]:
    """Extract component names from a WARRP AppStore blueprint YAML.
    Returns a list of component display names for progress UI.
    """
    if not os.path.isabs(file_path):
        file_path = os.path.join(PROJECT_ROOT, file_path)
    items: List[str] = []
    try:
        with open(file_path, 'r') as f:
            docs = list(yaml.safe_load_all(f))
        data = next((d for d in docs if isinstance(d, dict) and d.get("kind") == "WekaAppStore"), {})
        spec = (data or {}).get('spec', {})
        comps = ((spec.get('appStack') or {}).get('components')) or []
        for idx, c in enumerate(comps):
            if not isinstance(c, dict):
                continue
            # Mirror the operator's filter (handle_appstack_deployment): disabled components
            # are never deployed and never report status, so don't render a section for them.
            if not c.get('enabled', True):
                continue
            name = str(c.get('name') or f"component-{idx+1}")
            items.append(name)
    except Exception:
        # fallback single item
        items = ["Submitting blueprint"]
    return items


def _extract_wekaappstore_name(rendered: str) -> Optional[str]:
    """Return metadata.name of the first WekaAppStore doc in rendered multi-doc YAML, or None."""
    try:
        for doc in yaml.safe_load_all(rendered):
            if isinstance(doc, dict) and doc.get("kind") == "WekaAppStore":
                return ((doc.get("metadata") or {}).get("name"))
    except Exception:
        return None
    return None


# Support both query-style and path-style app selection
# e.g. /deploy-stream?app_name=openfold and /deploy-stream/openfold
@app.get("/deploy-stream/{app_name}")
@app.get("/deploy-stream")
async def deploy_stream(
    request: Request,
    app_name: Optional[str] = None,
    variables: str = "{}",
):
    """Server-Sent Events stream that emits deployment progress for a blueprint.

    Emits JSON events:
    - {type: 'init', items: [..component names..]}
    - {type: 'progress', currentIndex: i, name: 'component-name'}
    - {type: 'complete', ok: true, result: {...}}
    - {type: 'error', message: '...'}
    """
    def sse_event(payload: Dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    async def event_generator():
        # Parse variables JSON
        try:
            user_vars = json.loads(variables)
        except (json.JSONDecodeError, ValueError):
            yield sse_event({"type": "error", "message": "Invalid variables JSON"})
            return

        # Locate blueprint via dynamic discovery
        yaml_path = find_blueprint(app_name)
        if not yaml_path:
            yield sse_event({"type": "error", "message": "Unknown app"})
            return

        # Extract namespace from variables dict; default to "default" if absent or empty
        namespace = str(user_vars.get("namespace", "default") or "default").strip() or "default"

        # Required-field validation (namespace-preserving apps are exempt)
        if app_name not in NAMESPACE_PRESERVING_APPS:
            if not os.path.isabs(yaml_path):
                schema_path = os.path.join(PROJECT_ROOT, yaml_path)
            else:
                schema_path = yaml_path
            try:
                with open(schema_path, "r") as _sf:
                    raw_schema_text = _sf.read()
            except Exception:
                raw_schema_text = ""
            schema = parse_x_variables(raw_schema_text)
            for var_name, meta in schema.items():
                raw_val = str(user_vars.get(var_name, "") or "")
                if meta.get("required") and not raw_val.strip():
                    yield sse_event({"type": "error", "message": f"Required variable missing: {var_name}"})
                    return
                fmt_err = _validate_variable_value(var_name, meta, raw_val)
                if fmt_err:
                    yield sse_event({"type": "error", "message": f"Invalid value — {fmt_err}"})
                    return

        # Initial items
        items = get_blueprint_components(yaml_path)
        yield sse_event({
            "type": "init",
            "items": items,
            # compatibility for UIs expecting a message string
            "message": f"Initializing {len(items)} components"
        })

        # Render the blueprint, apply it, then track REAL per-component progress by polling
        # the WekaAppStore .status that the operator patches as each component finishes.
        try:
            # Load and render blueprint YAML as Jinja2 template with provided variables
            if not os.path.isabs(yaml_path):
                bp_path = os.path.join(PROJECT_ROOT, yaml_path)
            else:
                bp_path = yaml_path

            with open(bp_path, "r") as f:
                raw_tpl = f.read()

            # Use custom Jinja2 delimiters for variables to avoid clashing with Argo's {{ }} placeholders
            # Only change variable delimiters; keep block/comment delimiters default
            env = Environment(variable_start_string="[[", variable_end_string="]]")
            template = env.from_string(raw_tpl)

            # Derive server-side vars before render so the blueprint can reference
            # them via [[ ]] even though they are not in x-variables.
            # Guard: only derive when the source keys are present so blueprints that
            # do not use these vars (e.g. cluster-init) are unaffected.
            if user_vars.get("quay_username") or user_vars.get("quay_password"):
                user_vars["quay_dockerconfigjson"] = build_quay_dockerconfigjson(
                    user_vars.get("quay_username", ""),
                    user_vars.get("quay_password", ""),
                )
            if user_vars.get("join_ip_ports"):
                user_vars.update(split_endpoints(user_vars.get("join_ip_ports", "")))

            rendered = template.render(**user_vars)

            try:
                docs = [d for d in yaml.safe_load_all(rendered) if isinstance(d, dict) and d.get("kind")]
            except yaml.YAMLError as ye:
                yield sse_event({"type": "error", "message": f"Rendered blueprint is not valid YAML: {ye}"})
                return

            # Stamp the submitted variables onto the WekaAppStore CR so the blueprint
            # page can later show them in read-only fields and offer Uninstall.
            cr_name = None
            cr_namespace = None
            for d in docs:
                if isinstance(d, dict) and d.get("kind") == "WekaAppStore":
                    md = d.setdefault("metadata", {})
                    anns = md.setdefault("annotations", {})
                    anns["warp.io/gui-variables"] = json.dumps(_safe_gui_variables(user_vars), separators=(",", ":"))
                    cr_name = cr_name or md.get("name")
                    cr_namespace = cr_namespace or md.get("namespace")

            # Apply manifest with namespace overrides using the rendered documents
            logger.info(
                "Deploy-stream start: app=%s namespace=%s blueprint=%s items=%d",
                app_name, namespace, bp_path, len(items)
            )
            ns_for_apply = "" if app_name in NAMESPACE_PRESERVING_APPS else namespace
            result = apply_blueprint_documents_with_namespace(docs, namespace=ns_for_apply)

            # Non-appStack blueprints (no WekaAppStore CR in the manifest) have no
            # per-component operator status to poll — report submission complete immediately.
            # Namespace-preserving apps (app-store-install, cluster-init) ARE multi-component
            # appStacks and DO reach the poll loop below; only the namespace-override
            # suppression at line 3075 is skipped for them (ns_for_apply=""), not the poll.
            if not cr_name:
                yield sse_event({"type": "complete", "ok": True, "result": result, "message": "Deployment complete"})
                return

            # Poll the operator's real componentStatus until the stack reaches a terminal phase.
            # For namespace-preserving apps the CR is applied to its blueprint-declared namespace,
            # not the user-supplied one; use the CR's actual namespace for polling when available.
            poll_namespace = cr_namespace if (app_name in NAMESPACE_PRESERVING_APPS and cr_namespace) else namespace
            load_kube_config()
            custom_api = client.CustomObjectsApi()
            emitted: Dict[str, str] = {}
            deadline = time.time() + parse_deploy_timeout(raw_tpl)
            while True:
                if await request.is_disconnected():
                    return
                # Keepalive comment: without periodic bytes an ingress/proxy will
                # drop an idle SSE connection between component phase changes,
                # surfacing as "Stream connection error" in the browser.
                yield ": ping\n\n"
                try:
                    cr = await asyncio.to_thread(
                        custom_api.get_namespaced_custom_object,
                        group="warp.io", version="v1alpha1",
                        namespace=poll_namespace, plural="wekaappstores", name=cr_name,
                    )
                except ApiException:
                    cr = None
                status = (cr or {}).get("status", {}) or {}
                comp_statuses = status.get("componentStatus", []) or []
                # Emit a per-component event whenever a component's phase changes.
                for comp in comp_statuses:
                    cname = comp.get("name")
                    cphase = comp.get("phase")
                    if cname and emitted.get(cname) != cphase:
                        emitted[cname] = cphase
                        yield sse_event({
                            "type": "component",
                            "name": cname,
                            "phase": cphase,
                            "message": _redact_secrets(comp.get("message", ""), user_vars),
                        })
                phase = status.get("appStackPhase")
                if phase == "Ready":
                    yield sse_event({"type": "complete", "ok": True, "result": result, "message": "Deployment complete"})
                    return
                if phase == "Failed":
                    failed = next((c for c in comp_statuses if c.get("phase") in ("Failed", "Error")), None)
                    raw_msg = (f"{failed.get('name')}: {failed.get('message', 'failed')}"
                               if failed else "Deployment failed")
                    msg = _redact_secrets(raw_msg, user_vars)
                    yield sse_event({"type": "complete", "ok": False, "result": result, "message": msg})
                    return
                if time.time() > deadline:
                    yield sse_event({"type": "error", "message": "Timed out waiting for components to become ready"})
                    return
                await asyncio.sleep(2)
        except FileNotFoundError as e:
            yield sse_event({"type": "error", "message": str(e)})
        except ApiException as e:
            yield sse_event({
                "type": "error",
                "message": f"Kubernetes API error: {e.reason}",
                "status": e.status
            })
        except Exception as e:
            yield sse_event({"type": "error", "message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
