from fastapi import FastAPI, Request, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict, Any, List
import os
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

from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException
from jinja2 import Environment

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WEKA App Store")

class ClusterInitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Paths to exclude from blocking
        exempt_paths = ["/healthz", "/readyz", "/static", "/welcome", "/init-cluster", "/init-logs", "/cluster-status", "/cluster-info"]
        
        # Allow exempt paths
        if any(request.url.path.startswith(p) for p in exempt_paths):
            return await call_next(request)
        
        # Check initialization status
        initialized = await is_cluster_initialized()
        
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

async def is_cluster_initialized():
    try:
        load_kube_config()
        custom_api = client.CustomObjectsApi()
        cr = custom_api.get_namespaced_custom_object(
            group="warp.io",
            version="v1alpha1",
            namespace="default",
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
_default_git_sync_link = os.getenv("GIT_SYNC_LINK", "manifests")
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
_candidates.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "manifests"))

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

# Load logo as base64 once for reuse in templates
LOGO_B64 = None
try:
    logo_path = os.path.join(BASE_DIR, "app_store_logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as _lf:
            LOGO_B64 = base64.b64encode(_lf.read()).decode('ascii')
except Exception:
    LOGO_B64 = None

# Helper: load kube config
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
    """Collects cluster status: CPU/GPU node counts, GPU Operator, K8s version, CRD installed, default StorageClass.

    Also computes used vs free resources based on pod resource requests scheduled on Ready nodes.
    """
    load_kube_config()

    core = client.CoreV1Api()
    ext = client.ApiextensionsV1Api()
    storage_api = client.StorageV1Api()

    cpu_nodes = 0
    gpu_nodes = 0
    cpu_milli_total = 0
    gpu_devices_total = 0

    ready_node_names = set()

    def _cpu_to_millicores(val: Any) -> int:
        try:
            if val is None:
                return 0
            s = str(val).strip()
            if s.endswith('m'):
                return int(float(s[:-1]))
            # handle plain integers/floats like '4' or '4.5'
            return int(float(s) * 1000)
        except Exception:
            return 0

    try:
        nodes = core.list_node().items
        for n in nodes:
            # Only count Ready nodes
            conditions = {c.type: c.status for c in (n.status.conditions or [])}
            is_ready = conditions.get("Ready") == "True"
            if not is_ready:
                continue
            ready_node_names.add(n.metadata.name)
            alloc = n.status.allocatable or {}
            # Tally CPU millicores from all Ready nodes
            cpu_milli_total += _cpu_to_millicores(alloc.get('cpu'))
            # GPU detection: allocatable nvidia.com/gpu > 0
            gpu_alloc = alloc.get("nvidia.com/gpu")
            try:
                gpu_count = int(gpu_alloc) if gpu_alloc is not None else 0
            except ValueError:
                # Sometimes it's a string number
                try:
                    gpu_count = int(str(gpu_alloc))
                except Exception:
                    gpu_count = 0
            gpu_devices_total += max(gpu_count, 0)
            if gpu_count and gpu_count > 0:
                gpu_nodes += 1
            else:
                cpu_nodes += 1
    except Exception as e:
        # If we can't list nodes, surface minimal info
        return {
            "cpu_nodes": None,
            "gpu_nodes": None,
            "gpu_operator_installed": None,
            "k8s_version": None,
            "app_store_crd_installed": None,
            "app_store_cluster_init_present": None,
            "app_store_crs": [],
            "default_storage_class": None,
            "default_storage_class_details": None,
            "error": f"Error fetching node data: {e}"
        }

    # Compute resource usage (based on requests) on Ready nodes only
    cpu_milli_used = 0
    gpu_devices_used = 0
    try:
        pods = core.list_pod_for_all_namespaces().items
        for p in pods:
            # Skip completed/failed pods
            phase = (p.status.phase or "").lower()
            if phase in ("succeeded", "failed"):
                continue
            node_name = getattr(p.spec, 'node_name', None)
            if node_name not in ready_node_names:
                continue
            # Sum container requests
            containers = list(p.spec.containers or [])
            init_containers = list(getattr(p.spec, 'init_containers', []) or [])
            for c in containers + init_containers:
                res = getattr(c, 'resources', None)
                reqs = getattr(res, 'requests', None) if res else None
                if not reqs:
                    continue
                cpu_val = reqs.get('cpu') if isinstance(reqs, dict) else None
                gpu_val = reqs.get('nvidia.com/gpu') if isinstance(reqs, dict) else None
                cpu_milli_used += _cpu_to_millicores(cpu_val)
                try:
                    if gpu_val is not None:
                        gpu_devices_used += int(str(gpu_val))
                except Exception:
                    pass
    except Exception:
        # If listing pods fails, leave used as 0
        pass

    # Kubernetes version
    try:
        version_api = client.VersionApi()
        version = version_api.get_code().git_version
    except Exception:
        version = None

    # NVIDIA GPU Operator detection (more accurate):
    # 1) Check presence of ClusterPolicy custom resources (group=nvidia.com, version=v1, plural=clusterpolicies)
    # 2) Fallback: check for the NVIDIA device plugin DaemonSet readiness in common namespaces
    gpu_operator_installed = None
    try:
        co_api = client.CustomObjectsApi()
        try:
            cps = co_api.list_cluster_custom_object(group="nvidia.com", version="v1", plural="clusterpolicies")
            items = (cps or {}).get("items", [])
            if isinstance(items, list) and len(items) > 0:
                gpu_operator_installed = True
            else:
                gpu_operator_installed = False
        except ApiException as ae:
            if ae.status == 404:
                gpu_operator_installed = False
            else:
                gpu_operator_installed = None
        # If not clearly installed, try DaemonSet probe
        if gpu_operator_installed is False:
            try:
                apps_api = client.AppsV1Api()
                candidate_namespaces = ["gpu-operator-resources", "nvidia-device-plugin", "kube-system", "gpu-operator"]
                found_ready = False
                for ns in candidate_namespaces:
                    try:
                        dss = apps_api.list_namespaced_daemon_set(ns).items
                    except ApiException as ae2:
                        if ae2.status in (403, 404):
                            continue
                        raise
                    for ds in dss:
                        name = ds.metadata.name or ""
                        if "nvidia-device-plugin" in name:
                            status_ds = ds.status or {}
                            desired = status_ds.desired_number_scheduled or 0
                            ready = status_ds.number_ready or 0
                            if desired and ready and ready > 0:
                                found_ready = True
                                break
                    if found_ready:
                        break
                if found_ready:
                    gpu_operator_installed = True
            except Exception:
                pass
    except Exception:
        gpu_operator_installed = None

    # WekaAppStore CRD + CRs detection
    app_store_crd_installed = None
    cluster_init_present = None
    applied_crs: List[Dict[str, str]] = []
    try:
        # Correct CRD name as defined in warrp-crd.yaml
        ext.read_custom_resource_definition("wekaappstores.warp.io")
        app_store_crd_installed = True
        try:
            co_api = client.CustomObjectsApi()
            crs = co_api.list_cluster_custom_object(group="warp.io", version="v1alpha1", plural="wekaappstores")
            items = (crs or {}).get("items", [])
            names = []
            for it in items:
                meta = (it or {}).get("metadata", {}) or {}
                ns = meta.get("namespace") or "default"
                nm = meta.get("name") or ""
                if nm:
                    names.append({"namespace": ns, "name": nm})
            applied_crs = names
            cluster_init_present = any(x.get("name") == "app-store-cluster-init" for x in names)
        except ApiException as ae2:
            if ae2.status == 404:
                # CRD exists but no CRs
                applied_crs = []
                cluster_init_present = False
            else:
                applied_crs = []
                cluster_init_present = None
        except Exception:
            applied_crs = []
            cluster_init_present = None
    except ApiException as ae:
        if ae.status == 404:
            app_store_crd_installed = False
            cluster_init_present = False
        else:
            app_store_crd_installed = None
    except Exception:
        app_store_crd_installed = None

    # Default StorageClass detection
    default_sc = None
    default_sc_details = None
    try:
        scs = storage_api.list_storage_class().items
        for sc in scs:
            ann = (sc.metadata.annotations or {})
            is_default = ann.get("storageclass.kubernetes.io/is-default-class") or ann.get("storageclass.beta.kubernetes.io/is-default-class")
            if str(is_default).lower() == "true":
                default_sc = sc.metadata.name
                # Collect detailed fields for UI
                default_sc_details = {
                    "name": sc.metadata.name,
                    "provisioner": sc.provisioner,
                    "parameters": dict(sc.parameters or {}),
                    "reclaimPolicy": getattr(sc, "reclaim_policy", None) or (getattr(sc, "reclaimPolicy", None)),
                    "volumeBindingMode": getattr(sc, "volume_binding_mode", None) or (getattr(sc, "volumeBindingMode", None)),
                    "allowVolumeExpansion": getattr(sc, "allow_volume_expansion", None) or (getattr(sc, "allowVolumeExpansion", None)),
                    "annotations": dict(sc.metadata.annotations or {}),
                }
                break
    except Exception:
        default_sc = None
        default_sc_details = None

    # Convert CPU millicores to cores (rounded to 2 decimals)
    cpu_cores_total = round(cpu_milli_total / 1000.0, 2)
    cpu_cores_used = round(cpu_milli_used / 1000.0, 2)
    cpu_cores_free = round(max(cpu_milli_total - cpu_milli_used, 0) / 1000.0, 2)

    gpu_devices_used = int(gpu_devices_used)
    gpu_devices_free = max(int(gpu_devices_total) - gpu_devices_used, 0)

    return {
        "cpu_nodes": cpu_nodes,
        "gpu_nodes": gpu_nodes,
        "cpu_cores_total": cpu_cores_total,
        "cpu_cores_used": cpu_cores_used,
        "cpu_cores_free": cpu_cores_free,
        "gpu_devices_total": gpu_devices_total,
        "gpu_devices_used": gpu_devices_used,
        "gpu_devices_free": gpu_devices_free,
        "gpu_operator_installed": gpu_operator_installed,
        "k8s_version": version,
        "app_store_crd_installed": app_store_crd_installed,
        "app_store_cluster_init_present": cluster_init_present,
        "app_store_crs": [f"{x['namespace']}/{x['name']}" for x in applied_crs],
        "default_storage_class": default_sc,
        "default_storage_class_details": default_sc_details,
    }


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


def apply_blueprint_with_namespace(file_path: str, namespace: str) -> Dict[str, Any]:
    """Load a blueprint YAML and override namespaces before applying.

    Behavior:
    - If a non-empty `namespace` is provided, set metadata.namespace on namespaced root object(s),
      and override spec.appStack.components[].targetNamespace to that namespace.
    - If `namespace` is empty, preserve namespaces defined in the YAML. For WARRP CRs lacking
      metadata.namespace, default to "default" so the API path is valid.
    - Applies each document, using CustomObjectsApi for CRs to avoid missing generated client classes.
    """
    load_kube_config()

    if not os.path.isabs(file_path):
        file_path = os.path.join(PROJECT_ROOT, file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    applied_kinds: list[str] = []
    k8s_client = client.ApiClient()
    co_api = client.CustomObjectsApi(k8s_client)

    with open(file_path, "r") as f:
        docs = list(yaml.safe_load_all(f))

    for doc in docs:
        if not isinstance(doc, dict):
            # Skip non-dict docs
            continue

        # Ensure metadata exists
        md = doc.setdefault("metadata", {}) if isinstance(doc, dict) else {}

        # If namespace override provided, set it on the document (namespaced kinds)
        if isinstance(md, dict) and namespace:
            md["namespace"] = namespace

        # Override targetNamespace per component only when user provided namespace
        try:
            spec = doc.get("spec") or {}
            app_stack = spec.get("appStack") or {}
            components = app_stack.get("components") or []
            for comp in components:
                if isinstance(comp, dict) and namespace and "targetNamespace" in comp:
                    comp["targetNamespace"] = namespace
        except Exception:
            # Non-fatal if structure differs
            pass

        # Decide how to apply: use CustomObjectsApi for CRs without generated clients
        api_version = str(doc.get("apiVersion") or "")
        kind = str(doc.get("kind") or "")

        # Resolve effective namespace for this document
        # Priority: provided `namespace` (if non-empty) -> document metadata.namespace -> None
        doc_ns = namespace if namespace else (md.get("namespace") if isinstance(md, dict) else None)

        if api_version.startswith("warrp.io/") or kind == "WarrpAppStore" or api_version.startswith("warp.io/") or kind == "WekaAppStore":
            # Handle WARRP/WARP CRs via CustomObjectsApi to avoid missing generated client classes
            try:
                group, version = api_version.split("/", 1)
            except ValueError:
                # Fallback defaults per known kinds
                if kind == "WekaAppStore":
                    group, version = "warp.io", "v1alpha1"
                else:
                    group, version = "warrp.io", (api_version or "v1alpha1")
            # Explicit plural mapping for known CRDs; fallback to naive pluralization
            lower_kind = (kind or "CustomResource").lower()
            if lower_kind == "warrpappstore":
                plural = "warrpappstores"
            elif lower_kind == "wekaappstore":
                plural = "wekaappstores"
            else:
                plural = lower_kind + "s"
            name = (md or {}).get("name")
            if not name:
                raise ValueError("CustomResource document missing metadata.name")
            # For namespaced CRs, ensure we have a valid namespace. Default to 'default' if absent.
            cr_ns = doc_ns or "default"
            # Make sure target namespace exists
            ensure_namespace_exists(cr_ns)
            body = _with_last_applied_annotation(doc)
            try:
                # Try create first
                co_api.create_namespaced_custom_object(group=group, version=version, namespace=cr_ns, plural=plural, body=body)
            except ApiException as ae:
                if ae.status == 409:
                    # Already exists -> patch to update without requiring resourceVersion
                    co_api.patch_namespaced_custom_object(group=group, version=version, namespace=cr_ns, plural=plural, name=name, body=body)
                else:
                    raise
            applied_kinds.append(kind or "CustomResource")
            continue

        # For non-CR documents: ensure namespace exists if we have one to use
        if doc_ns:
            ensure_namespace_exists(doc_ns)

        # Fallback: use utils for built-in or known kinds
        doc_with_ann = _with_last_applied_annotation(doc)
        created = utils.create_from_dict(k8s_client, data=doc_with_ann, namespace=(doc_ns or None), verbose=False)
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
        for obj, _ in tuples:
            try:
                applied_kinds.append(obj.kind)
            except Exception:
                applied_kinds.append(str(obj))

    return {"applied": applied_kinds}


def apply_blueprint_content_with_namespace(content: str, namespace: str) -> Dict[str, Any]:
    """Apply a blueprint provided as YAML string, with namespace override logic.

    Mirrors apply_blueprint_with_namespace, but reads from provided content instead of a file.
    """
    load_kube_config()

    applied_kinds: list[str] = []
    k8s_client = client.ApiClient()
    co_api = client.CustomObjectsApi(k8s_client)

    docs = list(yaml.safe_load_all(content))

    for doc in docs:
        if not isinstance(doc, dict):
            continue

        md = doc.setdefault("metadata", {}) if isinstance(doc, dict) else {}

        if isinstance(md, dict) and namespace:
            md["namespace"] = namespace

        try:
            spec = doc.get("spec") or {}
            app_stack = spec.get("appStack") or {}
            components = app_stack.get("components") or []
            for comp in components:
                if isinstance(comp, dict) and namespace and "targetNamespace" in comp:
                    comp["targetNamespace"] = namespace
        except Exception:
            pass

        api_version = str(doc.get("apiVersion") or "")
        kind = str(doc.get("kind") or "")

        doc_ns = namespace if namespace else (md.get("namespace") if isinstance(md, dict) else None)

        if api_version.startswith("warrp.io/") or kind == "WarrpAppStore" or api_version.startswith("warp.io/") or kind == "WekaAppStore":
            try:
                group, version = api_version.split("/", 1)
            except ValueError:
                if kind == "WekaAppStore":
                    group, version = "warp.io", "v1alpha1"
                else:
                    group, version = "warrp.io", (api_version or "v1alpha1")
            lower_kind = (kind or "CustomResource").lower()
            if lower_kind == "warrpappstore":
                plural = "warrpappstores"
            elif lower_kind == "wekaappstore":
                plural = "wekaappstores"
            else:
                plural = lower_kind + "s"
            name = (md or {}).get("name")
            if not name:
                raise ValueError("CustomResource document missing metadata.name")
            cr_ns = doc_ns or "default"
            ensure_namespace_exists(cr_ns)
            body = _with_last_applied_annotation(doc)
            try:
                co_api.create_namespaced_custom_object(group=group, version=version, namespace=cr_ns, plural=plural, body=body)
            except ApiException as ae:
                if ae.status == 409:
                    co_api.patch_namespaced_custom_object(group=group, version=version, namespace=cr_ns, plural=plural, name=name, body=body)
                else:
                    raise
            applied_kinds.append(kind or "CustomResource")
            continue

        if doc_ns:
            ensure_namespace_exists(doc_ns)

        doc_with_ann = _with_last_applied_annotation(doc)
        created = utils.create_from_dict(k8s_client, data=doc_with_ann, namespace=(doc_ns or None), verbose=False)

        def _to_tuples(created_any):
            tuples: list[tuple] = []
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
        for obj, _ in tuples:
            try:
                applied_kinds.append(obj.kind)
            except Exception:
                applied_kinds.append(str(obj))

    return {"applied": applied_kinds}


def install_warrp_crd() -> Dict[str, Any]:
    crd_path = os.path.join(PROJECT_ROOT, "warrp-crd.yaml")
    return apply_yaml(crd_path, namespace=None)


def infer_requirements_from_yaml(file_path: str) -> Dict[str, int]:
    """Infer minimal CPU/GPU node requirements from a blueprint YAML.
    Heuristic:
    - If NVIDIA GPU Operator (or component naming suggesting GPU) is enabled -> require >=1 GPU node.
    - Always require >=1 CPU node.
    """
    if not os.path.isabs(file_path):
        file_path = os.path.join(PROJECT_ROOT, file_path)
    req = {"cpu_nodes": 1, "gpu_nodes": 0}
    try:
        with open(file_path, 'r') as f:
            docs = list(yaml.safe_load_all(f))
        data = docs[0] if docs else {}
        spec = (data or {}).get('spec', {})
        comps = ((spec.get('appStack') or {}).get('components')) or []
        for c in comps:
            name = (c or {}).get('name', '') or ''
            desc = (c or {}).get('description', '') or ''
            helm = (c or {}).get('helmChart', {}) or {}
            chart_name = str(helm.get('name') or helm.get('repository') or '').lower()
            if any(s in name.lower() for s in ['nvidia', 'gpu-operator', 'gpu']) or 'nvidia' in chart_name:
                req['gpu_nodes'] = max(req['gpu_nodes'], 1)
    except Exception:
        # If parsing fails, fall back to defaults
        pass
    return req


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    status = get_cluster_status()
    auth = get_auth_status()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "status": status,
        "auth": auth,
        "logo_b64": LOGO_B64,
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    auth = get_auth_status()
    status = get_cluster_status()
    # Use detected namespace if available, else default
    detected_ns = (auth.get("details", {}) or {}).get("namespace") if isinstance(auth, dict) else None
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "auth": auth,
        "status": status,
        "detected_namespace": detected_ns or "default",
        "logo_b64": LOGO_B64,
    })


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


@app.post("/api/secret/huggingface")
async def save_huggingface_key(api_key: str = Form(...), namespace: str = Form("default")):
    try:
        result = create_or_update_secret(
            name="hf-api-key",
            namespace=namespace.strip() or "default",
            string_data={"HF_API_KEY": api_key},
        )
        return JSONResponse({"ok": True, "secret_name": result["name"], "namespace": result["namespace"], "action": result["action"]})
    except ApiException as ae:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


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


@app.post("/api/secret/nvidia")
async def save_nvidia_key(api_key: str = Form(...), namespace: str = Form("default")):
    try:
        result = create_or_update_secret(
            name="nvidia-api-key",
            namespace=namespace.strip() or "default",
            string_data={"NVIDIA_API_KEY": api_key},
        )
        return JSONResponse({"ok": True, "secret_name": result["name"], "namespace": result["namespace"], "action": result["action"]})
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


@app.get("/blueprint/{name}", response_class=HTMLResponse)
async def blueprint_detail(request: Request, name: str):
    # Map known apps to their YAML manifests if available. Unknown names are allowed to render
    # a template page (per-blueprint) even if there is no YAML yet.
    app_map = {
        "oss-rag": os.path.join(BLUEPRINTS_DIR, "oss-rag", "oss-rag-stack.yaml"),
        "nvidia-rag": os.path.join("Production Deployments", "nvidia-rag.yaml"),
        "nvidia-vss": os.path.join("Production Deployments", "nvidia-vss.yaml"),
        "cluster-init": os.path.join(BLUEPRINTS_DIR, "cluster_init", "app-store-cluster-init.yaml"),
        # Wire OpenFold to its blueprint manifest folder so Deploy works
        "openfold": os.path.join(BLUEPRINTS_DIR, "openfold-protein", "openfold-stack.yaml"),
        # "ai-agent-enterprise-research": os.path.join("Production Deployments", "ai-agent-enterprise-research.yaml"),
    }
    yaml_path = app_map.get(name)
    status = get_cluster_status()
    # If there is no YAML mapped for this blueprint, use safe defaults
    reqs = infer_requirements_from_yaml(yaml_path) if yaml_path else {"cpu_nodes": 1, "gpu_nodes": 0}
    meets = {
        "cpu": None if status.get('cpu_nodes') is None else status.get('cpu_nodes', 0) >= reqs.get('cpu_nodes', 0),
        "gpu": None if status.get('gpu_nodes') is None else status.get('gpu_nodes', 0) >= reqs.get('gpu_nodes', 0),
    }
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
    # Choose a specific template if present (except for RAG pages which keep generic)
    preferred = f"blueprint_{name}.html"
    use_template = "blueprint.html"
    try:
        if name not in {"oss-rag", "nvidia-rag"}:
            if os.path.exists(os.path.join(TEMPLATES_DIR, preferred)):
                use_template = preferred
    except Exception:
        use_template = "blueprint.html"

    return templates.TemplateResponse(use_template, {
        "request": request,
        "name": name,
        "yaml_path": yaml_path,
        "status": status,
        "requirements": reqs,
        "meets": meets,
        "oss_img_b64": oss_img_b64,
        "logo_b64": LOGO_B64,
    })


@app.post("/deploy")
async def deploy(app_name: str = Form(...), namespace: str = Form("default")):
    # Map app names to yaml paths
    app_map = {
        "oss-rag": os.path.join(BLUEPRINTS_DIR, "oss-rag", "oss-rag-stack.yaml"),
        "nvidia-rag": os.path.join("Production Deployments", "nvidia-rag.yaml"),
        "nvidia-vss": os.path.join("Production Deployments", "nvidia-vss.yaml"),
        "cluster-init": os.path.join(BLUEPRINTS_DIR, "cluster_init", "app-store-cluster-init.yaml"),
        # OpenFold deployment mapping
        "openfold": os.path.join(BLUEPRINTS_DIR, "openfold-protein", "openfold-stack.yaml"),
    }
    yaml_path = app_map.get(app_name)
    if not yaml_path:
        return JSONResponse({"ok": False, "error": "Unknown app"}, status_code=400)

    # For cluster-init, preserve namespaces defined in the YAML (do not override)
    effective_ns = "" if app_name == "cluster-init" else namespace

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
    return templates.TemplateResponse("welcome.html", {
        "request": request,
        "logo_b64": LOGO_B64,
        "title": "Welcome to WEKA App Store"
    })

@app.get("/cluster-status")
async def get_cluster_status():
    """Get the current initialization status."""
    try:
        load_kube_config()
        custom_api = client.CustomObjectsApi()
        cr = custom_api.get_namespaced_custom_object(
            group="warp.io",
            version="v1alpha1",
            namespace="default",
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
                if cond.get("type") in ["Error", "Failed"] or cond.get("status") == "False" and cond.get("type") in ["Ready", "Initialized"]:
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
                        # Use the first document which should be the WekaAppStore CR
                        manifest = docs[0] if docs else {}
                        # The manifest is a list of appStack components or a single CR. 
                        # In this case it's a single WekaAppStore CR.
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
async def initialize_cluster():
    """Trigger cluster initialization by applying the init CR."""
    try:
        # Load the init manifest path
        init_manifest_path = os.path.join(BLUEPRINTS_DIR, "cluster_init", "app-store-cluster-init.yaml")
        
        if not os.path.exists(init_manifest_path):
            return JSONResponse({"ok": False, "error": f"Init manifest not found at {init_manifest_path}"}, status_code=404)

        # Use the standard blueprint application method which handles multi-doc YAMLs and CRDs
        result = apply_blueprint_with_namespace(init_manifest_path, namespace="")
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
        link = os.environ.get("GIT_SYNC_LINK", "manifests")
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
        data = docs[0] if docs else {}
        spec = (data or {}).get('spec', {})
        comps = ((spec.get('appStack') or {}).get('components')) or []
        for idx, c in enumerate(comps):
            if not isinstance(c, dict):
                continue
            name = str(c.get('name') or f"component-{idx+1}")
            items.append(name)
    except Exception:
        # fallback single item
        items = ["Submitting blueprint"]
    return items


# Support both query-style and path-style app selection
# e.g. /deploy-stream?app_name=openfold and /deploy-stream/openfold
@app.get("/deploy-stream/{app_name}")
@app.get("/deploy-stream")
async def deploy_stream(
    request: Request,
    app_name: str,
    namespace: str = "default",
    storage_class: Optional[str] = None,
    vllm_chat_model: Optional[str] = None,
    vllm_embed_model: Optional[str] = None,
    # Backward-compat param: previously used single vllm_model
    vllm_model: Optional[str] = None,
    # New OpenFold-specific configurable parameters
    weka_cluster_filesystem: Optional[str] = None,
    openfold_storage_capacity: Optional[str] = None,
    deployment_name: Optional[str] = None,
):
    """Server-Sent Events stream that emits deployment progress for a blueprint.

    Emits JSON events:
    - {type: 'init', items: [..component names..]}
    - {type: 'progress', currentIndex: i, name: 'component-name'}
    - {type: 'complete', ok: true, result: {...}}
    - {type: 'error', message: '...'}
    """
    app_map = {
        "oss-rag": os.path.join(BLUEPRINTS_DIR, "oss-rag", "oss-rag-stack.yaml"),
        "nvidia-rag": os.path.join("Production Deployments", "nvidia-rag.yaml"),
        "nvidia-vss": os.path.join("Production Deployments", "nvidia-vss.yaml"),
        "cluster-init": os.path.join(BLUEPRINTS_DIR, "cluster_init", "app-store-cluster-init.yaml"),
        # OpenFold deployment mapping
        "openfold": os.path.join(BLUEPRINTS_DIR, "openfold-protein", "openfold-stack.yaml"),
    }
    yaml_path = app_map.get(app_name)

    # For cluster-init, preserve namespaces defined in the YAML (do not override)
    if app_name == "cluster-init":
        namespace = ""

    def sse_event(payload: Dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    async def event_generator():
        # Normalize incoming parameters: trim whitespace and convert empty strings to None
        def _norm(val: Optional[str]) -> Optional[str]:
            if isinstance(val, str):
                v = val.strip()
                return v if v else None
            return val

        norm_storage_class = _norm(storage_class)
        norm_chat_model = _norm(vllm_chat_model)
        norm_embed_model = _norm(vllm_embed_model)
        norm_legacy_model = _norm(vllm_model)
        norm_weka_fs = _norm(weka_cluster_filesystem)
        norm_of_capacity = _norm(openfold_storage_capacity)
        norm_deploy_name = _norm(deployment_name)

        # Validate app
        if not yaml_path:
            yield sse_event({"type": "error", "message": "Unknown app"})
            return
        # Initial items
        items = get_blueprint_components(yaml_path)
        yield sse_event({
            "type": "init",
            "items": items,
            # compatibility for UIs expecting a message string
            "message": f"Initializing {len(items)} components"
        })

        # Stream a simple progress over the items while we submit the blueprint
        try:
            for i, item in enumerate(items):
                # If client disconnected, stop
                if await request.is_disconnected():
                    return
                yield sse_event({
                    "type": "progress",
                    "currentIndex": i,
                    "name": item,
                    "message": f"Applying {item} ({i+1}/{len(items)})"
                })
                # Small delay to allow UI to render progression
                try:
                    await asyncio.sleep(0.15)
                except Exception:
                    # Fallback (shouldn't block event loop often)
                    time.sleep(0.1)

            # Load and render blueprint YAML as Jinja2 template with provided variables
            if not os.path.isabs(yaml_path):
                bp_path = os.path.join(PROJECT_ROOT, yaml_path)
            else:
                bp_path = yaml_path

            with open(bp_path, 'r') as f:
                raw_tpl = f.read()

            # Use custom Jinja2 delimiters for variables to avoid clashing with Argo's {{ }} placeholders
            # Only change variable delimiters; keep block/comment delimiters default
            env = Environment(variable_start_string='[[', variable_end_string=']]')
            template = env.from_string(raw_tpl)
            # For cluster-init, avoid injecting a default namespace into the template to preserve file-defined namespaces
            render_ns = ("" if app_name == "cluster-init" else (namespace or "default"))
            # Backward compatibility: if old vllm_model is provided but new chat model is empty,
            # treat it as the chat model.
            chat_model_var = norm_chat_model or norm_legacy_model or None
            rendered = template.render(
                namespace=render_ns,
                storage_class=norm_storage_class,
                # New variables
                vllm_chat_model=chat_model_var,
                vllm_embed_model=norm_embed_model,
                # Legacy variable kept for existing templates
                vllm_model=chat_model_var,
                # OpenFold-specific variables
                weka_cluster_filesystem=norm_weka_fs,
                openfold_storage_capacity=norm_of_capacity,
                deployment_name=norm_deploy_name,
            )

            # Apply manifest with namespace overrides using rendered content
            result = apply_blueprint_content_with_namespace(rendered, namespace=namespace)
            yield sse_event({
                "type": "complete",
                "ok": True,
                "result": result,
                "message": "Deployment complete"
            })
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
