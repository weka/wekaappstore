from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict, Any, List
import os
import yaml
import base64
import json
import time
import copy

from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException

app = FastAPI(title="WEKA App Store")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

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
    global _config_loaded
    if _config_loaded:
        return
    try:
        # Try in-cluster first
        config.load_incluster_config()
        _config_loaded = True
    except Exception:
        # Fall back to local kubeconfig
        try:
            config.load_kube_config()
            _config_loaded = True
        except Exception as e:
            raise RuntimeError(f"Unable to load Kubernetes configuration: {e}")


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
            data = yaml.safe_load(f)
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
    return templates.TemplateResponse("index.html", {
        "request": request,
        "status": status,
        "logo_b64": LOGO_B64,
    })


@app.get("/status")
async def status_endpoint():
    return JSONResponse(get_cluster_status())


@app.get("/blueprint/{name}", response_class=HTMLResponse)
async def blueprint_detail(request: Request, name: str):
    app_map = {
        "oss-rag": os.path.join("Production Deployments", "oss-rag-stack.yaml"),
        "nvidia-rag": os.path.join("Production Deployments", "nvidia-rag.yaml"),
        "nvidia-vss": os.path.join("Production Deployments", "nvidia-vss.yaml"),
        "cluster-init": os.path.join("Production Deployments", "app-store-cluster-init.yaml"),
    }
    yaml_path = app_map.get(name)
    if not yaml_path:
        return RedirectResponse(url="/", status_code=302)
    status = get_cluster_status()
    reqs = infer_requirements_from_yaml(yaml_path)
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
    return templates.TemplateResponse("blueprint.html", {
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
        "oss-rag": os.path.join("Production Deployments", "oss-rag-stack.yaml"),
        "nvidia-rag": os.path.join("Production Deployments", "nvidia-rag.yaml"),
        "nvidia-vss": os.path.join("Production Deployments", "nvidia-vss.yaml"),
        "cluster-init": os.path.join("Production Deployments", "app-store-cluster-init.yaml"),
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
    """Delete the app-store-cluster-init WekaAppStore CR from the cluster."""
    try:
        load_kube_config()
        co_api = client.CustomObjectsApi()
        # Attempt delete in default namespace; ignore 404
        try:
            resp = co_api.delete_namespaced_custom_object(
                group="warp.io",
                version="v1alpha1",
                namespace="default",
                plural="wekaappstores",
                name="app-store-cluster-init",
            )
            return JSONResponse({"ok": True, "result": resp})
        except ApiException as ae:
            if ae.status == 404:
                return JSONResponse({"ok": True, "result": "Not present"})
            raise
    except ApiException as e:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {e.reason}", "status": e.status}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# Convenience root for health
@app.get("/healthz")
async def healthz():
    return {"ok": True}


# Uvicorn entry point: `uvicorn webapp.main:app --reload`



def get_blueprint_components(file_path: str) -> List[str]:
    """Extract component names from a WARRP AppStore blueprint YAML.
    Returns a list of component display names for progress UI.
    """
    if not os.path.isabs(file_path):
        file_path = os.path.join(PROJECT_ROOT, file_path)
    items: List[str] = []
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
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


@app.get("/deploy-stream")
async def deploy_stream(request: Request, app_name: str, namespace: str = "default"):
    """Server-Sent Events stream that emits deployment progress for a blueprint.

    Emits JSON events:
    - {type: 'init', items: [..component names..]}
    - {type: 'progress', currentIndex: i, name: 'component-name'}
    - {type: 'complete', ok: true, result: {...}}
    - {type: 'error', message: '...'}
    """
    app_map = {
        "oss-rag": os.path.join("Production Deployments", "oss-rag-stack.yaml"),
        "nvidia-rag": os.path.join("Production Deployments", "nvidia-rag.yaml"),
        "nvidia-vss": os.path.join("Production Deployments", "nvidia-vss.yaml"),
        "cluster-init": os.path.join("Production Deployments", "app-store-cluster-init.yaml"),
    }
    yaml_path = app_map.get(app_name)

    # For cluster-init, preserve namespaces defined in the YAML (do not override)
    if app_name == "cluster-init":
        namespace = ""

    def sse_event(payload: Dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    async def event_generator():
        # Validate app
        if not yaml_path:
            yield sse_event({"type": "error", "message": "Unknown app"})
            return
        # Initial items
        items = get_blueprint_components(yaml_path)
        yield sse_event({"type": "init", "items": items})

        # Stream a simple progress over the items while we submit the blueprint
        try:
            for i, item in enumerate(items):
                # If client disconnected, stop
                if await request.is_disconnected():
                    return
                yield sse_event({"type": "progress", "currentIndex": i, "name": item})
                # Small delay to allow UI to render progression
                time.sleep(0.15)

            # Apply manifest with namespace overrides
            result = apply_blueprint_with_namespace(yaml_path, namespace=namespace)
            yield sse_event({"type": "complete", "ok": True, "result": result})
        except FileNotFoundError as e:
            yield sse_event({"type": "error", "message": str(e)})
        except ApiException as e:
            yield sse_event({"type": "error", "message": f"Kubernetes API error: {e.reason}", "status": e.status})
        except Exception as e:
            yield sse_event({"type": "error", "message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
