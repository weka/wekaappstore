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
import uuid

from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException
from jinja2 import Environment
from webapp.inspection import collect_cluster_inspection, collect_weka_inspection, flatten_cluster_status
from webapp.planning import (
    ApplyGateway,
    LocalPlanningSessionStore,
    PlanCompilationError,
    PlanningSessionNotFoundError,
    PlanningSessionService,
    build_stage_error,
    compile_plan_to_wekaappstore,
    compile_plan_to_yaml,
    derive_fit_findings_from_snapshot,
    merge_inspection_results,
    supported_family_catalog,
    validate_structured_plan,
)
from webapp.planning.inspection_tools import PlanningInspectionTools

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
PLANNING_SESSIONS_DIR = os.path.join(PROJECT_ROOT, ".planning-sessions")

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
    """Collect bounded cluster status through the shared inspection service."""
    snapshot = collect_cluster_inspection(load_kube_config=load_kube_config)
    return flatten_cluster_status(snapshot)


def _new_correlation_id() -> str:
    return f"plan-{uuid.uuid4().hex[:12]}"


def build_planning_inspection_snapshot(*, correlation_id: Optional[str] = None) -> Dict[str, Any]:
    """Build one merged planner-facing inspection snapshot from bounded cluster and WEKA services."""
    correlation_id = correlation_id or _new_correlation_id()
    cluster_result = PLANNING_INSPECTION_TOOLS.inspect(
        "cluster_snapshot", correlation_id=correlation_id
    )
    weka_result = PLANNING_INSPECTION_TOOLS.inspect(
        "weka_storage", correlation_id=correlation_id
    )
    snapshot = merge_inspection_results(
        [cluster_result["result"], weka_result["result"]],
        correlation_id=correlation_id,
    )
    snapshot["sources"] = {
        "cluster": cluster_result["audit"],
        "weka": weka_result["audit"],
    }
    return snapshot


def build_fit_findings_from_inspection(
    *,
    inspection_snapshot: Optional[Dict[str, Any]] = None,
    required_domains: Optional[List[str]] = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    snapshot = inspection_snapshot or build_planning_inspection_snapshot(
        correlation_id=correlation_id
    )
    return derive_fit_findings_from_snapshot(snapshot, required_domains=required_domains)


PLANNING_INSPECTION_TOOLS = PlanningInspectionTools(
    cluster_collector=lambda: collect_cluster_inspection(load_kube_config=load_kube_config),
    weka_collector=lambda: collect_weka_inspection(load_kube_config=load_kube_config),
)


def build_default_planning_draft(
    *,
    session: Any,
    request_summary: str,
    conversation: List[Dict[str, Any]],
    family_match: Any,
    inspection_snapshot: Dict[str, Any],
    fit_findings: Dict[str, Any],
) -> Dict[str, Any]:
    requested_namespace = None
    for turn in reversed(conversation):
        if turn.get("role") != "user":
            continue
        message = str(turn.get("message") or "")
        for token in message.replace(",", " ").split():
            if token.endswith("-namespace"):
                requested_namespace = token.strip(".!?")
                break
        if requested_namespace:
            break

    family_catalog = supported_family_catalog()
    family_metadata = family_catalog.get(family_match.family) if family_match.family else None
    display_name = (
        family_metadata.display_name if family_metadata is not None else (family_match.family or "this blueprint")
    )
    namespace_value = requested_namespace or "ai-platform"
    unresolved_questions: list[dict[str, Any]] = []
    assistant_message = (
        f"I mapped this request to {display_name} and drafted a backend-owned planning session."
    )
    draft_summary = (
        f"Drafted a {display_name} session using the latest inspection snapshot and current conversation state."
    )
    if requested_namespace is None:
        unresolved_questions.append(
            {
                "question": "Which namespace should receive this deployment?",
                "field_path": "namespace_strategy.target_namespace",
                "blocking": True,
                "install_critical": True,
            }
        )
        assistant_message = (
            f"I mapped this request to {display_name}, but I still need the target namespace before the draft can validate."
        )
        draft_summary = "Waiting for the deployment namespace before the draft can validate."

    return {
        "assistant_message": assistant_message,
        "draft_summary": draft_summary,
        "plan": {
            "request_summary": request_summary,
            "blueprint_family": family_match.family,
            "namespace_strategy": {
                "mode": "explicit",
                "target_namespace": namespace_value if not unresolved_questions else None,
            },
            "components": [],
            "prerequisites": [
                "Review the generated plan before preview or apply.",
            ],
            "fit_findings": fit_findings,
            "unresolved_questions": unresolved_questions,
            "reasoning_summary": draft_summary,
        },
        "follow_ups": unresolved_questions,
        "inspection_snapshot": inspection_snapshot,
    }


def create_planning_session_service() -> PlanningSessionService:
    return PlanningSessionService(
        session_store=LocalPlanningSessionStore(PLANNING_SESSIONS_DIR),
        inspection_tools=PLANNING_INSPECTION_TOOLS,
        planner=build_default_planning_draft,
    )


def get_planning_session_service() -> PlanningSessionService:
    service = getattr(app.state, "planning_session_service", None)
    if service is None:
        service = create_planning_session_service()
        app.state.planning_session_service = service
    return service


def _planning_session_not_found(session_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Planning session '{session_id}' was not found.")


def _planning_session_context(session: Any) -> Dict[str, Any]:
    latest_revision = session.latest_revision
    draft_plan = latest_revision.structured_plan if latest_revision is not None else None
    return {
        "session": session,
        "draft_revision": latest_revision,
        "draft_plan": draft_plan,
        "unanswered_follow_ups": session.unanswered_follow_ups,
        "transcript": session.turns,
        "session_history": {
            "restarted_from_session_id": session.restarted_from_session_id,
            "replacement_session_id": session.replacement_session_id,
            "restart_count": session.restart_count,
        },
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


def build_structured_plan_preview(
    payload: Dict[str, Any],
    *,
    inspection_snapshot: Optional[Dict[str, Any]] = None,
    required_domains: Optional[List[str]] = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    correlation_id = correlation_id or _new_correlation_id()
    working_payload = copy.deepcopy(payload)
    resolved_fit_findings = None
    if inspection_snapshot is not None:
        try:
            resolved_fit_findings = build_fit_findings_from_inspection(
                inspection_snapshot=inspection_snapshot,
                required_domains=required_domains,
                correlation_id=correlation_id,
            )
        except Exception as exc:
            return {
                "valid": False,
                "errors": [build_stage_error("inspection", exc, correlation_id=correlation_id)],
                "warnings": [],
                "compiled_document": None,
                "yaml": None,
                "correlation_id": correlation_id,
                "failure_stage": "inspection",
                "inspection_snapshot": inspection_snapshot,
                "fit_findings": None,
            }
        working_payload["fit_findings"] = resolved_fit_findings

    validation = validate_structured_plan(working_payload)
    if not validation.valid or validation.plan is None:
        return {
            "valid": False,
            "errors": [error.to_dict() for error in validation.errors],
            "warnings": [warning.to_dict() for warning in validation.warnings],
            "compiled_document": None,
            "yaml": None,
            "correlation_id": correlation_id,
            "failure_stage": "validation",
            "inspection_snapshot": inspection_snapshot,
            "fit_findings": resolved_fit_findings,
        }

    try:
        compiled_document = compile_plan_to_wekaappstore(validation.plan)
        rendered_yaml = compile_plan_to_yaml(validation.plan)
    except PlanCompilationError as exc:
        return {
            "valid": False,
            "errors": [build_stage_error("yaml_generation", exc, correlation_id=correlation_id)],
            "warnings": [warning.to_dict() for warning in validation.warnings],
            "compiled_document": None,
            "yaml": None,
            "correlation_id": correlation_id,
            "failure_stage": "yaml_generation",
            "inspection_snapshot": inspection_snapshot,
            "fit_findings": resolved_fit_findings,
        }
    return {
        "valid": True,
        "errors": [],
        "warnings": [warning.to_dict() for warning in validation.warnings],
        "compiled_document": compiled_document,
        "yaml": rendered_yaml,
        "correlation_id": correlation_id,
        "failure_stage": None,
        "inspection_snapshot": inspection_snapshot,
        "fit_findings": resolved_fit_findings,
    }


def execute_structured_plan_apply(
    payload: Dict[str, Any],
    *,
    namespace_override: str = "",
    inspection_snapshot: Optional[Dict[str, Any]] = None,
    required_domains: Optional[List[str]] = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    correlation_id = correlation_id or _new_correlation_id()
    preview = build_structured_plan_preview(
        payload,
        inspection_snapshot=inspection_snapshot,
        required_domains=required_domains,
        correlation_id=correlation_id,
    )
    if not preview["valid"] or preview["compiled_document"] is None or preview["yaml"] is None:
        return {
            "ok": False,
            "validation": {
                "valid": False,
                "errors": preview["errors"],
                "warnings": preview["warnings"],
            },
            "compiled_document": None,
            "yaml": None,
            "result": None,
            "correlation_id": correlation_id,
            "failure_stage": preview.get("failure_stage") or "validation",
            "inspection_snapshot": preview.get("inspection_snapshot"),
        }

    document_namespace = preview["compiled_document"]["metadata"].get("namespace", "")
    effective_namespace = namespace_override or document_namespace
    try:
        result = PLANNING_APPLY_GATEWAY.apply_content(preview["yaml"], effective_namespace)
    except Exception as exc:
        return {
            "ok": False,
            "validation": {
                "valid": True,
                "errors": [build_stage_error("apply_handoff", exc, correlation_id=correlation_id)],
                "warnings": preview["warnings"],
            },
            "compiled_document": preview["compiled_document"],
            "yaml": preview["yaml"],
            "result": None,
            "correlation_id": correlation_id,
            "failure_stage": "apply_handoff",
            "inspection_snapshot": preview.get("inspection_snapshot"),
        }

    return {
        "ok": True,
        "validation": {
            "valid": True,
            "errors": [],
            "warnings": preview["warnings"],
        },
        "compiled_document": preview["compiled_document"],
        "yaml": preview["yaml"],
        "result": result,
        "correlation_id": correlation_id,
        "failure_stage": None,
        "inspection_snapshot": preview.get("inspection_snapshot"),
    }


def apply_structured_plan(
    payload: Dict[str, Any],
    *,
    namespace_override: str = "",
    inspection_snapshot: Optional[Dict[str, Any]] = None,
    required_domains: Optional[List[str]] = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    response = execute_structured_plan_apply(
        payload,
        namespace_override=namespace_override,
        inspection_snapshot=inspection_snapshot,
        required_domains=required_domains,
        correlation_id=correlation_id,
    )
    if not response["ok"]:
        errors = response["validation"]["errors"]
        if errors:
            raise PlanCompilationError(errors[0]["message"])
        raise PlanCompilationError("structured plan contains blocking validation issues")
    return response


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
        },
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    auth = await asyncio.to_thread(get_auth_status)
    status = await asyncio.to_thread(get_cluster_status)
    # Use detected namespace if available, else default
    detected_ns = (auth.get("details", {}) or {}).get("namespace") if isinstance(auth, dict) else None
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "request": request,
            "auth": auth,
            "status": status,
            "detected_namespace": detected_ns or "default",
            "logo_b64": LOGO_B64,
        },
    )


@app.post("/planning/sessions")
async def create_planning_session(
    request: Request,
    request_text: str = Form(...),
):
    service = get_planning_session_service()
    transition = await asyncio.to_thread(
        service.start_session,
        request_text,
        metadata={"source": "web"},
    )
    wants_json = "application/json" in request.headers.get("accept", "").lower()
    if wants_json:
        return JSONResponse(
            {
                "ok": True,
                "session_id": transition.session.session_id,
                "status": transition.session.status,
                "redirect_url": f"/planning/sessions/{transition.session.session_id}",
            }
        )
    return RedirectResponse(
        url=f"/planning/sessions/{transition.session.session_id}",
        status_code=303,
    )


@app.get("/planning/sessions/{session_id}", response_class=HTMLResponse)
async def planning_session_page(request: Request, session_id: str):
    service = get_planning_session_service()
    try:
        session = await asyncio.to_thread(service._session_store.load_session, session_id)
    except PlanningSessionNotFoundError as exc:
        raise _planning_session_not_found(session_id) from exc

    context = _planning_session_context(session)
    return templates.TemplateResponse(
        request,
        "planning_session.html",
        {
            "request": request,
            "logo_b64": LOGO_B64,
            **context,
        },
    )


@app.post("/planning/sessions/{session_id}/message")
async def planning_session_message(
    request: Request,
    session_id: str,
    message: str = Form(...),
):
    service = get_planning_session_service()
    try:
        transition = await asyncio.to_thread(
            service.process_turn,
            session_id,
            message=message,
            metadata={"source": "web"},
        )
    except PlanningSessionNotFoundError as exc:
        raise _planning_session_not_found(session_id) from exc

    wants_json = "application/json" in request.headers.get("accept", "").lower()
    if wants_json:
        return JSONResponse(
            {
                "ok": True,
                "session_id": transition.session.session_id,
                "status": transition.session.status,
                "assistant_message": transition.assistant_message,
            }
        )
    return RedirectResponse(url=f"/planning/sessions/{session_id}", status_code=303)


@app.post("/planning/sessions/{session_id}/follow-ups/{question_id}")
async def answer_planning_follow_up(
    request: Request,
    session_id: str,
    question_id: str,
    answer: str = Form(...),
):
    service = get_planning_session_service()
    try:
        transition = await asyncio.to_thread(
            service.answer_follow_up,
            session_id,
            question_id=question_id,
            answer=answer,
            metadata={"source": "web"},
        )
    except PlanningSessionNotFoundError as exc:
        raise _planning_session_not_found(session_id) from exc

    wants_json = "application/json" in request.headers.get("accept", "").lower()
    if wants_json:
        return JSONResponse(
            {
                "ok": True,
                "session_id": transition.session.session_id,
                "status": transition.session.status,
                "assistant_message": transition.assistant_message,
            }
        )
    return RedirectResponse(url=f"/planning/sessions/{session_id}", status_code=303)


@app.post("/planning/sessions/{session_id}/restart")
async def restart_planning_session(
    request: Request,
    session_id: str,
):
    service = get_planning_session_service()
    try:
        replacement = await asyncio.to_thread(
            service._session_store.restart_session,
            session_id,
            metadata={"source": "web", "action": "restart"},
        )
    except PlanningSessionNotFoundError as exc:
        raise _planning_session_not_found(session_id) from exc

    wants_json = "application/json" in request.headers.get("accept", "").lower()
    if wants_json:
        return JSONResponse(
            {
                "ok": True,
                "session_id": replacement.session_id,
                "status": replacement.status,
                "redirect_url": f"/planning/sessions/{replacement.session_id}",
                "restarted_from_session_id": replacement.restarted_from_session_id,
            }
        )
    return RedirectResponse(url=f"/planning/sessions/{replacement.session_id}", status_code=303)


@app.post("/planning/sessions/{session_id}/abandon")
async def abandon_planning_session(
    request: Request,
    session_id: str,
):
    service = get_planning_session_service()
    try:
        session = await asyncio.to_thread(
            service._session_store.abandon_session,
            session_id,
            metadata={"source": "web", "action": "abandon"},
        )
    except PlanningSessionNotFoundError as exc:
        raise _planning_session_not_found(session_id) from exc

    wants_json = "application/json" in request.headers.get("accept", "").lower()
    if wants_json:
        return JSONResponse(
            {
                "ok": True,
                "session_id": session.session_id,
                "status": session.status,
            }
        )
    return RedirectResponse(url=f"/planning/sessions/{session_id}", status_code=303)


@app.get("/planning/sessions/{session_id}/events")
async def planning_session_events(session_id: str):
    service = get_planning_session_service()
    try:
        session = await asyncio.to_thread(service._session_store.load_session, session_id)
    except PlanningSessionNotFoundError as exc:
        raise _planning_session_not_found(session_id) from exc

    context = _planning_session_context(session)
    payload = {
        "type": "session-state",
        "session_id": session.session_id,
        "status": session.status,
        "draft_status": (
            None if context["draft_revision"] is None else context["draft_revision"].status
        ),
        "pending_follow_up_ids": [
            follow_up.question_id for follow_up in context["unanswered_follow_ups"]
        ],
        "turn_count": len(context["transcript"]),
    }

    async def event_generator():
        yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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

    # For cluster-init, use provided namespace but default to "default" if empty
    if app_name == "cluster-init" and not namespace:
        namespace = "default"

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
                logger.info("Deploy-stream progress: %s (%d/%d)", item, i+1, len(items))
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
            
            # For cluster-init, we use the provided namespace (defaulting to 'default')
            render_ns = namespace or "default"
            
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
            logger.info(
                "Deploy-stream start: app=%s namespace=%s blueprint=%s items=%d",
                app_name, namespace, bp_path, len(items)
            )
            ns_for_apply = "" if app_name == "cluster-init" else namespace
            result = apply_blueprint_content_with_namespace(rendered, namespace=ns_for_apply)
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
