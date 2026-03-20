from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import logging
import os
from typing import Any, Callable, Dict, Iterable, Optional

import yaml
from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException


logger = logging.getLogger(__name__)

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
    "PodSecurityPolicy",
    "RuntimeClass",
    "CSIDriver",
    "CSINode",
    "CertificateSigningRequest",
}


def _load_kube_config() -> None:
    config.load_kube_config()


def _ensure_namespace_exists(namespace: Optional[str]) -> None:
    if not namespace:
        return

    _load_kube_config()
    core = client.CoreV1Api()
    try:
        core.read_namespace(name=namespace)
        return
    except ApiException as exc:
        if exc.status != 404:
            raise

    body = client.V1Namespace(metadata=client.V1ObjectMeta(name=namespace))
    try:
        core.create_namespace(body=body)
    except ApiException as exc:
        if exc.status != 409:
            raise


def _with_last_applied_annotation(doc: Dict[str, Any]) -> Dict[str, Any]:
    try:
        doc_copy = copy.deepcopy(doc)
        metadata = doc_copy.setdefault("metadata", {}) if isinstance(doc_copy, dict) else {}
        annotations = metadata.setdefault("annotations", {}) if isinstance(metadata, dict) else {}
        annotations.pop("kubectl.kubernetes.io/last-applied-configuration", None)
        serialized = json.dumps(doc_copy, separators=(",", ":"), sort_keys=True)

        real_metadata = doc.setdefault("metadata", {}) if isinstance(doc, dict) else {}
        real_annotations = (
            real_metadata.setdefault("annotations", {}) if isinstance(real_metadata, dict) else {}
        )
        real_annotations["kubectl.kubernetes.io/last-applied-configuration"] = serialized
    except Exception:
        return doc

    return doc


def _api_version(doc: Dict[str, Any]) -> str:
    return str((doc or {}).get("apiVersion") or "")


def _kind(doc: Dict[str, Any]) -> str:
    return str((doc or {}).get("kind") or "")


def _metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    metadata = (doc or {}).get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _crd_scope_for(group: str, plural: str) -> str:
    try:
        _load_kube_config()
        extension_api = client.ApiextensionsV1Api()
        definition = extension_api.read_custom_resource_definition(f"{plural}.{group}")
        return str(definition.spec.scope or "Namespaced")
    except Exception:
        return "Namespaced"


def _is_cluster_scoped(doc: Dict[str, Any]) -> bool:
    kind = _kind(doc)
    if kind in CLUSTER_SCOPED_KINDS:
        return True

    api_version = _api_version(doc)
    if "/" not in api_version:
        return False

    group = api_version.split("/", 1)[0]
    plural = kind.lower() + "s"
    if kind.lower() == "warrpappstore":
        plural = "warrpappstores"
    elif kind.lower() == "wekaappstore":
        plural = "wekaappstores"

    return _crd_scope_for(group, plural) == "Cluster"


def _doc_id(doc: Dict[str, Any]) -> str:
    metadata = _metadata(doc)
    return (
        f"{doc.get('apiVersion')} {doc.get('kind')} "
        f"{metadata.get('namespace', '<cluster>')}/{metadata.get('name')}"
    )


def _to_tuples(created_any: Any) -> list[tuple[Any, Any]]:
    if created_any is None:
        return []

    if isinstance(created_any, list):
        tuples: list[tuple[Any, Any]] = []
        for item in created_any:
            if isinstance(item, tuple) and len(item) == 2:
                tuples.append(item)
            else:
                tuples.append((item, None))
        return tuples

    if isinstance(created_any, tuple) and len(created_any) == 2:
        return [created_any]

    return [(created_any, None)]


@dataclass(frozen=True)
class ApplyGatewayDependencies:
    load_kube_config: Callable[[], None] = _load_kube_config
    ensure_namespace_exists: Callable[[Optional[str]], None] = _ensure_namespace_exists
    is_cluster_scoped: Callable[[Dict[str, Any]], bool] = _is_cluster_scoped
    crd_scope_for: Callable[[str, str], str] = _crd_scope_for
    with_last_applied_annotation: Callable[[Dict[str, Any]], Dict[str, Any]] = (
        _with_last_applied_annotation
    )
    api_client_factory: Callable[[], Any] = client.ApiClient
    custom_objects_api_factory: Callable[[Any], Any] = client.CustomObjectsApi
    create_from_dict: Callable[..., Any] = utils.create_from_dict
    file_exists: Callable[[str], bool] = os.path.exists
    logger: logging.Logger = logger


@dataclass(frozen=True)
class ApplyGateway:
    project_root: Optional[str] = None
    dependencies: ApplyGatewayDependencies = ApplyGatewayDependencies()

    def apply_documents(self, documents: Iterable[Any], namespace_override: str) -> Dict[str, Any]:
        return apply_yaml_documents_with_namespace(
            documents,
            namespace_override,
            dependencies=self.dependencies,
        )

    def apply_content(self, content: str, namespace_override: str) -> Dict[str, Any]:
        return apply_yaml_content_with_namespace(
            content,
            namespace_override,
            dependencies=self.dependencies,
        )

    def apply_file(self, file_path: str, namespace_override: str) -> Dict[str, Any]:
        return apply_yaml_file_with_namespace(
            file_path,
            namespace_override,
            project_root=self.project_root,
            dependencies=self.dependencies,
        )


def _iter_documents(documents: Iterable[Any]) -> Iterable[Dict[str, Any]]:
    for doc in documents:
        if isinstance(doc, dict):
            yield copy.deepcopy(doc)


def _normalize_document_namespace(
    doc: Dict[str, Any],
    namespace_override: str,
    *,
    is_cluster_scoped_resource: bool,
) -> Dict[str, Any]:
    metadata = doc.setdefault("metadata", {})
    if isinstance(metadata, dict) and namespace_override and not is_cluster_scoped_resource:
        metadata["namespace"] = namespace_override

    try:
        components = (((doc.get("spec") or {}).get("appStack") or {}).get("components") or [])
        for component in components:
            if isinstance(component, dict) and namespace_override and "targetNamespace" in component:
                component["targetNamespace"] = namespace_override
    except Exception:
        pass

    return doc


def apply_yaml_documents_with_namespace(
    documents: Iterable[Any],
    namespace_override: str,
    *,
    dependencies: Optional[ApplyGatewayDependencies] = None,
) -> Dict[str, Any]:
    deps = dependencies or ApplyGatewayDependencies()
    deps.load_kube_config()

    applied_kinds: list[str] = []
    k8s_client = deps.api_client_factory()
    custom_objects_api = deps.custom_objects_api_factory(k8s_client)

    for index, doc in enumerate(_iter_documents(documents)):
        is_cluster_scoped_resource = deps.is_cluster_scoped(doc)
        _normalize_document_namespace(
            doc,
            namespace_override,
            is_cluster_scoped_resource=is_cluster_scoped_resource,
        )

        metadata = _metadata(doc)
        api_version = _api_version(doc)
        kind = _kind(doc)
        if is_cluster_scoped_resource:
            document_namespace = None
        else:
            document_namespace = (
                namespace_override if namespace_override else metadata.get("namespace")
            )

        deps.logger.info("APPLY doc[%d]: %s", index, _doc_id(doc))

        try:
            if (
                api_version.startswith("warrp.io/")
                or kind == "WarrpAppStore"
                or api_version.startswith("warp.io/")
                or kind == "WekaAppStore"
            ):
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

                if not is_cluster_scoped_resource and deps.crd_scope_for(group, plural) == "Cluster":
                    is_cluster_scoped_resource = True
                    document_namespace = None

                name = metadata.get("name")
                if not name:
                    raise ValueError("CustomResource document missing metadata.name")

                body = deps.with_last_applied_annotation(doc)
                if is_cluster_scoped_resource:
                    try:
                        custom_objects_api.create_cluster_custom_object(
                            group=group,
                            version=version,
                            plural=plural,
                            body=body,
                        )
                    except ApiException as exc:
                        if exc.status != 409:
                            raise
                        custom_objects_api.patch_cluster_custom_object(
                            group=group,
                            version=version,
                            plural=plural,
                            name=name,
                            body=body,
                            _content_type="application/merge-patch+json",
                        )
                else:
                    resource_namespace = document_namespace or "default"
                    deps.ensure_namespace_exists(resource_namespace)
                    try:
                        custom_objects_api.create_namespaced_custom_object(
                            group=group,
                            version=version,
                            namespace=resource_namespace,
                            plural=plural,
                            body=body,
                        )
                    except ApiException as exc:
                        if exc.status != 409:
                            raise
                        custom_objects_api.patch_namespaced_custom_object(
                            group=group,
                            version=version,
                            namespace=resource_namespace,
                            plural=plural,
                            name=name,
                            body=body,
                            _content_type="application/merge-patch+json",
                        )

                applied_kinds.append(kind or "CustomResource")
                continue

            if document_namespace:
                deps.ensure_namespace_exists(document_namespace)

            body = deps.with_last_applied_annotation(doc)
            create_result = deps.create_from_dict(
                k8s_client,
                data=body,
                namespace=None if is_cluster_scoped_resource else (document_namespace or None),
                verbose=False,
            )
            for obj, _ in _to_tuples(create_result):
                applied_kinds.append(getattr(obj, "kind", str(obj)))
        except ApiException as exc:
            deps.logger.error(
                "APPLY FAILED doc[%d]: %s status=%s reason=%s body=%s",
                index,
                _doc_id(doc),
                exc.status,
                exc.reason,
                getattr(exc, "body", None),
            )
            raise
        except Exception as exc:
            deps.logger.exception("APPLY FAILED doc[%d]: %s err=%s", index, _doc_id(doc), exc)
            raise

    return {"applied": applied_kinds}


def apply_yaml_content_with_namespace(
    content: str,
    namespace_override: str,
    *,
    dependencies: Optional[ApplyGatewayDependencies] = None,
) -> Dict[str, Any]:
    return apply_yaml_documents_with_namespace(
        yaml.safe_load_all(content),
        namespace_override,
        dependencies=dependencies,
    )


def apply_yaml_file_with_namespace(
    file_path: str,
    namespace_override: str,
    *,
    project_root: Optional[str] = None,
    dependencies: Optional[ApplyGatewayDependencies] = None,
) -> Dict[str, Any]:
    resolved_path = file_path
    if project_root and not os.path.isabs(resolved_path):
        resolved_path = os.path.join(project_root, resolved_path)

    deps = dependencies or ApplyGatewayDependencies()
    if not deps.file_exists(resolved_path):
        raise FileNotFoundError(f"YAML file not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as handle:
        return apply_yaml_documents_with_namespace(
            yaml.safe_load_all(handle),
            namespace_override,
            dependencies=deps,
        )
