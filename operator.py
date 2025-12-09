import logging
import kopf
import kr8s
import subprocess
import yaml
import tempfile
import os
import time
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


def load_values_from_reference(kind: str, name: str, key: str, namespace: str) -> Dict[str, Any]:
    """
    Load Helm values from a ConfigMap or Secret
    """
    try:
        if kind == "ConfigMap":
            cm = kr8s.objects.ConfigMap.get(name=name, namespace=namespace)
            values_yaml = cm.data.get(key, "")
        elif kind == "Secret":
            secret = kr8s.objects.Secret.get(name=name, namespace=namespace)
            import base64
            values_yaml = base64.b64decode(secret.data.get(key, "")).decode('utf-8')
        else:
            raise ValueError(f"Unsupported kind: {kind}")
        
        return yaml.safe_load(values_yaml) or {}
    except Exception as e:
        logging.error(f"Error loading values from {kind}/{name}: {str(e)}")
        return {}


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

    target_namespace = component.get('targetNamespace', namespace)

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

        # If the selector matched nothing on the chosen resource type, try a deployment as a common fallback
        if 'no matching resources' in stderr:
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


def handle_appstack_deployment(body, spec, name, namespace, status):
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
                    for values_ref in component['valuesFiles']:
                        ref_ns = values_ref.get('namespace', target_namespace)
                        ref_values = load_values_from_reference(
                            kind=values_ref['kind'],
                            name=values_ref['name'],
                            key=values_ref['key'],
                            namespace=ref_ns
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
                    # Write manifest to temp file and apply
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                        f.write(manifest_yaml)
                        manifest_file = f.name
                    
                    try:
                        cmd = ["kubectl", "apply", "-f", manifest_file, "-n", target_namespace]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                        
                        if result.returncode == 0:
                            comp_status['phase'] = 'Ready'
                            comp_status['message'] = 'Manifest applied successfully'
                        else:
                            comp_status['phase'] = 'Failed'
                            comp_status['message'] = f"Failed to apply manifest: {result.stderr}"
                            failed = True
                    finally:
                        if os.path.exists(manifest_file):
                            os.unlink(manifest_file)
            else:
                raise ValueError(f"Component {comp_name} must specify either helmChart or kubernetesManifest")
                
        except Exception as e:
            comp_status['phase'] = 'Failed'
            comp_status['message'] = f"Error deploying component: {str(e)}"
            failed = True
            logging.error(f"Error deploying component {comp_name}: {str(e)}")
        
        component_statuses.append(comp_status)
        
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
    
    return {
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


# Function to handle the creation of a new WarrpAppStore
@kopf.on.create('warp.io', 'v1alpha1', 'wekaappstores')
def create_warrpappstore_function(body, spec, name, namespace, status, **kwargs):
    logging.info(f"*** WarrpAppStore Created: {name}")
    
    # Check for AppStack deployment (multi-component)
    if 'appStack' in spec and spec['appStack']:
        return handle_appstack_deployment(body, spec, name, namespace, status)
    # Check if Helm chart is specified (single component)
    elif 'helmChart' in spec and spec['helmChart']:
        return handle_helm_deployment(body, spec, name, namespace, status)
    # Fall back to legacy pod-based deployment
    elif 'image' in spec and 'binary' in spec:
        return handle_pod_deployment(body, spec, name, namespace)
    else:
        error_msg = "Either appStack, helmChart, or image+binary must be specified"
        logging.error(error_msg)
        raise kopf.PermanentError(error_msg)


def handle_helm_deployment(body, spec, name, namespace, status):
    """Handle Helm-based deployment"""
    logging.info(f"Deploying {name} via Helm")
    
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
    
    return {
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


def handle_pod_deployment(body, spec, name, namespace):
    """Handle legacy pod-based deployment"""
    logging.info(f"Deploying {name} via Pod (legacy mode)")
    
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
    
    return {
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


# Handle updates to WarrpAppStore resources
@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores')
def update_warrpappstore_function(body, spec, name, namespace, status, **kwargs):
    logging.info(f"*** WarrpAppStore Updated: {name}")
    
    # For AppStack deployments, re-deploy the entire stack
    if 'appStack' in spec and spec['appStack']:
        return handle_appstack_deployment(body, spec, name, namespace, status)
    # For single Helm deployments, trigger upgrade
    elif 'helmChart' in spec and spec['helmChart']:
        return handle_helm_deployment(body, spec, name, namespace, status)
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
                # For raw manifests, attempt to delete using kubectl
                manifest_yaml = component['kubernetesManifest']
                target_namespace = component.get('targetNamespace', namespace)
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    f.write(manifest_yaml)
                    manifest_file = f.name
                
                try:
                    cmd = ["kubectl", "delete", "-f", manifest_file, "-n", target_namespace, "--ignore-not-found"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        logging.info(f"Component {comp_name} manifest deleted successfully")
                    else:
                        logging.warning(f"Failed to delete component {comp_name} manifest: {result.stderr}")
                finally:
                    if os.path.exists(manifest_file):
                        os.unlink(manifest_file)
    
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
