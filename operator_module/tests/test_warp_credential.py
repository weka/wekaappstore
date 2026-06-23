"""Unit tests for WarpCredential reconciler helpers and handlers (Phase 22).

Covers requirements OPS-01 through OPS-09 and API-08:
  OPS-01: PermanentError on unknown spec.type (reason='UnknownType')
  OPS-02: TemporaryError(delay=30) + reason='KeyMissing' when source Secret not found
  OPS-03: PermanentError + reason='EmptyKey' on whitespace-only key value
  OPS-04: nvidia-ngc derives two Secrets (apikey + dockerconfigjson)
  OPS-05: huggingface derives one Secret (HF_API_KEY)
  OPS-06: weka-storage derives one Secret with three keys (USERNAME, TOKEN, ENDPOINT)
  OPS-07: status.conditions, derivedSecrets, lastSyncTime updated after success
  OPS-08: delete handler logs warning and does NOT delete derived Secrets
  OPS-09: idempotency — create→409→patch flow restores deleted derived Secret
  API-08: no raw key value in any log record at any level (caplog assertion)

Test categories:
  1. Pure-derivation helper tests (no kr8s mocking required)
  2. _apply_secret_idempotent idempotency and error-dispatch tests
  3. reconcile_warpcredential handler-path tests (kr8s mocked)
  4. delete_warpcredential warning-only test
  5. API-08 caplog-based log-safety tests (load-bearing enforcement)

All kr8s interactions are mocked via unittest.mock.patch on 'main.kr8s.objects.Secret'.
No cluster, no network, no subprocess — pure in-process unit tests.
"""
from __future__ import annotations

import base64
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# --- sys.path defense-in-depth (conftest.py also does this) ---
OPERATOR_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(OPERATOR_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_MODULE_ROOT))


# ----------------------------------------------------------------------
# File-local mock helpers (NOT imported from test_appstack.py — per-file
# copy convention prevents cross-test-file import coupling, D-16)
# ----------------------------------------------------------------------


def _make_kr8s_secret(data_dict):
    """Return a MagicMock whose .data maps each key to base64-encoded value (mimics kr8s Secret).

    Mirrors test_appstack.py:62-69 verbatim (S-6 per-file-copy convention).
    The .data attribute matches what kr8s.objects.Secret.get() returns in production:
    each value is a base64-encoded UTF-8 string (Kubernetes Secret API convention).
    """
    secret = MagicMock()
    secret.data = {
        k: base64.b64encode(v.encode('utf-8') if isinstance(v, str) else v).decode('utf-8')
        for k, v in data_dict.items()
    }
    return secret


def _make_kr8s_server_error(status_code, message='server error'):
    """Construct a kr8s.ServerError with .response.status_code set.

    Mirrors test_appstack.py:72-80 verbatim (S-6 per-file-copy convention).
    kr8s 0.20.10 does not have AlreadyExistsError — all 4xx/5xx come through
    kr8s.ServerError with e.response.status_code (RESEARCH.md §1, Pitfall 1).
    """
    import kr8s

    err = kr8s.ServerError(message)
    response = MagicMock()
    response.status_code = status_code
    err.response = response
    return err


def _make_patch_obj():
    """Return a kopf patch stand-in supporting patch.status['key'] = value.

    The kopf patch object used in handler tests:
      - patch.status must be a real dict (not a MagicMock) so that
        assertions like patch_obj.status['conditions'][0]['reason'] work.
    Docstring cites RESEARCH.md §Section 6.
    """
    patch_obj = MagicMock()
    patch_obj.status = {}
    return patch_obj


def _make_secret_class_mock():
    """Return a factory callable that mimics the kr8s.objects.Secret constructor.

    Each invocation records the raw dict and returns a MagicMock with:
      .create() — defaults to no-op success (override side_effect per test)
      .patch()  — defaults to no-op success (override side_effect per test)
      .raw      — the raw dict passed at construction time

    The factory exposes .instances (list of returned mocks) so tests can assert
    on derived-Secret metadata, type, and data without reaching into internals.

    Note: ownerReferences must NOT appear in any recorded raw dict (T-22-05, OPS-08).
    Cited: RESEARCH.md §Section 6, Pattern S-5.
    """
    instances = []

    def _factory(raw_dict):
        mock_secret = MagicMock()
        mock_secret.raw = raw_dict
        mock_secret.create = MagicMock(return_value=None)
        mock_secret.patch = MagicMock(return_value=None)
        instances.append(mock_secret)
        return mock_secret

    _factory.instances = instances
    return _factory


# ----------------------------------------------------------------------
# 1. Pure-derivation tests (no kr8s mocking — helpers are pure functions)
# ----------------------------------------------------------------------


def test_ngc_apikey_data_is_b64_encoded():
    """OPS-04, D-12: _derive_ngc_payloads returns all NGC key aliases as standard padded base64.

    The Opaque secret mirrors the canonical ngc-api / ngc-api-key secrets that NVIDIA
    AIDP/NIM charts consume: NGC_API_KEY, NGC_CLI_API_KEY, NVIDIA_API_KEY, and apiKey,
    all holding the same key value.
    """
    from main import _derive_ngc_payloads

    apikey, _docker = _derive_ngc_payloads('my-secret-key')
    expected = base64.b64encode(b'my-secret-key').decode('ascii')
    assert apikey == {
        'NGC_API_KEY': expected,
        'NGC_CLI_API_KEY': expected,
        'NVIDIA_API_KEY': expected,
        'apiKey': expected,
    }


def test_ngc_docker_auth_is_oauthtoken_b64():
    """D-12: NGC docker secret auth field is base64('$oauthtoken:<key>') with standard padding."""
    from main import _derive_ngc_payloads

    _apikey, docker = _derive_ngc_payloads('my-secret-key')
    docker_json = json.loads(base64.b64decode(docker['.dockerconfigjson']))
    assert docker_json['auths']['nvcr.io']['username'] == '$oauthtoken'
    assert docker_json['auths']['nvcr.io']['password'] == 'my-secret-key'
    assert docker_json['auths']['nvcr.io']['auth'] == base64.b64encode(
        b'$oauthtoken:my-secret-key'
    ).decode('ascii')


def test_hf_payload_has_only_hf_api_key():
    """OPS-05: _derive_hf_payload returns exactly one key HF_API_KEY, base64-encoded."""
    from main import _derive_hf_payload

    data = _derive_hf_payload('hf-token-xyz')
    assert set(data.keys()) == {'HF_API_KEY'}
    assert base64.b64decode(data['HF_API_KEY']).decode() == 'hf-token-xyz'


def test_weka_payload_three_keys():
    """OPS-06: _derive_weka_payload returns exactly WEKA_API_USERNAME, TOKEN, ENDPOINT."""
    from main import _derive_weka_payload

    data = _derive_weka_payload('admin', 'tok-abc', 'https://weka-cluster:14000')
    assert set(data.keys()) == {'WEKA_API_USERNAME', 'WEKA_API_TOKEN', 'WEKA_API_ENDPOINT'}
    assert base64.b64decode(data['WEKA_API_USERNAME']).decode() == 'admin'
    assert base64.b64decode(data['WEKA_API_TOKEN']).decode() == 'tok-abc'
    assert base64.b64decode(data['WEKA_API_ENDPOINT']).decode() == 'https://weka-cluster:14000'


# ----------------------------------------------------------------------
# 2. _apply_secret_idempotent idempotency and error-dispatch tests
# ----------------------------------------------------------------------


def test_apply_secret_idempotent_creates_on_first_call():
    """OPS-09, D-02: on first call, create() is called once; patch() is NOT called."""
    from main import _apply_secret_idempotent

    secret = MagicMock()
    secret.raw = {'data': {'k': 'v'}, 'type': 'Opaque'}
    secret.create = MagicMock(return_value=None)
    secret.patch = MagicMock(return_value=None)

    _apply_secret_idempotent(secret, ctx='test-ctx')

    secret.create.assert_called_once()
    secret.patch.assert_not_called()


def test_apply_secret_idempotent_patches_on_409():
    """OPS-09, D-02: on 409 Conflict, patch() is called with exact {data, type} shape."""
    from main import _apply_secret_idempotent

    secret = MagicMock()
    secret.raw = {'data': {'k': 'v'}, 'type': 'Opaque'}
    secret.create.side_effect = _make_kr8s_server_error(409)
    secret.patch = MagicMock(return_value=None)

    _apply_secret_idempotent(secret, ctx='test-ctx')

    secret.create.assert_called_once()
    secret.patch.assert_called_once_with({'data': {'k': 'v'}, 'type': 'Opaque'})


def test_apply_secret_idempotent_500_raises_temporary():
    """D-10: ServerError(503) raises kopf.TemporaryError with delay=30."""
    import kopf
    from main import _apply_secret_idempotent

    secret = MagicMock()
    secret.raw = {'data': {}, 'type': 'Opaque'}
    secret.create.side_effect = _make_kr8s_server_error(503)

    with pytest.raises(kopf.TemporaryError) as exc_info:
        _apply_secret_idempotent(secret, ctx='test-ctx')

    assert getattr(exc_info.value, 'delay', None) == 30


def test_apply_secret_idempotent_403_raises_permanent():
    """D-10: ServerError(403) raises kopf.PermanentError (4xx non-409 path)."""
    import kopf
    from main import _apply_secret_idempotent

    secret = MagicMock()
    secret.raw = {'data': {}, 'type': 'Opaque'}
    secret.create.side_effect = _make_kr8s_server_error(403)

    with pytest.raises(kopf.PermanentError):
        _apply_secret_idempotent(secret, ctx='test-ctx')


def test_apply_secret_idempotent_timeout_raises_temporary():
    """D-10: kr8s.APITimeoutError raises kopf.TemporaryError with delay=30."""
    import kopf
    import kr8s
    from main import _apply_secret_idempotent

    secret = MagicMock()
    secret.raw = {'data': {}, 'type': 'Opaque'}
    secret.create.side_effect = kr8s.APITimeoutError('timeout')

    with pytest.raises(kopf.TemporaryError) as exc_info:
        _apply_secret_idempotent(secret, ctx='test-ctx')

    assert getattr(exc_info.value, 'delay', None) == 30


# ----------------------------------------------------------------------
# 3. reconcile_warpcredential handler-path tests (kr8s mocked)
# ----------------------------------------------------------------------


def test_reconcile_unknown_type_raises_permanent_with_status():
    """OPS-01, D-08: unknown spec.type -> PermanentError AND status reason='UnknownType'."""
    import kopf
    from main import reconcile_warpcredential

    patch_obj = _make_patch_obj()
    spec = {'type': 'invalid-type', 'displayName': 'X', 'secretRef': {'name': 'src', 'key': 'k'}}

    with pytest.raises(kopf.PermanentError):
        reconcile_warpcredential(
            body={}, spec=spec, name='test', namespace='weka-app-store',
            patch=patch_obj, logger=logging.getLogger('test'),
        )

    assert patch_obj.status['conditions'][0]['reason'] == 'UnknownType'


def test_reconcile_missing_secret_raises_temporary_with_status():
    """OPS-02, D-07: NotFoundError -> TemporaryError(delay=30) AND reason='KeyMissing'."""
    import kopf
    import kr8s
    from main import reconcile_warpcredential

    patch_obj = _make_patch_obj()
    spec = {'type': 'huggingface', 'displayName': 'HF Test', 'secretRef': {'name': 'src', 'key': 'k'}}

    with pytest.raises(kopf.TemporaryError) as exc_info:
        with patch('main.kr8s.objects.Secret') as mock_secret_cls:
            mock_secret_cls.get.side_effect = kr8s.NotFoundError('not found')
            reconcile_warpcredential(
                body={}, spec=spec, name='hf-test', namespace='weka-app-store',
                patch=patch_obj, logger=logging.getLogger('test'),
            )

    assert exc_info.value.delay == 30
    assert patch_obj.status['conditions'][0]['reason'] == 'KeyMissing'


def test_reconcile_empty_key_raises_permanent_with_status():
    """OPS-03, D-09: whitespace-only key -> PermanentError AND reason='EmptyKey'; message has key NAME not VALUE."""
    import kopf
    from main import reconcile_warpcredential

    patch_obj = _make_patch_obj()
    src_secret = _make_kr8s_secret({'NGC_API_KEY': '   '})
    spec = {'type': 'nvidia-ngc', 'displayName': 'NGC Test', 'secretRef': {'name': 'src', 'key': 'NGC_API_KEY'}}

    with pytest.raises(kopf.PermanentError) as exc_info:
        with patch('main.kr8s.objects.Secret') as mock_secret_cls:
            mock_secret_cls.get.return_value = src_secret
            reconcile_warpcredential(
                body={}, spec=spec, name='ngc-test', namespace='weka-app-store',
                patch=patch_obj, logger=logging.getLogger('test'),
            )

    assert patch_obj.status['conditions'][0]['reason'] == 'EmptyKey'
    # T-22-04: key NAME in message, not value
    assert 'NGC_API_KEY' in str(exc_info.value)
    assert '   ' not in str(exc_info.value)


def test_reconcile_ngc_success_creates_two_derived_secrets():
    """OPS-04, OPS-07: nvidia-ngc path creates two derived Secrets; status updated; no ownerReferences."""
    from main import reconcile_warpcredential

    patch_obj = _make_patch_obj()
    src_secret = _make_kr8s_secret({'NGC_API_KEY': 'my-key'})
    factory = _make_secret_class_mock()
    spec = {
        'type': 'nvidia-ngc',
        'displayName': 'NGC',
        'secretRef': {'name': 'src', 'key': 'NGC_API_KEY'},
    }

    with patch('main.kr8s.objects.Secret') as mock_secret_cls:
        mock_secret_cls.get.return_value = src_secret
        mock_secret_cls.side_effect = factory
        reconcile_warpcredential(
            body={}, spec=spec, name='ngc-test', namespace='weka-app-store',
            patch=patch_obj, logger=logging.getLogger('test'),
        )

    # Two derived Secrets must be constructed
    assert len(factory.instances) == 2

    # First: apikey Secret (Opaque)
    instance0 = factory.instances[0]
    assert instance0.raw['metadata']['name'] == 'warp-ngc-test-apikey'
    assert instance0.raw['type'] == 'Opaque'
    # No ownerReferences (T-22-05, OPS-08 — K8s GC must NOT cascade to derived Secrets)
    assert 'ownerReferences' not in instance0.raw['metadata']

    # Second: docker Secret (dockerconfigjson)
    instance1 = factory.instances[1]
    assert instance1.raw['metadata']['name'] == 'warp-ngc-test-docker'
    assert instance1.raw['type'] == 'kubernetes.io/dockerconfigjson'
    assert 'ownerReferences' not in instance1.raw['metadata']

    # Status: two conditions (KeyReady + DockerSecretReady), derivedSecrets, lastSyncTime
    conditions = patch_obj.status['conditions']
    condition_types = [c['type'] for c in conditions]
    assert 'KeyReady' in condition_types
    assert 'DockerSecretReady' in condition_types
    for c in conditions:
        assert c['status'] == 'True'

    derived = patch_obj.status['derivedSecrets']
    assert len(derived) == 2
    for d in derived:
        assert 'name' in d
        assert 'type' in d

    assert 'lastSyncTime' in patch_obj.status


def test_reconcile_hf_success_creates_one_token_secret():
    """OPS-05, OPS-07: huggingface path creates one warp-{name}-token Secret (Opaque, HF_API_KEY)."""
    from main import reconcile_warpcredential

    patch_obj = _make_patch_obj()
    src_secret = _make_kr8s_secret({'hf-token': 'hf-secret-value'})
    factory = _make_secret_class_mock()
    spec = {
        'type': 'huggingface',
        'displayName': 'HF',
        'secretRef': {'name': 'src', 'key': 'hf-token'},
    }

    with patch('main.kr8s.objects.Secret') as mock_secret_cls:
        mock_secret_cls.get.return_value = src_secret
        mock_secret_cls.side_effect = factory
        reconcile_warpcredential(
            body={}, spec=spec, name='hf-test', namespace='weka-app-store',
            patch=patch_obj, logger=logging.getLogger('test'),
        )

    assert len(factory.instances) == 1
    instance = factory.instances[0]
    assert instance.raw['metadata']['name'] == 'warp-hf-test-token'
    assert instance.raw['type'] == 'Opaque'
    assert 'HF_API_KEY' in instance.raw['data']
    assert 'ownerReferences' not in instance.raw['metadata']

    conditions = patch_obj.status['conditions']
    assert any(c['type'] == 'KeyReady' and c['status'] == 'True' for c in conditions)
    assert patch_obj.status['derivedSecrets'] == [{'name': 'warp-hf-test-token', 'type': 'Opaque'}]
    assert 'lastSyncTime' in patch_obj.status


def test_reconcile_weka_success_three_keys_and_endpoint_status():
    """OPS-06, OPS-07: weka-storage creates Secret with three keys; status.wekaEndpoint set."""
    from main import reconcile_warpcredential

    patch_obj = _make_patch_obj()
    src_secret = _make_kr8s_secret({
        'WEKA_API_USERNAME': 'admin',
        'WEKA_API_TOKEN': 'tok',
        'WEKA_API_ENDPOINT': 'https://weka:14000',
    })
    factory = _make_secret_class_mock()
    spec = {
        'type': 'weka-storage',
        'displayName': 'WEKA Prod',
        'secretRef': {'name': 'src', 'key': 'WEKA_API_TOKEN'},
        'endpoint': 'https://weka:14000',
    }

    with patch('main.kr8s.objects.Secret') as mock_secret_cls:
        mock_secret_cls.get.return_value = src_secret
        mock_secret_cls.side_effect = factory
        reconcile_warpcredential(
            body={}, spec=spec, name='weka-test', namespace='weka-app-store',
            patch=patch_obj, logger=logging.getLogger('test'),
        )

    assert len(factory.instances) == 1
    instance = factory.instances[0]
    assert instance.raw['metadata']['name'] == 'warp-weka-test-token'
    assert instance.raw['type'] == 'Opaque'
    assert set(instance.raw['data'].keys()) == {'WEKA_API_USERNAME', 'WEKA_API_TOKEN', 'WEKA_API_ENDPOINT'}
    assert 'ownerReferences' not in instance.raw['metadata']

    assert patch_obj.status['wekaEndpoint'] == 'https://weka:14000'
    assert 'lastSyncTime' in patch_obj.status


def test_reconcile_weka_missing_username_raises_permanent():
    """OPS-06, D-09: missing WEKA_API_USERNAME -> PermanentError, reason='KeyMissing',
    exception message references key NAME 'WEKA_API_USERNAME'."""
    import kopf
    from main import reconcile_warpcredential

    patch_obj = _make_patch_obj()
    # Source Secret missing WEKA_API_USERNAME
    src_secret = _make_kr8s_secret({
        'WEKA_API_TOKEN': 'tok',
        'WEKA_API_ENDPOINT': 'https://weka:14000',
    })
    spec = {
        'type': 'weka-storage',
        'displayName': 'WEKA',
        'secretRef': {'name': 'src', 'key': 'WEKA_API_TOKEN'},
        'endpoint': 'https://weka:14000',
    }

    with pytest.raises(kopf.PermanentError) as exc_info:
        with patch('main.kr8s.objects.Secret') as mock_secret_cls:
            mock_secret_cls.get.return_value = src_secret
            reconcile_warpcredential(
                body={}, spec=spec, name='weka-test', namespace='weka-app-store',
                patch=patch_obj, logger=logging.getLogger('test'),
            )

    # Status must be patched to an error condition
    assert patch_obj.status['conditions'][0]['status'] == 'False'
    # Exception message must reference the missing key NAME
    assert 'WEKA_API_USERNAME' in str(exc_info.value)


def test_reconcile_idempotent_restore_on_resume():
    """OPS-09, D-02: 409 on derived Secret create -> patch() called with exact {data, type} shape.

    Simulates the 'derived Secret manually deleted, then operator resumes' scenario.
    The reconcile detects the existing Secret (409) and patches it to restore the desired state.
    """
    from main import reconcile_warpcredential

    patch_obj = _make_patch_obj()
    src_secret = _make_kr8s_secret({'hf-token': 'hf-restore-test'})
    spec = {
        'type': 'huggingface',
        'displayName': 'HF Restore',
        'secretRef': {'name': 'src', 'key': 'hf-token'},
    }

    # Track constructed Secret instances so we can assert on them later
    constructed_instances = []

    def patching_factory(raw_dict):
        inst = MagicMock()
        inst.raw = raw_dict
        inst.patch = MagicMock(return_value=None)
        # Simulate 409 on create (Secret already exists — restore scenario)
        inst.create = MagicMock(side_effect=_make_kr8s_server_error(409))
        constructed_instances.append(inst)
        return inst

    with patch('main.kr8s.objects.Secret') as mock_secret_cls:
        mock_secret_cls.get.return_value = src_secret
        mock_secret_cls.side_effect = patching_factory
        reconcile_warpcredential(
            body={}, spec=spec, name='hf-test', namespace='weka-app-store',
            patch=patch_obj, logger=logging.getLogger('test'),
        )

    assert len(constructed_instances) == 1
    instance = constructed_instances[0]
    instance.create.assert_called_once()
    # Patch must be called with the exact two-key shape from D-02
    instance.patch.assert_called_once_with({
        'data': instance.raw['data'],
        'type': instance.raw['type'],
    })


# ----------------------------------------------------------------------
# 4. delete_warpcredential warning-only test (OPS-08, D-05)
# ----------------------------------------------------------------------


def test_delete_warpcredential_logs_warning_and_does_nothing():
    """OPS-08, D-05: delete handler emits one warning; kr8s Secret methods are NOT called."""
    from main import delete_warpcredential

    mock_logger = MagicMock()

    # Patch kr8s Secret so any call to .get/.create/.patch raises AssertionError
    def _fail(*args, **kwargs):
        raise AssertionError('kr8s Secret method must NOT be called from delete handler (OPS-08)')

    with patch('main.kr8s.objects.Secret') as mock_secret_cls:
        mock_secret_cls.get.side_effect = _fail
        mock_secret_cls.return_value.create.side_effect = _fail
        mock_secret_cls.return_value.patch.side_effect = _fail

        delete_warpcredential(name='hf-test', namespace='weka-app-store', logger=mock_logger)

    mock_logger.warning.assert_called_once()
    warning_call = mock_logger.warning.call_args
    warning_msg = warning_call[0][0] if warning_call[0] else str(warning_call)
    # Message must reference name and namespace
    assert 'hf-test' in warning_msg
    assert 'weka-app-store' in warning_msg


# ----------------------------------------------------------------------
# 5. API-08 caplog-based log-safety tests (load-bearing enforcement)
# ----------------------------------------------------------------------

# Load-bearing assertion for API-08: this is the only enforcement that no key value leaks into operator logs at any level. DO NOT WEAKEN.
def test_no_key_in_logs_anywhere(caplog):
    """API-08, T-22-01, D-03: raw key value MUST NOT appear in any log record at any log level."""
    from main import reconcile_warpcredential

    test_key = 'super-secret-test-key-value-do-not-leak-42'
    src_secret = _make_kr8s_secret({'NGC_API_KEY': test_key})
    patch_obj = _make_patch_obj()
    factory = _make_secret_class_mock()

    with caplog.at_level(logging.DEBUG, logger='main'):
        with caplog.at_level(logging.DEBUG):
            with patch('main.kr8s.objects.Secret') as mock_secret_cls:
                mock_secret_cls.get.return_value = src_secret
                mock_secret_cls.side_effect = factory
                reconcile_warpcredential(
                    body={},
                    spec={
                        'type': 'nvidia-ngc',
                        'displayName': 'NGC Test',
                        'secretRef': {'name': 'src', 'key': 'NGC_API_KEY'},
                    },
                    name='ngc-test',
                    namespace='weka-app-store',
                    patch=patch_obj,
                    logger=logging.getLogger('test'),
                )

    # Must have captured at least one log record (confirms caplog was actually active)
    assert len(caplog.records) >= 1, 'No log records captured — was caplog.at_level active?'
    # Confirm that records from the module logger or the injected test logger were captured
    assert any(r.name in ('main', 'test') for r in caplog.records), (
        'No records from main or test logger — module log paths may not be covered'
    )

    # Across ALL captured records at ANY level, the raw key value must not appear
    for record in caplog.records:
        msg = record.getMessage()
        assert test_key not in msg, (
            f'Key leaked in log record [{record.levelname}] [{record.name}]: {msg!r}'
        )
        assert test_key not in str(record.args or ''), (
            f'Key leaked in log record args [{record.levelname}]: {record.args!r}'
        )


def test_no_key_in_exception_message_on_empty_key(caplog):
    """T-22-04, D-09: empty-key PermanentError message has key NAME but NOT key VALUE (whitespace)."""
    import kopf
    from main import reconcile_warpcredential

    # Whitespace is the key value; it must not appear in the exception message
    src_secret = _make_kr8s_secret({'NGC_API_KEY': '   '})
    patch_obj = _make_patch_obj()

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(kopf.PermanentError) as exc_info:
            with patch('main.kr8s.objects.Secret') as mock_secret_cls:
                mock_secret_cls.get.return_value = src_secret
                reconcile_warpcredential(
                    body={},
                    spec={
                        'type': 'nvidia-ngc',
                        'displayName': 'NGC Test',
                        'secretRef': {'name': 'src', 'key': 'NGC_API_KEY'},
                    },
                    name='ngc-test',
                    namespace='weka-app-store',
                    patch=patch_obj,
                    logger=logging.getLogger('test'),
                )

    # T-22-04: key NAME is acceptable in the error message
    assert 'NGC_API_KEY' in str(exc_info.value)

    # Key VALUE (whitespace) must not appear quoted or otherwise in the exception message
    assert "'   '" not in str(exc_info.value)

    # Also scan caplog records (defensive — even for whitespace values, don't leak)
    for record in caplog.records:
        msg = record.getMessage()
        assert "'   '" not in msg, (
            f'Whitespace key value leaked in log [{record.levelname}]: {msg!r}'
        )
