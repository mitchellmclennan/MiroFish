from pathlib import Path
from types import SimpleNamespace

import pytest
from zep_cloud.core.api_error import ApiError as ZepApiError

from app.utils import zep


def _clear_zep_env(monkeypatch):
    """Isolate a test from any ambient Zep endpoint configuration."""

    monkeypatch.delenv("ZEP_API_URL", raising=False)
    monkeypatch.delenv("ZEP_BASE_URL", raising=False)
    monkeypatch.delenv("ZEP_MODE", raising=False)


def test_permanent_zep_errors_fail_without_retry():
    calls = []

    def operation():
        calls.append(True)
        raise ZepApiError(status_code=400, body={"message": "bad query"})

    with pytest.raises(ZepApiError):
        zep.call_zep_read_with_retry(
            operation,
            operation_name="permanent failure",
            sleep=lambda _seconds: None,
        )

    assert len(calls) == 1


def test_rate_limit_retry_respects_retry_after():
    calls = []
    sleeps = []

    def operation():
        calls.append(True)
        if len(calls) == 1:
            raise ZepApiError(
                status_code=429,
                headers={"Retry-After": "7"},
                body={"message": "slow down"},
            )
        return "ok"

    result = zep.call_zep_read_with_retry(
        operation,
        operation_name="rate limited read",
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert len(calls) == 2
    assert sleeps == [7.0]


def test_zep_client_is_shared_and_uses_an_explicit_timeout(monkeypatch):
    created = []

    def fake_zep(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(kwargs=kwargs)

    _clear_zep_env(monkeypatch)
    monkeypatch.setattr(zep, "Zep", fake_zep)
    zep.clear_zep_client_cache()

    first = zep.get_zep_client(" test-key ", timeout=12)
    second = zep.get_zep_client("test-key", timeout=12)

    assert first is second
    assert created == [{
        "api_key": "test-key",
        "base_url": zep.ZEP_CLOUD_BASE_URL,
        "timeout": 12.0,
    }]
    zep.clear_zep_client_cache()


def test_zep_mode_defaults_to_cloud_with_the_cloud_base_url(monkeypatch):
    _clear_zep_env(monkeypatch)

    assert zep.get_zep_mode() == "cloud"
    assert zep.is_local_zep_mode() is False
    assert zep.get_zep_base_url() == "https://api.getzep.com/api/v2"


def test_zep_base_url_is_configurable_in_cloud_mode(monkeypatch):
    _clear_zep_env(monkeypatch)
    monkeypatch.setenv("ZEP_BASE_URL", "https://zep-proxy.example.com/api/v2/")

    assert zep.get_zep_base_url() == "https://zep-proxy.example.com/api/v2"
    assert zep.get_zep_mode() == "cloud"


def test_local_mode_requires_an_explicit_base_url(monkeypatch):
    _clear_zep_env(monkeypatch)
    monkeypatch.setenv("ZEP_MODE", "local")

    with pytest.raises(ValueError, match="ZEP_BASE_URL"):
        zep.get_zep_base_url()


def test_local_mode_accepts_an_explicit_local_base_url(monkeypatch):
    _clear_zep_env(monkeypatch)
    monkeypatch.setenv("ZEP_MODE", "local")
    monkeypatch.setenv("ZEP_BASE_URL", "http://localhost:8000/api/v2")

    assert zep.is_local_zep_mode() is True
    assert zep.get_zep_base_url() == "http://localhost:8000/api/v2"


def test_invalid_zep_mode_is_rejected(monkeypatch):
    _clear_zep_env(monkeypatch)
    monkeypatch.setenv("ZEP_MODE", "trial")

    with pytest.raises(ValueError, match="ZEP_MODE"):
        zep.get_zep_mode()


def test_zep_client_uses_the_configured_base_url(monkeypatch):
    """B1：ZEP_BASE_URL必须真正作用于共享SDK客户端（含缓存键）。"""

    created = []

    def fake_zep(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(kwargs=kwargs)

    _clear_zep_env(monkeypatch)
    monkeypatch.setenv("ZEP_BASE_URL", "http://localhost:8000/api/v2")
    monkeypatch.setattr(zep, "Zep", fake_zep)
    zep.clear_zep_client_cache()

    first = zep.get_zep_client("test-key")

    assert created == [{
        "api_key": "test-key",
        "base_url": "http://localhost:8000/api/v2",
        "timeout": zep.ZEP_HTTP_REQUEST_TIMEOUT_SECONDS,
    }]

    monkeypatch.setenv("ZEP_BASE_URL", "https://api.getzep.com/api/v2")
    second = zep.get_zep_client("test-key")

    assert first is not second
    assert created[-1]["base_url"] == "https://api.getzep.com/api/v2"
    zep.clear_zep_client_cache()


def test_zep_client_rejects_self_hosted_endpoint_override(monkeypatch):
    monkeypatch.setenv("ZEP_API_URL", "https://example.invalid")

    with pytest.raises(ValueError, match="ZEP_API_URL"):
        zep.get_zep_client("test-key")


def test_zep_client_uses_internal_timeout_and_ignores_env_overrides(monkeypatch):
    created = []

    def fake_zep(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(kwargs=kwargs)

    _clear_zep_env(monkeypatch)
    monkeypatch.setenv("ZEP_REQUEST_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("ZEP_INGESTION_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(zep, "Zep", fake_zep)
    zep.clear_zep_client_cache()

    zep.get_zep_client("test-key")

    assert created == [{
        "api_key": "test-key",
        "base_url": zep.ZEP_CLOUD_BASE_URL,
        "timeout": zep.ZEP_HTTP_REQUEST_TIMEOUT_SECONDS,
    }]
    assert zep.ZEP_HTTP_REQUEST_TIMEOUT_SECONDS == 60.0
    # The ingestion wait window must keep allowing slow asynchronous
    # extraction; assert the floor, not the exact trial-tuned value.
    assert zep.ZEP_INGESTION_WAIT_TIMEOUT_SECONDS >= 600
    zep.clear_zep_client_cache()


def test_zep_timeout_policy_is_not_exposed_in_env_example():
    env_example = Path(__file__).resolve().parents[2] / ".env.example"
    contents = env_example.read_text(encoding="utf-8")

    assert "ZEP_REQUEST_TIMEOUT_SECONDS" not in contents
    assert "ZEP_INGESTION_TIMEOUT_SECONDS" not in contents


def test_zep_deployment_config_is_documented_in_env_example():
    env_example = Path(__file__).resolve().parents[2] / ".env.example"
    contents = env_example.read_text(encoding="utf-8")

    assert "ZEP_BASE_URL" in contents
    assert "ZEP_MODE" in contents
