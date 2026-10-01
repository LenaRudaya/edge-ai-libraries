# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ViPPET reachability and readiness checks for performance tests."""

import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

FATAL_PREFLIGHT_EXIT_CODE = 2


class PreflightError(RuntimeError):
    """Raised when ViPPET cannot become ready for performance discovery."""


def _remaining_timeout(
    deadline: float,
    request_timeout: float,
    monotonic: Callable[[], float],
) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise PreflightError("readiness timeout expired")
    return min(request_timeout, remaining)


def _state_description(payload: dict[str, Any]) -> str:
    return (
        f"status={payload.get('status', 'unknown')!r}, "
        f"ready={payload.get('ready', 'missing')!r}, "
        f"message={payload.get('message')!r}"
    )


def _get_json_object(
    client: httpx.Client, url: str, timeout: float
) -> tuple[httpx.Response, dict[str, Any]]:
    response = client.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object, got {type(payload).__name__}")
    return response, payload


def diagnose(exc: BaseException, base_url: str) -> tuple[str, str]:
    """Return ``(why, remedy)`` for a failed pre-flight request."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 404:
            return (
                "the endpoint does not exist at this URL, so --base-url "
                "probably lacks the API prefix or points at another service",
                f"set --base-url (VIPPET_BASE_URL) to the ViPPET API root, "
                f"e.g. http://<host>/api/v1 (currently {base_url})",
            )
        if code in (401, 403):
            return (
                f"the server refused access (HTTP {code}); a proxy or gateway "
                "in front of ViPPET may require authentication",
                "point --base-url at an endpoint reachable without "
                "interactive login, or configure access for this host",
            )
        if code >= 500:
            return (
                f"ViPPET or its reverse proxy returned HTTP {code}, so the "
                "service is up but failing or still starting",
                "check `docker compose ps` and `docker logs vippet`, then "
                "re-run; raise --readiness-timeout if it is still starting",
            )
        return (
            f"the server answered HTTP {code}, which ViPPET does not return "
            "for a healthy instance",
            f"verify that --base-url ({base_url}) points at the ViPPET API",
        )
    if isinstance(exc, httpx.TimeoutException):
        return (
            "the request timed out, so ViPPET is overloaded, blocked by a "
            "firewall, or --timeout/--readiness-timeout is too short",
            "check `docker logs vippet`, network access to the host, and "
            "raise --timeout or --readiness-timeout",
        )
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError)):
        return (
            "nothing accepted the connection, so ViPPET is not running or "
            "--base-url points at the wrong host or port",
            "start ViPPET (`make run`), confirm it with `docker compose ps`, "
            f"and check --base-url (VIPPET_BASE_URL, currently {base_url})",
        )
    if isinstance(exc, httpx.HTTPError):
        return (
            "the HTTP request could not be completed",
            f"verify --base-url ({base_url}) and network access to ViPPET",
        )
    if isinstance(exc, ValueError):
        return (
            "the response is not a ViPPET JSON object, so the URL points at "
            "another service, a proxy error page or a login page",
            f"verify that --base-url ({base_url}) points at the ViPPET API "
            "root (ending in /api/v1)",
        )
    return (
        "the readiness deadline expired before a response arrived",
        "raise --readiness-timeout and check `docker logs vippet`",
    )


def wait_for_vippet_ready(
    base_url: str,
    readiness_timeout_seconds: float,
    poll_interval: float,
    request_timeout: float,
    *,
    client: httpx.Client | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    report: Callable[[str], None] = print,
) -> None:
    """Check ViPPET health and wait for its status endpoint to report ready."""
    health_url = f"{base_url.rstrip('/')}/health"
    status_url = f"{base_url.rstrip('/')}/status"
    deadline = monotonic() + readiness_timeout_seconds
    owned_client = client is None
    active_client = client or httpx.Client(headers={"Accept": "application/json"})

    try:
        try:
            timeout = _remaining_timeout(deadline, request_timeout, monotonic)
            response, health_payload = _get_json_object(
                active_client, health_url, timeout
            )
            report(
                f"[pre-flight] GET {health_url}: OK "
                f"(HTTP {response.status_code}, healthy={health_payload.get('healthy')!r})"
            )
        except (httpx.HTTPError, ValueError, PreflightError) as exc:
            observed = f"{type(exc).__name__}: {exc}"
            report(f"[pre-flight] GET {health_url}: FAILED ({observed})")
            why, remedy = diagnose(exc, base_url)
            raise PreflightError(
                f"ViPPET pre-flight failed for {health_url}; observed {observed}. "
                f"Why: {why}. Remedy: {remedy}."
            ) from exc

        last_observed = "no status response received"
        last_error: BaseException | None = None
        while monotonic() < deadline:
            try:
                timeout = _remaining_timeout(deadline, request_timeout, monotonic)
                response, payload = _get_json_object(active_client, status_url, timeout)
                last_observed = _state_description(payload)
                last_error = None
                if payload.get("ready") is True:
                    report(
                        f"[pre-flight] GET {status_url}: READY "
                        f"(HTTP {response.status_code}, {last_observed})"
                    )
                    return
                report(
                    f"[pre-flight] GET {status_url}: WAITING "
                    f"(HTTP {response.status_code}, {last_observed})"
                )
            except (httpx.HTTPError, ValueError, PreflightError) as exc:
                last_observed = f"{type(exc).__name__}: {exc}"
                last_error = exc
                report(f"[pre-flight] GET {status_url}: WAITING ({last_observed})")

            remaining = deadline - monotonic()
            if remaining > 0:
                sleep(min(poll_interval, remaining))

        if last_error is not None and not isinstance(last_error, PreflightError):
            why, remedy = diagnose(last_error, base_url)
        else:
            why = (
                "ViPPET is healthy but has not finished initialising (e.g. "
                "still loading pipelines or models), or initialisation failed"
            )
            remedy = (
                "wait and re-run, raise --readiness-timeout (currently "
                f"{readiness_timeout_seconds:g}s), and check `docker logs "
                "vippet` for startup errors"
            )
        raise PreflightError(
            f"ViPPET pre-flight timed out after {readiness_timeout_seconds:g}s "
            f"waiting for {status_url}; last observed {last_observed}. "
            f"Why: {why}. Remedy: {remedy}."
        )
    finally:
        if owned_client:
            active_client.close()


def run_preflight_or_exit(
    base_url: str,
    readiness_timeout_seconds: float,
    poll_interval: float,
    request_timeout: float,
    *,
    on_failure: Callable[[list[str]], None] | None = None,
) -> list[str]:
    """Run pre-flight and return its progress lines.

    The lines are collected instead of printed so the caller can show them
    in the pytest header. On failure ``on_failure`` receives the lines seen
    so far, then pytest terminates with :data:`FATAL_PREFLIGHT_EXIT_CODE`.
    """
    lines: list[str] = []
    try:
        wait_for_vippet_ready(
            base_url,
            readiness_timeout_seconds,
            poll_interval,
            request_timeout,
            report=lines.append,
        )
    except PreflightError as exc:
        if on_failure is not None:
            on_failure(lines)
        pytest.exit(str(exc), returncode=FATAL_PREFLIGHT_EXIT_CODE)
    return lines
