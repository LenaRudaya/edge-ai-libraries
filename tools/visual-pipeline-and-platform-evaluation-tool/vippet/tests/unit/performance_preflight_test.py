# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the performance-test ViPPET pre-flight."""

import unittest
from collections.abc import Callable
from unittest.mock import Mock, patch

import httpx

from tests.performance.perf_helpers import preflight


class _FakeClock:
    def __init__(self) -> None:
        self.current = 0.0

    def monotonic(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.current += seconds


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestPerformancePreflight(unittest.TestCase):
    def test_ready_service_reports_health_and_status(self) -> None:
        requested_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested_paths.append(request.url.path)
            if request.url.path.endswith("/health"):
                return httpx.Response(200, json={"healthy": True})
            return httpx.Response(
                200, json={"status": "ready", "ready": True, "message": None}
            )

        reports: list[str] = []
        with _client(handler) as client:
            preflight.wait_for_vippet_ready(
                "http://localhost/api/v1/",
                60,
                2,
                10,
                client=client,
                report=reports.append,
            )

        self.assertEqual(requested_paths, ["/api/v1/health", "/api/v1/status"])
        self.assertIn("GET http://localhost/api/v1/health: OK", reports[0])
        self.assertIn("GET http://localhost/api/v1/status: READY", reports[1])

    def test_initializing_service_is_polled_until_ready(self) -> None:
        status_responses = iter(
            [
                {"status": "initializing", "ready": False, "message": "Loading"},
                {"status": "ready", "ready": True, "message": None},
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/health"):
                return httpx.Response(200, json={"healthy": True})
            return httpx.Response(200, json=next(status_responses))

        clock = _FakeClock()
        reports: list[str] = []
        with _client(handler) as client:
            preflight.wait_for_vippet_ready(
                "http://localhost/api/v1",
                60,
                2,
                10,
                client=client,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                report=reports.append,
            )

        self.assertEqual(clock.current, 2)
        self.assertTrue(any("status='initializing'" in line for line in reports))
        self.assertIn("READY", reports[-1])

    def test_unreachable_health_endpoint_is_actionable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        reports: list[str] = []
        with _client(handler) as client:
            with self.assertRaisesRegex(
                preflight.PreflightError, r"/health.*connection refused.*Remedy"
            ):
                preflight.wait_for_vippet_ready(
                    "http://localhost/api/v1",
                    60,
                    2,
                    10,
                    client=client,
                    report=reports.append,
                )

        self.assertIn("/health: FAILED", reports[-1])

    def test_malformed_health_payload_is_fatal(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[{"healthy": True}])

        with _client(handler) as client:
            with self.assertRaisesRegex(
                preflight.PreflightError, "expected an object, got list"
            ):
                preflight.wait_for_vippet_ready(
                    "http://localhost/api/v1", 60, 2, 10, client=client
                )

    def test_status_timeout_reports_last_observed_state(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/health"):
                return httpx.Response(200, json={"healthy": True})
            return httpx.Response(
                200,
                json={
                    "status": "initializing",
                    "ready": False,
                    "message": "Loading models",
                },
            )

        clock = _FakeClock()
        with _client(handler) as client:
            with self.assertRaises(preflight.PreflightError) as context:
                preflight.wait_for_vippet_ready(
                    "http://localhost/api/v1",
                    5,
                    2,
                    10,
                    client=client,
                    monotonic=clock.monotonic,
                    sleep=clock.sleep,
                    report=lambda _message: None,
                )

        message = str(context.exception)
        self.assertIn("timed out after 5s", message)
        self.assertIn("/status", message)
        self.assertIn("status='initializing'", message)
        self.assertIn("message='Loading models'", message)
        self.assertIn("Remedy:", message)

    @patch.object(preflight.pytest, "exit")
    @patch.object(
        preflight,
        "wait_for_vippet_ready",
        side_effect=preflight.PreflightError("service unavailable"),
    )
    def test_fatal_preflight_uses_dedicated_exit_code(
        self, _mock_wait: Mock, mock_exit: Mock
    ) -> None:
        preflight.run_preflight_or_exit("http://localhost/api/v1", 60, 2, 10)

        mock_exit.assert_called_once_with(
            "service unavailable", returncode=preflight.FATAL_PREFLIGHT_EXIT_CODE
        )
        self.assertEqual(preflight.FATAL_PREFLIGHT_EXIT_CODE, 2)


def _health_error(
    status: int, **kwargs: object
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, **kwargs)  # type: ignore[arg-type]

    return handler


class TestPreflightDiagnosis(unittest.TestCase):
    """Every fatal message states what happened, why and what to do."""

    def _fail(self, handler: Callable[[httpx.Request], httpx.Response]) -> str:
        with _client(handler) as client:
            with self.assertRaises(preflight.PreflightError) as context:
                preflight.wait_for_vippet_ready(
                    "http://h/api/v1", 60, 2, 10, client=client, report=lambda _: None
                )
        message = str(context.exception)
        self.assertIn("Why:", message)
        self.assertIn("Remedy:", message)
        return message

    def test_connection_refused_points_to_start_and_base_url(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        message = self._fail(handler)
        self.assertIn("not running", message)
        self.assertIn("make run", message)
        self.assertIn("--base-url", message)

    def test_404_points_to_api_prefix(self) -> None:
        message = self._fail(_health_error(404, text="not found"))
        self.assertIn("API prefix", message)
        self.assertIn("/api/v1", message)

    def test_5xx_points_to_logs(self) -> None:
        message = self._fail(_health_error(503, text="bad gateway"))
        self.assertIn("HTTP 503", message)
        self.assertIn("docker logs vippet", message)

    def test_non_json_points_to_wrong_service(self) -> None:
        message = self._fail(_health_error(200, text="<html>login</html>"))
        self.assertIn("not a ViPPET JSON object", message)

    def test_request_timeout_points_to_timeouts(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        message = self._fail(handler)
        self.assertIn("--timeout", message)

    def test_not_ready_timeout_points_to_readiness_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/health"):
                return httpx.Response(200, json={"healthy": True})
            return httpx.Response(200, json={"status": "initializing", "ready": False})

        clock = _FakeClock()
        with _client(handler) as client:
            with self.assertRaises(preflight.PreflightError) as context:
                preflight.wait_for_vippet_ready(
                    "http://h/api/v1",
                    4,
                    2,
                    10,
                    client=client,
                    monotonic=clock.monotonic,
                    sleep=clock.sleep,
                    report=lambda _: None,
                )
        message = str(context.exception)
        self.assertIn("not finished initialising", message)
        self.assertIn("--readiness-timeout (currently 4s)", message)

    def test_status_endpoint_errors_are_diagnosed_on_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/health"):
                return httpx.Response(200, json={"healthy": True})
            return httpx.Response(404, text="nope")

        clock = _FakeClock()
        with _client(handler) as client:
            with self.assertRaises(preflight.PreflightError) as context:
                preflight.wait_for_vippet_ready(
                    "http://h/api/v1",
                    4,
                    2,
                    10,
                    client=client,
                    monotonic=clock.monotonic,
                    sleep=clock.sleep,
                    report=lambda _: None,
                )
        self.assertIn("API prefix", str(context.exception))


class TestRunPreflightOrExit(unittest.TestCase):
    def test_returns_collected_lines_on_success(self) -> None:
        def fake_wait(*_args: object, report: Callable[[str], None]) -> None:
            report("line one")
            report("line two")

        with patch.object(preflight, "wait_for_vippet_ready", side_effect=fake_wait):
            lines = preflight.run_preflight_or_exit("http://h/api/v1", 60, 2, 10)
        self.assertEqual(lines, ["line one", "line two"])

    @patch.object(preflight.pytest, "exit")
    def test_on_failure_receives_lines_before_exit(self, mock_exit: Mock) -> None:
        def fake_wait(*_args: object, report: Callable[[str], None]) -> None:
            report("[pre-flight] FAILED")
            raise preflight.PreflightError("down")

        seen: list[list[str]] = []
        with patch.object(preflight, "wait_for_vippet_ready", side_effect=fake_wait):
            preflight.run_preflight_or_exit(
                "http://h/api/v1", 60, 2, 10, on_failure=seen.append
            )
        self.assertEqual(seen, [["[pre-flight] FAILED"]])
        mock_exit.assert_called_once_with(
            "down", returncode=preflight.FATAL_PREFLIGHT_EXIT_CODE
        )


if __name__ == "__main__":
    unittest.main()
