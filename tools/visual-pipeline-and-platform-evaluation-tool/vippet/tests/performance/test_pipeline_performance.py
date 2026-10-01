# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmark tests for VIPPET pipelines.

Each test case runs a single (pipeline, variant, stream_count) combination,
collects hardware metrics during execution, and appends results to the
session-scoped results_collector for report generation on teardown.

Every outcome (success, job failure, HTTP/transport error, timeout) is
recorded in the collector so the console summary and the reports list the
same runs. Failure messages follow "what happened / why / what to do".
"""

import logging
import time
from typing import Any, NoReturn

import pytest
import httpx

from helpers.api_helpers import (
    start_performance_job,
    wait_for_job_completion,
)
from helpers.pipeline_case_helpers import PipelineCase

from perf_helpers.config import (
    BASE_URL,
    MAX_RETRIES,
    MAX_RUNTIME,
    OUTPUT_MODE,
    POLL_TIMEOUT,
    RETRY_DELAY_SECONDS,
)
from perf_helpers.console import shorten
from perf_helpers.hw_monitor import HardwareMonitor

logger = logging.getLogger(__name__)


def _build_performance_payload(case: PipelineCase, streams: int) -> dict[str, Any]:
    """Construct the POST /tests/performance request body."""
    return {
        "pipeline_performance_specs": [
            {
                "pipeline": {
                    "source": "variant",
                    "pipeline_id": case.pipeline_id,
                    "variant_id": case.variant_id,
                },
                "streams": streams,
            }
        ],
        "execution_config": {
            "output_mode": OUTPUT_MODE,
            "max_runtime": MAX_RUNTIME,
        },
    }


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)


def _transport_failure(label: str, exc: httpx.TransportError) -> NoReturn:
    _fail(
        f"{label}: lost connection to ViPPET at {BASE_URL} "
        f"({type(exc).__name__}: {shorten(exc)}). Why: the service stopped, "
        "restarted or is overloaded. What to do: check `docker compose ps` "
        "and `docker logs vippet`, then re-run."
    )


def _start_job(session: httpx.Client, payload: dict[str, Any], label: str) -> str:
    try:
        return start_performance_job(session, payload)  # type: ignore[arg-type]
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        body = shorten(exc.response.text)
        if code == 409:
            _fail(
                f"{label}: POST /tests/performance was rejected with HTTP 409 "
                f"({body}). Why: another job is still running and ViPPET runs "
                "one job at a time. What to do: wait for or stop the running "
                "job (UI, or DELETE /jobs/tests/performance/<job_id>) and re-run."
            )
        _fail(
            f"{label}: POST /tests/performance returned HTTP {code} ({body}). "
            "Why: ViPPET rejected the request for this pipeline/variant. What "
            "to do: check `docker logs vippet` for the validation error and "
            "confirm the case still exists with --dry-run."
        )
    except httpx.TransportError as exc:
        _transport_failure(label, exc)


def _attempt_performance_job(
    session: httpx.Client, payload: dict[str, Any], label: str
) -> dict[str, Any]:
    """Submit a performance job and wait for it to finish."""
    job_id = _start_job(session, payload, label)
    status_url = f"{BASE_URL}/jobs/tests/performance/{job_id}/status"
    try:
        return wait_for_job_completion(session, status_url)  # type: ignore[arg-type]
    except pytest.fail.Exception as exc:
        _fail(
            f"{label}: job {job_id} is still running after {POLL_TIMEOUT}s "
            f"({shorten(exc)}). Why: the run takes longer than "
            "vippet.max_job_duration. What to do: raise --max-job-duration "
            "(VIPPET_JOB_TIMEOUT_SECONDS) or bound each run with --max-runtime."
        )
    except AssertionError as exc:
        _fail(
            f"{label}: job {job_id} was not RUNNING at the first status poll "
            f"({shorten(exc)}). Why: the pipeline failed to start or ended "
            "almost immediately. What to do: check `docker logs vippet` for "
            f"job {job_id} and the pipeline's input source."
        )
    except httpx.HTTPStatusError as exc:
        _fail(
            f"{label}: GET {status_url} returned HTTP "
            f"{exc.response.status_code} ({shorten(exc.response.text)}). Why: "
            "ViPPET lost track of the job, e.g. after a restart. What to do: "
            "check `docker logs vippet` and re-run."
        )
    except httpx.TransportError as exc:
        _transport_failure(label, exc)


@pytest.mark.perf
def test_pipeline_performance(
    request: pytest.FixtureRequest,
    http_client: httpx.Client,
    pipeline_case: PipelineCase | None,
    stream_count: int,
    hw_monitor: HardwareMonitor,
    results_collector: list[dict[str, Any]],
) -> None:
    """Run a performance benchmark for a single (pipeline, variant, stream_count) combination."""
    assert pipeline_case is not None
    callspec = getattr(request.node, "callspec", None)
    test_id = callspec.id if callspec is not None else request.node.name
    label = (
        f"pipeline_id={pipeline_case.pipeline_id} "
        f"variant={pipeline_case.device_family} streams={stream_count}"
    )
    logger.info(
        "Running performance benchmark: pipeline='%s' variant=%s streams=%d",
        pipeline_case.pipeline_name,
        pipeline_case.device_family,
        stream_count,
    )

    payload = _build_performance_payload(pipeline_case, stream_count)
    final_status: dict[str, Any] = {}
    attempts = 0
    failure: str | None = None

    start_time = time.time()
    hw_monitor.start()
    try:
        while True:
            attempts += 1
            final_status = _attempt_performance_job(http_client, payload, label)
            if final_status.get("state") == "COMPLETED" or attempts > MAX_RETRIES:
                break
            logger.warning(
                "Attempt %d/%d failed (state=%s, error=%s) - retrying after %.1fs",
                attempts,
                MAX_RETRIES + 1,
                final_status.get("state"),
                final_status.get("error_message"),
                RETRY_DELAY_SECONDS,
            )
            time.sleep(RETRY_DELAY_SECONDS)
    except pytest.fail.Exception as exc:
        # Hard failure (HTTP/transport error, job timeout, ...) already
        # reported with a "what/why/what to do" message by _attempt_performance_job.
        # Record it for the results entry below, then re-raise so this test
        # fails immediately: the finally block still runs (stops hw_monitor,
        # appends the results entry), but the soft-failure checks after the
        # try/except/finally (state != COMPLETED, fps == 0) are skipped since
        # re-raising here propagates out of the function.
        failure = str(exc)
        raise
    finally:
        hw_stats = hw_monitor.stop()
        state = final_status.get("state")
        is_success = failure is None and state == "COMPLETED"
        entry: dict[str, Any] = {
            "test_id": test_id,
            "pipeline_name": pipeline_case.pipeline_name,
            "pipeline_id": pipeline_case.pipeline_id,
            "variant_name": pipeline_case.device_family,
            "variant_id": pipeline_case.variant_id,
            "streams": stream_count,
            "status": "success" if is_success else "failed",
            "total_fps": final_status.get("total_fps") if is_success else None,
            "per_stream_fps": (
                final_status.get("per_stream_fps") if is_success else None
            ),
            "result": final_status or None,
            "hw_metrics": hw_stats,
            "duration_seconds": time.time() - start_time,
            "job_id": final_status.get("job_id", ""),
            "error": failure or final_status.get("error_message"),
        }
        results_collector.append(entry)

    job_id = final_status.get("job_id") or "?"
    if state != "COMPLETED":
        message = (
            f"{label}: job {job_id} ended in state {state!r} after {attempts} "
            f"attempt(s); error_message={final_status.get('error_message')!r}. "
            "Why: ViPPET could not run this pipeline on this device. What to "
            f"do: check `docker logs vippet` for job {job_id}, confirm the "
            f"{pipeline_case.device_family} device works, or raise --max-retries."
        )
        entry["error"] = message
        _fail(message)

    for key in ("total_fps", "per_stream_fps"):
        value = final_status.get(key) or 0
        if value <= 0:
            message = (
                f"{label}: job {job_id} completed but reported {key}={value}. "
                "Why: the pipeline processed no frames (empty or unreadable "
                "input, or a --max-runtime too short to measure). What to do: "
                "check the pipeline's input source and --max-runtime."
            )
            entry["status"] = "failed"
            entry["error"] = message
            _fail(message)
