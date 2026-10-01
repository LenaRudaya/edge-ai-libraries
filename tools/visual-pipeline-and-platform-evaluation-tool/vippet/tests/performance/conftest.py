# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and console reporting for VIPPET performance benchmark tests.

Console output of a normal run, in order:

1. header: effective settings and ViPPET readiness (``pytest_report_header``)
2. discovered hardware and the benchmark matrix (``pytest_collection_finish``)
3. one ``[perf]`` status line per test (``pytest_runtest_logreport``)
4. benchmark summary and artefact paths (``pytest_terminal_summary``)
"""

import dataclasses
import logging
import os
import re
import sys
import time
from collections.abc import Generator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import httpx

from perf_helpers.settings import ENV_CONFIG_ORIGIN, SETTINGS_REMEDY, SettingsError

try:
    from perf_helpers.config import (
        BASE_URL,
        CREATE_LATEST_LINK,
        METRICS_SAMPLE_INTERVAL,
        METRICS_URL,
        PERF_RESULTS_DIR,
        POLL_INTERVAL,
        POLL_TIMEOUT,
        READINESS_TIMEOUT_SECONDS,
        REQUEST_TIMEOUT,
        RESULT_FORMATS,
        SETTINGS,
        STREAM_COUNTS,
    )
except SettingsError as exc:
    # Same message and exit code (2) as the CLI instead of a traceback.
    # pytest.exit() cannot be used here: pytest wraps any Exception raised
    # while importing a conftest (including pytest's Exit) into a
    # ConftestImportFailure and prints a traceback with exit code 4.
    # SystemExit is a BaseException and is not wrapped.
    print(
        f"error: invalid performance config: {exc}. {SETTINGS_REMEDY}",
        file=sys.stderr,
    )
    raise SystemExit(2) from None

# Propagate perf config to env vars consumed by functional helpers. This must
# run BEFORE importing ``helpers.*``: helpers.config reads these env vars once
# at import time. These keys are env-backed in perf_helpers.settings, so an
# already-exported value is what BASE_URL/POLL_* resolved to; setdefault only
# fills in YAML/default values.
os.environ.setdefault("VIPPET_BASE_URL", BASE_URL)
os.environ.setdefault("VIPPET_JOB_TIMEOUT_SECONDS", str(POLL_TIMEOUT))
os.environ.setdefault("VIPPET_JOB_POLL_INTERVAL", str(POLL_INTERVAL))

from helpers.pipeline_case_helpers import (  # noqa: E402
    PipelineCase,
    wrap_cases_for_pytest,
)
from perf_helpers.console import (  # noqa: E402
    discovery_failure_message,
    format_artefacts,
    format_hardware,
    format_matrix,
    format_results_summary,
    format_settings,
    format_test_line,
    no_cases_reason,
)
from perf_helpers.discovery import discover_matrix  # noqa: E402
from perf_helpers.hw_monitor import HardwareMonitor  # noqa: E402
from perf_helpers.matrix import Matrix, MatrixFilters  # noqa: E402
from perf_helpers.preflight import (  # noqa: E402
    FATAL_PREFLIGHT_EXIT_CODE,
    run_preflight_or_exit,
)
from perf_helpers.reporters import ResultExporter, generate_html_report  # noqa: E402

logger = logging.getLogger(__name__)

_QUICK_STREAM_COUNTS: set[int] = {1, 3}
_QUICK_VARIANTS: set[str] = {"CPU", "GPU"}

_PIPELINE_CASES: list[PipelineCase | object] | None = None
_CASE_IDS: list[str] | None = None

# Console state shared between hooks (single pytest process, no xdist).
_PREFLIGHT_LINES: list[str] = []
_MATRIX: Matrix | None = None
_TERMINAL: Any = None
_ARTEFACTS: list[tuple[str, Path]] = []
_ARTEFACT_ERROR: list[str] = []
_SESSION_DURATION: list[float] = []

_PARAM_ID_RE = re.compile(r"\[(.*)\]$")


# --------------------------------------------------------------------------- #
# Console helpers
# --------------------------------------------------------------------------- #


def _terminal(config: pytest.Config) -> Any:
    return config.pluginmanager.get_plugin("terminalreporter")


def _write_lines(text_or_lines: str | list[str]) -> None:
    lines = (
        text_or_lines.splitlines() if isinstance(text_or_lines, str) else text_or_lines
    )
    for line in lines:
        if _TERMINAL is not None:
            _TERMINAL.write_line(line)
        else:
            print(line, file=sys.stderr)


def _collect_system_info(devices: Mapping[str, list[str]]) -> dict[str, Any]:
    """Build the report's system section from the discovered device names."""
    labels = {"CPU": "Processor", "GPU": "GPU", "NPU": "NPU"}
    system = {
        labels[family]: " / ".join(names)
        for family, names in devices.items()
        if family in labels and names
    }
    return {"system": system}


def _results() -> list[dict[str, Any]]:
    return _RESULTS_COLLECTOR_REF[0] if _RESULTS_COLLECTOR_REF else []


# --------------------------------------------------------------------------- #
# 1. Readiness (header)
# --------------------------------------------------------------------------- #


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    """Verify ViPPET readiness once before performance test collection."""
    global _TERMINAL
    _TERMINAL = _terminal(session.config)
    _PREFLIGHT_LINES[:] = run_preflight_or_exit(
        BASE_URL,
        READINESS_TIMEOUT_SECONDS,
        POLL_INTERVAL,
        REQUEST_TIMEOUT,
        on_failure=_write_lines,
    )
    _load_matrix()


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Show effective settings and the pre-flight result under the session header."""
    return [
        "ViPPET performance benchmark",
        *format_settings(SETTINGS, os.environ.get(ENV_CONFIG_ORIGIN)).splitlines(),
        "Readiness:",
        *(f"  {line}" for line in _PREFLIGHT_LINES),
    ]


# --------------------------------------------------------------------------- #
# 2. Hardware + matrix (after collection)
# --------------------------------------------------------------------------- #


def _load_matrix() -> Matrix:
    """Discover the matrix once, right after readiness.
    Runs in ``pytest_sessionstart`` because ``pytest.exit`` raised during
    module collection is reported as a collection error with a traceback;
    at session start it terminates cleanly with the given exit code.
    """
    global _MATRIX
    if _MATRIX is not None:
        return _MATRIX
    try:
        _MATRIX = discover_matrix(MatrixFilters.from_settings(SETTINGS))
    except Exception as exc:
        logger.debug("Pipeline discovery failed", exc_info=True)
        _write_lines(_PREFLIGHT_LINES)  # the header is never printed here
        pytest.exit(
            discovery_failure_message(BASE_URL, exc),
            returncode=FATAL_PREFLIGHT_EXIT_CODE,
        )
    logger.debug("Available device families: %s", _MATRIX.available_families)
    for excl in _MATRIX.excluded:
        logger.debug(
            "Excluded pipeline=%s variant=%s reason=%s (%s)",
            excl.pipeline_id,
            excl.variant,
            excl.reason.value,
            excl.detail,
        )
    return _MATRIX


def _discover_case_params() -> tuple[list[PipelineCase | object], list[str]]:
    """Wrap the discovered matrix for ``pytest.mark.parametrize``.
    Filtering (pipelines / variants / skip lists / host families) is done by
    :func:`perf_helpers.matrix.build_matrix`; missing-model handling stays in
    :func:`wrap_cases_for_pytest`.
    """
    matrix = _load_matrix()
    if not matrix.included:
        skip = pytest.mark.skip(reason=no_cases_reason(matrix))
        return [pytest.param(None, marks=skip)], ["no-cases"]
    cases = [PipelineCase(**dataclasses.asdict(case)) for case in matrix.included]
    missing = {pid: set(models) for pid, models in matrix.missing_models.items()}
    return wrap_cases_for_pytest(cases, missing)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate the cross-product parametrization: pipeline_case x stream_count."""
    global _PIPELINE_CASES, _CASE_IDS

    if (
        "pipeline_case" not in metafunc.fixturenames
        or "stream_count" not in metafunc.fixturenames
    ):
        return

    if _PIPELINE_CASES is None or _CASE_IDS is None:
        _PIPELINE_CASES, _CASE_IDS = _discover_case_params()

    params = []
    ids = []

    for case_param, case_id in zip(_PIPELINE_CASES, _CASE_IDS):
        actual_case: PipelineCase | None = None
        is_skipped = False
        skip_marks: list[pytest.Mark] = []

        if isinstance(case_param, PipelineCase):
            actual_case = case_param
        else:
            # pytest.param wrapper (ParameterSet) with .values and .marks attrs
            wrapped: Any = case_param
            actual_case = wrapped.values[0]
            skip_marks = list(wrapped.marks)
            is_skipped = any(m.name == "skip" for m in skip_marks)

        for streams in STREAM_COUNTS:
            marks: list[Any] = [pytest.mark.perf]
            marks.extend(skip_marks)

            if not is_skipped and actual_case is not None:
                device_family = actual_case.device_family.upper()
                variant_families = set(device_family.split("_"))
                is_quick = (
                    variant_families <= _QUICK_VARIANTS
                    and streams in _QUICK_STREAM_COUNTS
                )
                if is_quick:
                    marks.append(pytest.mark.perf_quick)
                marks.append(pytest.mark.perf_full)

            params.append(pytest.param(case_param, streams, marks=marks))
            ids.append(f"{case_id}_x{streams}")

    metafunc.parametrize(["pipeline_case", "stream_count"], params, ids=ids)


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session: pytest.Session) -> None:
    """Print discovered hardware and the benchmark matrix after collection."""
    if _MATRIX is None or _TERMINAL is None:
        return
    _TERMINAL.write_sep("-", "discovered hardware")
    _write_lines(format_hardware(_MATRIX))
    _TERMINAL.write_sep("-", "benchmark matrix")
    _write_lines(format_matrix(_MATRIX))
    _write_lines(
        f"Selected for this session: {len(session.items)} test(s) "
        "(after -k/-m and other pytest filters)"
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def http_client() -> Generator[httpx.Client, None, None]:
    """Reusable HTTP client for all performance tests."""
    client = httpx.Client(
        headers={"Accept": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    yield client
    client.close()


@pytest.fixture
def hw_monitor() -> HardwareMonitor:
    """Create a HardwareMonitor instance for per-test HW sampling."""
    return HardwareMonitor(METRICS_URL, METRICS_SAMPLE_INTERVAL)


@pytest.fixture(scope="session")
def results_collector(request: pytest.FixtureRequest) -> list[dict[str, Any]]:
    """Session-scoped accumulator that exports results on teardown."""
    results: list[dict[str, Any]] = []
    start_time = time.time()

    def _finalize() -> None:
        total_duration = time.time() - start_time
        _SESSION_DURATION[:] = [total_duration]
        if not results:
            return
        benchmark_id = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = Path(PERF_RESULTS_DIR) / benchmark_id

        system_info = _collect_system_info(_MATRIX.devices if _MATRIX else {})

        hw_families: dict[str, list[str]] = {}
        for r in results:
            family = r.get("variant_name", "").upper()
            for part in family.split("_"):
                if part in {"CPU", "GPU", "NPU"}:
                    hw_families.setdefault(part, [])
                    device_name = system_info.get("system", {}).get(
                        part if part != "CPU" else "Processor", ""
                    )
                    if device_name and device_name not in hw_families[part]:
                        hw_families[part].append(device_name)

        n_skipped = sum(1 for r in results if r["status"] == "skipped")
        result_dict: dict[str, Any] = {
            "benchmark_id": benchmark_id,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": total_duration,
            "test_cases": results,
            "summary": {
                "total": len(results),
                "success": sum(1 for r in results if r["status"] == "success"),
                "failed": sum(1 for r in results if r["status"] == "failed"),
                "skipped": n_skipped,
            },
            "hardware": hw_families,
            "system_info": system_info,
        }

        try:
            exporter = ResultExporter(output_dir, formats=RESULT_FORMATS)
            _ARTEFACTS.extend(exporter.export(result_dict))
            html_path = output_dir / f"{benchmark_id}.html"
            html_path.write_text(generate_html_report([result_dict]))
            _ARTEFACTS.append(("HTML", html_path))
        except OSError as exc:
            message = (
                f"could not write benchmark results to {output_dir}: {exc}. "
                "Why: the directory is not writable or the disk is full. "
                "What to do: pass a writable --results-dir (PERF_RESULTS_DIR) "
                "and re-run."
            )
            _ARTEFACT_ERROR[:] = [message]
            pytest.fail(message, pytrace=False)

        if CREATE_LATEST_LINK:
            latest_link = Path(PERF_RESULTS_DIR) / "latest"
            try:
                latest_link.unlink(missing_ok=True)
                latest_link.symlink_to(output_dir.name)
                _ARTEFACTS.append(("latest", latest_link))
            except OSError as exc:
                message = (
                    f"results were written, but the 'latest' link {latest_link} "
                    f"could not be updated: {exc}. Why: the filesystem does not "
                    "support symlinks or a directory has that name. What to "
                    "do: re-run with --no-latest-link or remove that path."
                )
                _ARTEFACT_ERROR[:] = [message]
                pytest.fail(message, pytrace=False)

    request.addfinalizer(_finalize)
    return results


_RESULTS_COLLECTOR_REF: list[list[dict[str, Any]]] = []


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:  # type: ignore[type-arg]
    """Capture skipped perf tests into the results collector."""
    if call.when != "setup":
        return
    if call.excinfo is None:
        return
    if not call.excinfo.errisinstance(pytest.skip.Exception):
        return
    if not any(m.name == "perf" for m in item.iter_markers()):
        return

    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return
    params = callspec.params
    case_param = params.get("pipeline_case")
    stream_count = params.get("stream_count", 0)

    actual_case: PipelineCase | None = None
    if isinstance(case_param, PipelineCase):
        actual_case = case_param
    elif case_param is not None:
        wrapped: Any = case_param
        vals = getattr(wrapped, "values", None)
        if vals:
            actual_case = vals[0]

    skip_reason = str(call.excinfo.value)

    entry = {
        "test_id": callspec.id,
        "pipeline_name": actual_case.pipeline_name if actual_case else "unknown",
        "pipeline_id": actual_case.pipeline_id if actual_case else "",
        "variant_name": actual_case.device_family if actual_case else "",
        "variant_id": actual_case.variant_id if actual_case else "",
        "streams": stream_count,
        "status": "skipped",
        "total_fps": None,
        "per_stream_fps": None,
        "result": None,
        "hw_metrics": {"sample_count": 0},
        "duration_seconds": 0,
        "job_id": "",
        "error": skip_reason,
    }

    if _RESULTS_COLLECTOR_REF:
        _RESULTS_COLLECTOR_REF[0].append(entry)


@pytest.fixture(autouse=True, scope="session")
def _bind_results_ref(results_collector: list[dict[str, Any]]) -> None:
    """Bind the results_collector list to the module-level ref for the skip hook."""
    _RESULTS_COLLECTOR_REF.clear()
    _RESULTS_COLLECTOR_REF.append(results_collector)


# --------------------------------------------------------------------------- #
# 3. Per-test status
# --------------------------------------------------------------------------- #


def _report_detail(report: pytest.TestReport) -> str | None:
    longrepr = report.longrepr
    if longrepr is None:
        return None
    if isinstance(longrepr, tuple) and len(longrepr) == 3:  # (path, line, reason)
        return str(longrepr[2]).removeprefix("Skipped: ")
    crash = getattr(longrepr, "reprcrash", None)
    if crash is not None and getattr(crash, "message", None):
        return str(crash.message)
    text = str(longrepr).strip().splitlines()
    return text[-1] if text else None


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Print one ``[perf]`` status line per finished performance test."""
    if _TERMINAL is None or "perf" not in report.keywords:
        return
    if report.when == "call":
        outcome = report.outcome
    elif report.when in ("setup", "teardown") and report.outcome != "passed":
        outcome = "error" if report.failed else report.outcome
    else:
        return

    match = _PARAM_ID_RE.search(report.nodeid)
    test_id = match.group(1) if match else report.nodeid
    entry = next((r for r in reversed(_results()) if r.get("test_id") == test_id), None)
    _write_lines(format_test_line(test_id, outcome, entry, _report_detail(report)))


# --------------------------------------------------------------------------- #
# 4. Summary + artefact paths
# --------------------------------------------------------------------------- #


def pytest_terminal_summary(
    terminalreporter: Any, exitstatus: int, config: pytest.Config
) -> None:
    """Print the benchmark summary followed by the written artefact paths."""
    if _MATRIX is None or config.option.collectonly:
        return
    duration = _SESSION_DURATION[0] if _SESSION_DURATION else None
    terminalreporter.write_sep("=", "ViPPET performance summary")
    for line in format_results_summary(_results(), duration).splitlines():
        terminalreporter.write_line(line)
    error = _ARTEFACT_ERROR[0] if _ARTEFACT_ERROR else None
    for line in format_artefacts(_ARTEFACTS, error).splitlines():
        terminalreporter.write_line(line)
