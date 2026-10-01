# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Plain-text console rendering shared by the CLI ``--dry-run`` and pytest.

Every function here is pure (no I/O) and returns a string, so the same
output is produced whether it is printed by :mod:`perf_helpers.cli` or
written to the pytest terminal by ``conftest.py``.

User-facing error messages follow a "what happened / why / what to do"
layout so they are actionable without reading the source.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .matrix import Exclusion, Matrix
from .settings import ResolvedSettings, SPECS

_MAX_ERROR_CHARS = 300


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a left-aligned text table with a dashed header underline."""
    cells = [list(map(str, headers))] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    lines = []
    for n, row in enumerate(cells):
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
        if n == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def render(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(map(str, value)) if value else "(none)"
    return str(value)


def shorten(text: Any, limit: int = _MAX_ERROR_CHARS) -> str:
    """Collapse whitespace and truncate *text* for single-line output."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _fmt_number(value: Any, decimals: int = 2) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return f"{value:.{decimals}f}"


# --------------------------------------------------------------------------- #
# Settings / hardware / matrix
# --------------------------------------------------------------------------- #


def format_settings(settings: ResolvedSettings, origin: str | None = None) -> str:
    """Render every setting with its value and source.
    *origin* is the config the user selected when the CLI hands pytest a
    resolved temporary YAML file (see ``ENV_CONFIG_ORIGIN``).
    """
    rows = [
        (spec.key, render(settings[spec.key]), settings.sources[spec.key])
        for spec in SPECS
    ]
    header = f"Config file: {settings.config_path}"
    if origin:
        header = (
            f"Config file: {origin} (resolved with env vars and CLI flags by "
            "perf_helpers.cli; 'yaml' below includes those overrides)"
        )
    return header + "\n" + table(("setting", "value", "source"), rows)


def format_hardware(matrix: Matrix) -> str:
    """Render the device families (and device names) ViPPET reported."""
    families = ", ".join(matrix.available_families) or "(none)"
    lines = [f"Host device families: {families}"]
    rows = [
        (family, " / ".join(matrix.devices.get(family) or []) or "(name not reported)")
        for family in matrix.available_families
    ]
    if rows:
        lines.append(table(("family", "device"), rows))
    else:
        lines.append(
            "  No CPU/GPU/NPU device reported by GET /devices. Why: ViPPET "
            "could not enumerate OpenVINO devices. What to do: check the "
            "container's device access (e.g. /dev/dri, /dev/accel) and "
            "`docker logs vippet`."
        )
    return "\n".join(lines)


def exclusion_counts(excluded: Iterable[Exclusion]) -> str:
    counts = Counter(e.reason.value for e in excluded)
    return ", ".join(f"{reason}={n}" for reason, n in sorted(counts.items()))


def format_matrix(matrix: Matrix) -> str:
    """Render the matrix, exclusions and run-time (missing model) skips."""
    out: list[str] = []
    streams = ", ".join(map(str, matrix.stream_counts))

    rows = matrix.rows()
    will_run = [
        (case.pipeline_id, case.device_family, s, case.case_id + f"_x{s}")
        for case, s in rows
        if case.pipeline_id not in matrix.missing_models
    ]
    out.append(
        f"Matrix: {len(will_run)} run(s) = pipeline x variant x streams [{streams}]"
    )
    out.append(
        table(("pipeline", "variant", "streams", "test id"), will_run)
        if will_run
        else "  (empty)"
    )

    out.append("")
    out.append(f"Excluded: {len(matrix.excluded)} pipeline/variant(s)")
    out.append(
        table(
            ("pipeline", "variant", "reason", "detail"),
            [
                (
                    e.pipeline_id or e.pipeline_name or "?",
                    e.variant,
                    e.reason.value,
                    e.detail,
                )
                for e in matrix.excluded
            ],
        )
        if matrix.excluded
        else "  (none)"
    )

    skipped = [
        (
            case.pipeline_id,
            case.device_family,
            s,
            ", ".join(matrix.missing_models[case.pipeline_id]),
        )
        for case, s in rows
        if case.pipeline_id in matrix.missing_models
    ]
    out.append("")
    out.append(f"Skipped at run time: missing_models: {len(skipped)} run(s)")
    out.append(
        table(("pipeline", "variant", "streams", "missing models"), skipped)
        if skipped
        else "  (none)"
    )
    if skipped:
        out.append(
            "  What to do: install the listed models through the ViPPET UI "
            "Models page or POST /models/download, then re-run."
        )
    return "\n".join(out)


def no_cases_reason(matrix: Matrix | None) -> str:
    """Actionable reason used when the matrix has no runnable case."""
    if matrix is not None and matrix.excluded:
        return (
            f"All {len(matrix.excluded)} discovered pipeline/variant case(s) "
            f"were excluded ({exclusion_counts(matrix.excluded)}). Why: the "
            "config filters or the host's device families rule them out. "
            "What to do: run the CLI with --dry-run to see the reason per "
            "case, then adjust --pipelines, --variants, --skip-pipelines or "
            "--skip-variants."
        )
    if matrix is not None and not matrix.available_families:
        return (
            "No benchmark case: ViPPET reports no CPU/GPU/NPU device. Why: "
            "OpenVINO device enumeration failed inside the container. What "
            "to do: check device access (/dev/dri, /dev/accel) and "
            "`docker logs vippet`."
        )
    return (
        "No benchmark case: GET /pipelines returned no pipeline with a "
        "CPU/GPU/NPU variant. Why: ViPPET has no pipelines loaded. What to "
        "do: check `docker logs vippet` for pipeline loading errors."
    )


def discovery_failure_message(base_url: str, exc: BaseException) -> str:
    return (
        "Pipeline discovery failed: GET /pipelines, /devices or /models at "
        f"{base_url} raised {type(exc).__name__}: {shorten(exc)}. Why: ViPPET "
        "passed the readiness check but a discovery endpoint failed or "
        "returned unexpected data. What to do: check `docker logs vippet`, "
        "confirm --base-url (VIPPET_BASE_URL) points at the ViPPET API and "
        "re-run; use --dry-run to repeat discovery without running jobs."
    )


# --------------------------------------------------------------------------- #
# Per-test status / summary / artefacts
# --------------------------------------------------------------------------- #


def format_test_line(
    test_id: str,
    outcome: str,
    entry: Mapping[str, Any] | None = None,
    detail: str | None = None,
) -> str:
    """One ``[perf]`` status line for a finished test."""
    parts = [f"[perf] {outcome.upper():<7} {test_id}"]
    if entry is not None:
        if entry.get("total_fps") is not None:
            parts.append(f"total_fps={_fmt_number(entry.get('total_fps'))}")
            parts.append(f"per_stream_fps={_fmt_number(entry.get('per_stream_fps'))}")
        duration = entry.get("duration_seconds")
        if duration:
            parts.append(f"duration={_fmt_number(duration, 1)}s")
        if entry.get("job_id"):
            parts.append(f"job_id={entry['job_id']}")
    if detail:
        # Keep only "what happened"; the full why/what-to-do text is in
        # pytest's FAILURES section and in the reports.
        what = detail.removeprefix("Failed: ").split(" Why: ", 1)[0]
        parts.append(f"- {shorten(what, 200)}")
    return "  ".join(parts)


def format_results_summary(
    results: Sequence[Mapping[str, Any]], duration_seconds: float | None = None
) -> str:
    counts = Counter(str(r.get("status")) for r in results)
    header = (
        f"Benchmark runs: {len(results)} total, {counts.get('success', 0)} "
        f"success, {counts.get('failed', 0)} failed, "
        f"{counts.get('skipped', 0)} skipped"
    )
    if duration_seconds is not None:
        header += f" in {duration_seconds:.1f}s"
    if not results:
        return header
    rows = [
        (
            r.get("test_id") or r.get("pipeline_id") or "?",
            r.get("status"),
            _fmt_number(r.get("total_fps")),
            _fmt_number(r.get("per_stream_fps")),
            _fmt_number(r.get("duration_seconds"), 1),
        )
        for r in results
    ]
    lines = [
        header,
        table(("test id", "status", "total_fps", "per_stream_fps", "secs"), rows),
    ]
    no_hw = [
        r
        for r in results
        if r.get("status") != "skipped"
        and not (r.get("hw_metrics") or {}).get("sample_count")
    ]
    if no_hw:
        lines.append(
            f"Note: {len(no_hw)} run(s) have no hardware metrics. Why: the "
            "metrics endpoint was unreachable or returned no known metrics. "
            "What to do: check --metrics-url (VIPPET_METRICS_URL) and that "
            "the metrics service is running (`docker compose ps`)."
        )
    return "\n".join(lines)


def format_artefacts(paths: Sequence[tuple[str, Path]], error: str | None) -> str:
    lines: list[str] = []
    if paths:
        lines.append("Artefacts:")
        width = max(len(label) for label, _ in paths)
        lines.extend(f"  {label.ljust(width)}  {path}" for label, path in paths)
    if error:
        lines.append(f"error: {error}")
    if not lines:
        lines.append(
            "No artefacts written: no performance test produced a result "
            "(all cases deselected, or the session stopped early)."
        )
    return "\n".join(lines)
