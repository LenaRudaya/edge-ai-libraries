# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the performance-benchmark console formatters."""

import unittest
from pathlib import Path
from typing import Any

from tests.performance.perf_helpers import console
from tests.performance.perf_helpers.matrix import Matrix, MatrixFilters, build_matrix
from tests.performance.perf_helpers.settings import resolve_settings

_PIPELINES: list[dict[str, Any]] = [
    {
        "id": "od",
        "name": "Object Detection",
        "variants": [{"id": "1", "name": "CPU"}, {"id": "2", "name": "NPU"}],
    },
    {"id": "lpr", "name": "LPR", "variants": [{"id": "3", "name": "CPU"}]},
]
_DEVICES = [{"device_family": "CPU", "full_device_name": "Core Ultra 7"}]


def _matrix(**filters: Any) -> Matrix:
    return build_matrix(
        _PIPELINES, _DEVICES, {"lpr": {"LPRNet"}}, MatrixFilters(**filters)
    )


class TestHardwareAndMatrix(unittest.TestCase):
    def test_hardware_lists_families_and_names(self) -> None:
        text = console.format_hardware(_matrix())
        self.assertIn("Host device families: CPU", text)
        self.assertIn("Core Ultra 7", text)

    def test_hardware_without_devices_is_actionable(self) -> None:
        text = console.format_hardware(Matrix())
        self.assertIn("(none)", text)
        self.assertIn("What to do", text)

    def test_matrix_lists_runs_exclusions_and_missing_models(self) -> None:
        text = console.format_matrix(_matrix(stream_counts=(1,)))
        self.assertIn("Matrix: 1 run(s)", text)
        self.assertIn("object_detection_cpu_x1", text)
        self.assertIn("unsupported_family", text)
        self.assertIn("Skipped at run time: missing_models: 1 run(s)", text)
        self.assertIn("LPRNet", text)
        self.assertIn("POST /models/download", text)


class TestActionableMessages(unittest.TestCase):
    def _assert_actionable(self, message: str) -> None:
        self.assertIn("Why:", message)
        self.assertIn("What to do:", message)

    def test_no_cases_when_everything_excluded_counts_reasons(self) -> None:
        message = console.no_cases_reason(_matrix(skip_pipelines=("od", "lpr")))
        self._assert_actionable(message)
        self.assertIn("skip_pipelines=2", message)
        self.assertIn("--dry-run", message)

    def test_no_cases_without_devices(self) -> None:
        message = console.no_cases_reason(build_matrix([], [], {}, MatrixFilters()))
        self._assert_actionable(message)
        self.assertIn("no CPU/GPU/NPU device", message)

    def test_no_cases_without_pipelines(self) -> None:
        message = console.no_cases_reason(
            build_matrix([], _DEVICES, {}, MatrixFilters())
        )
        self._assert_actionable(message)
        self.assertIn("no pipelines loaded", message)

    def test_discovery_failure(self) -> None:
        message = console.discovery_failure_message(
            "http://h/api/v1", RuntimeError("boom\nsecond line")
        )
        self._assert_actionable(message)
        self.assertIn("RuntimeError: boom second line", message)
        self.assertIn("--dry-run", message)


class TestRunOutput(unittest.TestCase):
    def test_passed_line_shows_fps_duration_and_job(self) -> None:
        line = console.format_test_line(
            "od_cpu_x1",
            "passed",
            {
                "total_fps": 120.5,
                "per_stream_fps": 120.5,
                "duration_seconds": 12.34,
                "job_id": "j1",
            },
        )
        self.assertTrue(line.startswith("[perf] PASSED  od_cpu_x1"))
        self.assertIn("total_fps=120.50", line)
        self.assertIn("duration=12.3s", line)
        self.assertIn("job_id=j1", line)

    def test_failed_line_keeps_only_what_happened(self) -> None:
        line = console.format_test_line(
            "od_gpu_x1",
            "failed",
            None,
            "Failed: job j3 ended in state 'FAILED'. Why: x. What to do: y.",
        )
        self.assertIn("- job j3 ended in state 'FAILED'.", line)
        self.assertNotIn("Why:", line)
        self.assertNotIn("Failed:", line)

    def test_summary_counts_and_missing_hw_note(self) -> None:
        results = [
            {
                "test_id": "a",
                "status": "success",
                "total_fps": 10.0,
                "hw_metrics": {"sample_count": 0},
            },
            {"test_id": "b", "status": "failed", "hw_metrics": {"sample_count": 3}},
            {"test_id": "c", "status": "skipped", "hw_metrics": {"sample_count": 0}},
        ]
        text = console.format_results_summary(results, 4.2)
        self.assertIn(
            "Benchmark runs: 3 total, 1 success, 1 failed, 1 skipped in 4.2s", text
        )
        self.assertIn("1 run(s) have no hardware metrics", text)
        self.assertIn("--metrics-url", text)

    def test_artefacts_paths_error_and_empty(self) -> None:
        text = console.format_artefacts(
            [("JSON", Path("/r/b.json")), ("latest", Path("/r/latest"))], None
        )
        self.assertIn("Artefacts:", text)
        self.assertIn("JSON    /r/b.json", text)
        self.assertIn("error: boom", console.format_artefacts([], "boom"))
        self.assertIn("No artefacts written", console.format_artefacts([], None))


class TestSettings(unittest.TestCase):
    def test_origin_replaces_temp_config_path(self) -> None:
        settings = resolve_settings("default", env={})
        self.assertIn(
            f"Config file: {settings.config_path}", console.format_settings(settings)
        )
        text = console.format_settings(settings, "/cfg/quick.yaml")
        self.assertTrue(text.startswith("Config file: /cfg/quick.yaml (resolved"))
        self.assertIn("vippet.base_url", text)


if __name__ == "__main__":
    unittest.main()
