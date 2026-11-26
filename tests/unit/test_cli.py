"""Smoke tests for the CLI scaffold: help, version and subcommand registration."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from elevate import __version__
from elevate.cli import NOT_IMPLEMENTED_EXIT_CODE, app

runner = CliRunner()

SUBCOMMANDS = ["verify-setup", "exit-check", "run", "experiment", "e2e"]


def test_help_lists_every_subcommand() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in SUBCOMMANDS:
        assert name in result.output


def test_version_flag_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["verify-setup"],
        ["exit-check", "--phase", "0"],
        ["run", "V4"],
        ["experiment", "configs/experiments/E1.yaml"],
        ["e2e", "--tiny"],
    ],
)
def test_stub_subcommands_exit_non_zero(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == NOT_IMPLEMENTED_EXIT_CODE


def test_exit_check_rejects_unknown_phase() -> None:
    result = runner.invoke(app, ["exit-check", "--phase", "9"])
    assert result.exit_code != 0
    assert result.exit_code != NOT_IMPLEMENTED_EXIT_CODE
