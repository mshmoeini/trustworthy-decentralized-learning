"""Tests for YAML configuration loading."""

from pathlib import Path

import pytest

from tdl.config import load_config


def test_load_config_reads_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\nepochs: 3\n", encoding="utf-8")

    assert load_config(config_path) == {"seed": 42, "epochs": 3}


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(config_path)
