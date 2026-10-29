from __future__ import annotations

import app.config as app_config


def test_project_dotenv_can_enable_pad_over_yaml_default(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("PAD_ENABLED=true\n", encoding="utf-8")
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("cv:\n  pad_enabled: false\n", encoding="utf-8")

    monkeypatch.setattr(app_config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_config, "CONFIG_YAML", yaml_path)
    monkeypatch.delenv("PAD_ENABLED", raising=False)
    app_config.get_settings.cache_clear()
    try:
        settings = app_config.get_settings()
        assert settings.pad_enabled is True
    finally:
        app_config.get_settings.cache_clear()


def test_process_environment_overrides_project_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("PAD_ENABLED=true\n", encoding="utf-8")
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("cv:\n  pad_enabled: false\n", encoding="utf-8")

    monkeypatch.setattr(app_config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_config, "CONFIG_YAML", yaml_path)
    monkeypatch.setenv("PAD_ENABLED", "false")
    app_config.get_settings.cache_clear()
    try:
        settings = app_config.get_settings()
        assert settings.pad_enabled is False
    finally:
        app_config.get_settings.cache_clear()


def test_configs_dotenv_can_enable_pad_when_project_dotenv_is_missing(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / ".env").write_text("PAD_ENABLED=true\n", encoding="utf-8")
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("cv:\n  pad_enabled: false\n", encoding="utf-8")

    monkeypatch.setattr(app_config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_config, "CONFIG_YAML", yaml_path)
    monkeypatch.delenv("PAD_ENABLED", raising=False)
    app_config.get_settings.cache_clear()
    try:
        settings = app_config.get_settings()
        assert settings.pad_enabled is True
    finally:
        app_config.get_settings.cache_clear()


def test_project_dotenv_overrides_configs_dotenv(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / ".env").write_text("PAD_ENABLED=true\n", encoding="utf-8")
    (tmp_path / ".env").write_text("PAD_ENABLED=false\n", encoding="utf-8")
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("cv:\n  pad_enabled: true\n", encoding="utf-8")

    monkeypatch.setattr(app_config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_config, "CONFIG_YAML", yaml_path)
    monkeypatch.delenv("PAD_ENABLED", raising=False)
    app_config.get_settings.cache_clear()
    try:
        settings = app_config.get_settings()
        assert settings.pad_enabled is False
    finally:
        app_config.get_settings.cache_clear()
