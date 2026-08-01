"""Configuration loader for MedQA-USMLE.
Reads reproducibility.yaml and returns a structured dict.
Supports ${ENV_VAR} substitution for API keys.
"""

import copy
import os
import re
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "reproducibility.yaml"


def _substitute_env_vars(value):
    """Replace ${ENV_VAR} patterns with environment variable values."""
    if isinstance(value, str):
        def _replace(match):
            env_var = match.group(1)
            return os.environ.get(env_var, match.group(0))
        return re.sub(r"\$\{(\w+)\}", _replace, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(v) for v in value]
    return value


def load_config(path: str | Path | None = None) -> dict:
    path = path or CONFIG_PATH
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Try loading .env files before substitution so env vars are available
    _load_dotenv_for_config(config)

    config = _substitute_env_vars(config)
    return config


def deep_merge(base: dict, override: dict | None = None) -> dict:
    """Deep-merge an override dict onto a base config.

    Nested dicts are merged recursively (override wins per key), so a partial
    override such as ``{"model": {"name": "..."}}`` keeps every other section
    (judge_model, embedding, rag, ...) from the base config intact.

    Args:
        base: Base configuration dict (already env-substituted).
        override: Partial override dict, or None.

    Returns:
        A new merged dict (the base is not mutated).
    """
    if not override:
        return base
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_dotenv_for_config(config: dict):
    """Load env vars referenced in config from .env files if not in environment."""
    import re

    def _collect_missing(config_obj, missing_set):
        if isinstance(config_obj, str):
            for m in re.finditer(r"\$\{(\w+)\}", config_obj):
                var = m.group(1)
                if var not in os.environ:
                    missing_set.add(var)
        elif isinstance(config_obj, dict):
            for v in config_obj.values():
                _collect_missing(v, missing_set)
        elif isinstance(config_obj, list):
            for v in config_obj:
                _collect_missing(v, missing_set)

    missing = set()
    _collect_missing(config, missing)
    if not missing:
        return

    env_files = [
        Path.home() / ".hermes" / ".env",
        Path.cwd() / ".env",
    ]

    for env_path in env_files:
        if not env_path.exists():
            continue
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                if key in missing:
                    # Strip quotes and whitespace
                    val = val.strip().strip("'\"")
                    os.environ[key] = val
                    missing.discard(key)
