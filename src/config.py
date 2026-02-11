"""Load config.yaml and provide typed access to settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    id: str = "openai/gpt-4o"
    max_tokens: int = 4096
    temperature: float = 0.7
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    evaluator_model: Optional[str] = None
    reflector_model: Optional[str] = None


class KnowledgeConfig(BaseModel):
    root_dir: str = "./knowledge"
    rga_context_lines: int = 6
    rga_max_matches: int = 4


class ReflexionConfig(BaseModel):
    max_trials: int = 7
    min_success_score: float = 0.82
    reflection_top_k: int = 4


class ToTConfig(BaseModel):
    branch_factor: int = 4
    max_depth: int = 5
    beam_width: int = 3
    search_algo: str = "bfs"


class DbConfig(BaseModel):
    path: str = "./data/reflexion.db"


class AppConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    knowledge: KnowledgeConfig = KnowledgeConfig()
    reflexion: ReflexionConfig = ReflexionConfig()
    tot: ToTConfig = ToTConfig()
    db: DbConfig = DbConfig()


_config: AppConfig | None = None


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load config from YAML file. Cached after first load."""
    global _config
    if _config is not None:
        return _config

    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        _config = AppConfig(**raw)
    else:
        _config = AppConfig()

    # litellm env vars — if api_base/api_key set in config, propagate
    if _config.model.api_base:
        os.environ.setdefault("OPENAI_API_BASE", _config.model.api_base)
    if _config.model.api_key:
        os.environ.setdefault("OPENAI_API_KEY", _config.model.api_key)

    return _config


def get_config() -> AppConfig:
    """Get already-loaded config (loads if needed)."""
    return load_config()
