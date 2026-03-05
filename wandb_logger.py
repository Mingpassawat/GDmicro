import logging
import os
import importlib
from typing import Dict, Optional, Any

LOGGER = logging.getLogger(__name__)

_WANDB = None
_WANDB_RUN = None
_ENABLED = False
_BASE_PROJECT = "GDmicro"
_BASE_ENTITY = None
_BASE_MODE = "online"
_BASE_RUN_NAME = None
_BASE_CONFIG = {}
_BASE_TAGS = []
_GROUP_NAME = None


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def init_wandb(
    enabled: bool = True,
    project: str = "GDmicro",
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    mode: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    group_name: Optional[str] = None,
    create_run: bool = True,
):
    global _WANDB, _WANDB_RUN, _ENABLED
    global _BASE_PROJECT, _BASE_ENTITY, _BASE_MODE, _BASE_RUN_NAME, _BASE_CONFIG, _BASE_TAGS, _GROUP_NAME

    _ENABLED = _to_bool(enabled, default=True)
    if not _ENABLED:
        LOGGER.info("W&B disabled by configuration.")
        return False

    try:
        wandb_lib = importlib.import_module('wandb')
    except Exception as exc:
        LOGGER.warning("W&B is enabled but package is unavailable (%s). Continuing without W&B.", exc)
        _ENABLED = False
        return False

    _WANDB = wandb_lib

    run_mode = mode or os.environ.get("WANDB_MODE", "online")
    _BASE_PROJECT = project
    _BASE_ENTITY = entity
    _BASE_MODE = run_mode
    _BASE_RUN_NAME = run_name
    _BASE_CONFIG = config or {}
    _BASE_TAGS = tags or []
    _GROUP_NAME = group_name or os.environ.get("WANDB_RUN_GROUP")

    if not create_run:
        LOGGER.info("W&B prepared | project=%s mode=%s create_run=%s", project, run_mode, str(create_run))
        return True

    try:
        _WANDB_RUN = _WANDB.init(
            project=project,
            entity=entity,
            name=run_name,
            group=_GROUP_NAME,
            mode=run_mode,
            config=config or {},
            tags=tags or [],
            reinit=True,
        )
        LOGGER.info("W&B initialized | project=%s mode=%s run_name=%s", project, run_mode, run_name)
        return True
    except Exception as exc:
        LOGGER.warning("Failed to initialize W&B (%s). Continuing without W&B.", exc)
        _WANDB_RUN = None
        _ENABLED = False
        return False


def start_fold_run(fold: int, total_folds: int, extra_config: Optional[Dict[str, Any]] = None):
    global _WANDB_RUN, _GROUP_NAME
    if not _ENABLED or _WANDB is None:
        return False

    try:
        if _WANDB_RUN is not None:
            _WANDB.finish()
            _WANDB_RUN = None

        if not _GROUP_NAME:
            _GROUP_NAME = f"cv-{_WANDB.util.generate_id()}"

        run_base_name = _BASE_RUN_NAME if _BASE_RUN_NAME else "GDmicro"
        run_name = f"{run_base_name}-fold-{int(fold)}"
        run_config = dict(_BASE_CONFIG)
        run_config.update({
            "fold": int(fold),
            "total_folds": int(total_folds),
        })
        if extra_config:
            run_config.update(extra_config)

        _WANDB_RUN = _WANDB.init(
            project=_BASE_PROJECT,
            entity=_BASE_ENTITY,
            name=run_name,
            group=_GROUP_NAME,
            job_type=f"fold-{int(fold)}",
            mode=_BASE_MODE,
            config=run_config,
            tags=list(_BASE_TAGS) + [f"fold-{int(fold)}"],
            reinit=True,
        )
        LOGGER.info("W&B fold run initialized | group=%s run=%s", _GROUP_NAME, run_name)
        return True
    except Exception as exc:
        LOGGER.warning("Failed to start W&B fold run (%s).", exc)
        return False


def is_enabled() -> bool:
    return bool(_ENABLED and _WANDB_RUN is not None)


def log(metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
    if not is_enabled():
        return
    try:
        if step is None:
            _WANDB.log(metrics, commit=commit)
        else:
            _WANDB.log(metrics, step=step, commit=commit)
    except Exception as exc:
        LOGGER.warning("W&B log failed (%s).", exc)


def summary_update(values: Dict[str, Any]):
    if not is_enabled():
        return
    try:
        for key, value in values.items():
            _WANDB_RUN.summary[key] = value
    except Exception as exc:
        LOGGER.warning("W&B summary update failed (%s).", exc)


def save_file(file_path: str):
    if not is_enabled():
        return
    try:
        _WANDB.save(file_path, policy='now')
    except Exception as exc:
        LOGGER.warning("W&B save file failed (%s).", exc)


def finish():
    global _WANDB_RUN
    if not is_enabled():
        return
    try:
        _WANDB.finish()
    except Exception as exc:
        LOGGER.warning("W&B finish failed (%s).", exc)
    finally:
        _WANDB_RUN = None
