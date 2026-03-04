import logging
import os
import importlib
from typing import Dict, Optional, Any

LOGGER = logging.getLogger(__name__)

_WANDB = None
_WANDB_RUN = None
_ENABLED = False


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
):
    global _WANDB, _WANDB_RUN, _ENABLED

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

    try:
        _WANDB_RUN = _WANDB.init(
            project=project,
            entity=entity,
            name=run_name,
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
