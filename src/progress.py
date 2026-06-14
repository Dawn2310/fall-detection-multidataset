"""
progress.py — Compact logger for run_experiments when running overnight.
- Print timestamp + elapsed time since start.
- Print overall progress (variant X/N, model Y/M).
- flush=True to update log file in real-time.
"""
import time
from datetime import datetime, timedelta


_START_TIME: float | None = None
_CTX = {
    "stage":      "",   # e.g.: "INTRA" / "CROSS"
    "variant":    "",
    "var_idx":    0,
    "var_total":  0,
    "model":      "",
    "mdl_idx":    0,
    "mdl_total":  0,
}


def start() -> None:
    """Start counting total time."""
    global _START_TIME
    _START_TIME = time.time()
    log(f"=== PIPELINE STARTED @ {datetime.now():%Y-%m-%d %H:%M:%S} ===")


def set_stage(stage: str, var_total: int = 0, mdl_total: int = 0) -> None:
    _CTX["stage"] = stage
    _CTX["var_total"] = var_total
    if mdl_total:
        _CTX["mdl_total"] = mdl_total


def set_variant(name: str, idx: int) -> None:
    _CTX["variant"] = name
    _CTX["var_idx"] = idx


def set_model(name: str, idx: int) -> None:
    _CTX["model"] = name
    _CTX["mdl_idx"] = idx


def _elapsed_str() -> str:
    if _START_TIME is None:
        return "00:00:00"
    secs = int(time.time() - _START_TIME)
    return str(timedelta(seconds=secs))


def _prefix() -> str:
    now = datetime.now().strftime("%H:%M:%S")
    parts = [f"[{now}", f"elapsed {_elapsed_str()}"]
    if _CTX["stage"]:
        parts.append(_CTX["stage"])
    if _CTX["var_total"]:
        parts.append(f"var {_CTX['var_idx']}/{_CTX['var_total']} {_CTX['variant']}")
    if _CTX["mdl_total"]:
        parts.append(f"mdl {_CTX['mdl_idx']}/{_CTX['mdl_total']} {_CTX['model']}")
    return " | ".join(parts) + "]"


def log(msg: str = "") -> None:
    """Print log with timestamp + immediate flush."""
    if msg == "":
        print("", flush=True)
        return
    print(f"{_prefix()} {msg}", flush=True)


def banner(msg: str, char: str = "#") -> None:
    line = char * 70
    print("", flush=True)
    print(line, flush=True)
    print(f"{_prefix()} {msg}", flush=True)
    print(line, flush=True)
