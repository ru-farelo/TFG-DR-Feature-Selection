import time
import threading
from typing import Dict

try:
    from codecarbon import EmissionsTracker
    _HAS_CODECARBON = True
except Exception:
    EmissionsTracker = None
    _HAS_CODECARBON = False

_task_trackers = {}
_task_emissions = {}
_lock = threading.Lock()
_enabled = False  # Global flag to enable/disable tracking


def set_tracking_enabled(enabled: bool):
    """Enable or disable carbon tracking globally."""
    global _enabled
    _enabled = enabled
    if enabled and not _HAS_CODECARBON:
        print("  Warning: CodeCarbon tracking enabled but codecarbon package not installed!")
        print("   Install with: pip install codecarbon")


def is_tracking_enabled() -> bool:
    """Check if carbon tracking is enabled."""
    return _enabled and _HAS_CODECARBON


def _safe_get_emissions_kg(tracker) -> float:
    """Try several ways to obtain emissions (kg) from a tracker instance."""
    if tracker is None:
        return 0.0

    # Try documented/public methods/attributes in a robust order
    try:
        # get_emissions_data may return a dataframe with 'emissions' column
        df = tracker.get_emissions_data()  # type: ignore
        try:
            import pandas as _pd

            if isinstance(df, _pd.DataFrame) and "emissions" in df.columns:
                val = float(df["emissions"].sum())
                return val
        except Exception:
            pass
    except Exception:
        pass

    for attr in ("last_emissions", "emissions", "final_emissions", "_last_emissions", "_total_emissions"):
        if hasattr(tracker, attr):
            try:
                v = getattr(tracker, attr)
                if v is None:
                    continue
                return float(v)
            except Exception:
                continue

    # Fallback: return 0.0 if we couldn't obtain a value
    return 0.0


def start_task(task_name: str):
    """Start a CodeCarbon task and keep the tracker instance.

    If CodeCarbon isn't available or tracking is disabled, this becomes a no-op.
    
    Uses measure_power_secs=1 to track only the Python process CPU/GPU,
    not the entire system.
    """
    if not is_tracking_enabled():
        return

    with _lock:
        try:
            # Configure tracker to measure only this process (not entire system)
            tracker = EmissionsTracker(
                project_name=task_name,
                measure_power_secs=1,  # Measure every second for accuracy
                save_to_file=False,     # Don't save intermediate CSV files
                save_to_api=False,      # Don't send to CodeCarbon API
                log_level="error",      # Reduce noise in logs
                tracking_mode="process" # Track only this Python process
            )
            tracker.start()
            _task_trackers[task_name] = tracker
        except Exception:
            # If starting a tracker fails, ensure it doesn't crash the pipeline
            _task_trackers[task_name] = None


def end_task(task_name: str):
    """Stop the tracker for `task_name` and record emissions in grams.

    Returns the emissions for the task in grams (float). If tracking isn't
    enabled or available, returns 0.0.
    """
    if not is_tracking_enabled():
        return 0.0

    with _lock:
        tracker = _task_trackers.pop(task_name, None)

    if tracker is None:
        return 0.0

    try:
        tracker.stop()
    except Exception:
        pass

    kg = _safe_get_emissions_kg(tracker)
    grams = float(kg) * 1000.0

    with _lock:
        _task_emissions[task_name] = _task_emissions.get(task_name, 0.0) + grams

    return grams


def get_and_reset_emissions() -> Dict[str, float]:
    """Return the recorded emissions per task in grams and reset the store."""
    with _lock:
        copy = dict(_task_emissions)
        _task_emissions.clear()
    return copy


def has_codecarbon() -> bool:
    return _HAS_CODECARBON
