#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np


def _downsample_time_indices(n_frames: int, max_frames: int) -> np.ndarray:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if n_frames <= max_frames:
        return np.arange(n_frames, dtype=np.int64)
    raw = np.linspace(0, n_frames - 1, num=max_frames)
    idx = np.unique(np.round(raw).astype(np.int64))
    if idx[0] != 0:
        idx = np.insert(idx, 0, 0)
    if idx[-1] != n_frames - 1:
        idx = np.append(idx, n_frames - 1)
    return idx


def save_simulation_data_npz(
    filename: str,
    *,
    config_metadata: dict[str, Any],
    setup_description: str,
    stop_reason: str,
    stop_time: float,
    dt: float,
    x_values: np.ndarray,
    t_values: np.ndarray,
    u_num: np.ndarray,
    v_num: np.ndarray,
    max_frames: int,
) -> str:
    idx = _downsample_time_indices(int(t_values.shape[0]), int(max_frames))
    t_saved = t_values[idx]
    u_saved = u_num[:, idx]
    v_saved = v_num[:, idx]
    step_indices = np.round(t_saved / float(dt)).astype(np.int64)

    config_json = json.dumps(config_metadata, sort_keys=True)
    np.savez_compressed(
        filename,
        schema_version=np.asarray(1, dtype=np.int64),
        config_json=np.asarray(config_json),
        setup_description=np.asarray(setup_description),
        stop_reason=np.asarray(stop_reason),
        stop_time=np.asarray(stop_time, dtype=np.float64),
        dt=np.asarray(dt, dtype=np.float64),
        x_values=np.asarray(x_values, dtype=np.float64),
        t_values=np.asarray(t_saved, dtype=np.float64),
        u_num=np.asarray(u_saved, dtype=np.float64),
        v_num=np.asarray(v_saved, dtype=np.float64),
        downsample_indices=np.asarray(idx, dtype=np.int64),
        step_indices=np.asarray(step_indices, dtype=np.int64),
    )
    return filename


def load_simulation_data_npz(filename: str) -> dict:
    with np.load(filename, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    out["schema_version"] = int(out["schema_version"].item())
    out["config"] = json.loads(str(out["config_json"].item()))
    out["setup_description"] = str(out["setup_description"].item())
    out["stop_reason"] = str(out["stop_reason"].item())
    out["stop_time"] = float(out["stop_time"].item())
    out["dt"] = float(out["dt"].item())
    out["config_json"] = str(out["config_json"].item())
    return out

