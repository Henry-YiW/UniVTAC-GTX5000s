"""tactileVLA (RTac1 TactileVLA baseline) policy for UniVTAC evaluation.

The model needs the RTac1 repo's own Python environment (pinned torch, patched
transformers), so ``Policy`` runs it in ``tactilevla_server.py`` under that
environment's interpreter and exchanges observations and actions over
localhost HTTP, the same way ``policy/smolvla`` does.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_TACTILEVLA_DIR = Path(__file__).resolve().parent
sys.path.append(str(_TACTILEVLA_DIR.parent))  # for the relative _base_policy import

from .._base_policy import BasePolicy  # noqa: E402

# Prompts must match the ones the training data was built with.
_TRAINING_INSTRUCTIONS = {
    "grasp_classify": "use grasped tool for tactile sensing to move to target surface.",
    "insert_HDMI": "insert the HDMI to the fixed slot.",
    "insert_hole": "insert the stick to the hole.",
    "insert_tube": "insert the tube to the fixed slot.",
    "lift_can": "grasp the can and lifts it vertically without slippage.",
    "lift_bottle": "grasp the bottle and lift it vertically, keeping its final base within 5 cm of the wall.",
    "pull_out_key": "pull out the key.",
    "put_bottle_in_shelf": "grasp the bottle, then position it into the shelf cavity.",
}

# chunk[0] is a placeholder for the current step; the first executable action is chunk[1].
_CHUNK_OFFSET = 1
_IMAGE_SIZE = 224


# ---------------------------------------------------------------------------
# RTac1 <-> UniVTAC action layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ActionMapping:
    arm_slice: tuple[int, int] | None = None
    gripper_index: int | None = None
    gripper_slot_idx: int = 28


def _canonicalize_action_joint_rep(rep: str | None) -> str | None:
    if rep is None:
        return None
    normalized = str(rep).strip().lower()
    if normalized in {"abs", "absolute"}:
        return "absolute"
    if normalized in {"relative", "mix"}:
        return normalized
    return None


def _infer_layout(action_dim: int) -> str:
    from openpi.models_pytorch.rtac1_model_config import RTAC1_RESERVED_ACTION_DIM
    from openpi.models_pytorch.rtac1_model_config import RTAC1_SINGLE_ARM_ACTION_REP_DIM

    single = RTAC1_SINGLE_ARM_ACTION_REP_DIM
    dual = 2 * single + 9
    if action_dim == single:
        return "single48"
    if action_dim in {dual, dual + RTAC1_RESERVED_ACTION_DIM}:
        return "dual105"
    if action_dim == 91:
        return "legacy91"
    return "unknown"


def _build_state(qpos8: np.ndarray, state_dim: int, layout: str, mapping: _ActionMapping) -> np.ndarray:
    state = np.zeros((1, state_dim), dtype=np.float32)
    if layout in {"single48", "dual105"}:
        state[0, 9:16] = qpos8[:7]
        state[0, 16 + mapping.gripper_slot_idx] = qpos8[7]
    elif layout == "legacy91" or state_dim > 9 + mapping.gripper_slot_idx:
        state[0, 9 + mapping.gripper_slot_idx] = qpos8[7]
    return state


def _extract_action8(action_vec: np.ndarray, layout: str, mapping: _ActionMapping) -> np.ndarray:
    out = np.zeros(8, dtype=np.float32)
    if mapping.arm_slice is not None and mapping.gripper_index is not None:
        start, end = mapping.arm_slice
        out[:7] = action_vec[start:end]
        out[7] = action_vec[mapping.gripper_index]
    elif layout in {"single48", "dual105"}:
        out[:7] = action_vec[9:16]
        out[7] = action_vec[16 + mapping.gripper_slot_idx]
    elif layout == "legacy91":
        out[7] = action_vec[9 + mapping.gripper_slot_idx]
    else:
        raise ValueError(
            f"Unsupported action_dim={action_vec.shape[0]}; set arm_slice and gripper_index in deploy.yml"
        )
    return out


def _to_absolute_qpos(action8: np.ndarray, qpos8_base: np.ndarray, action_rep: str) -> np.ndarray:
    if action_rep == "absolute":
        return action8.copy()
    if action_rep == "relative":
        return qpos8_base + action8
    # mix: arm joints are deltas, the gripper target is absolute.
    resolved = action8.copy()
    resolved[:7] += qpos8_base[:7]
    return resolved


# ---------------------------------------------------------------------------
# Observation helpers (run in the IsaacLab process)
# ---------------------------------------------------------------------------

def _to_numpy_uint8(img: Any) -> np.ndarray:
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC image, got {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 255.0)
            if arr.max() <= 1.0:
                arr = arr * 255.0
        arr = arr.astype(np.uint8)
    return arr


def _resize(img: np.ndarray) -> np.ndarray:
    import cv2

    if img.shape[:2] == (_IMAGE_SIZE, _IMAGE_SIZE):
        return img
    return cv2.resize(img, (_IMAGE_SIZE, _IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Model side (runs in the RTac1 environment, see tactilevla_server.py)
# ---------------------------------------------------------------------------

class _TactileVLALocalPolicy:
    """Loads an RTac1 checkpoint and turns observations into 8D qpos targets.

    Every step runs one inference and returns a temporal ensemble over all
    chunks whose first ``chunk_first_n`` executable actions cover the step.
    """

    def __init__(self, args: dict):
        sys.path.insert(0, str(Path(args["tactilevla_repo"]).expanduser()))
        from openpi.policies.rtac1_inference_wrapper import RTac1InferenceWrapper

        self.checkpoint_dir = Path(args["checkpoint_dir"]).expanduser()
        self.prompt = str(args.get("prompt") or _TRAINING_INSTRUCTIONS[args["task_name"]])
        self.chunk_first_n = max(0, int(args.get("chunk_first_n", 20)))
        self.ensemble_k = float(args.get("ensemble_k", 0.01))
        self.tactile_key = str(args.get("tactile_key", "right_tactile_gripper"))
        self.tactile_sensor = str(args.get("tactile_sensor", "GelSightMini"))
        self.action_rep = self._resolve_action_rep(str(args.get("action_rep", "auto")))

        arm_slice = None
        if args.get("arm_slice"):
            start, end = (int(v) for v in str(args["arm_slice"]).split(":"))
            if end - start != 7:
                raise ValueError(f"arm_slice must span 7 dims, got {args['arm_slice']!r}")
            arm_slice = (start, end)
        self.mapping = _ActionMapping(
            arm_slice=arm_slice,
            gripper_index=int(args["gripper_index"]) if args.get("gripper_index") is not None else None,
            gripper_slot_idx=int(args.get("gripper_slot_idx", 28)),
        )

        self.wrapper = RTac1InferenceWrapper(
            checkpoint_dir=str(self.checkpoint_dir),
            domain_name=str(args["domain_name"]),
            device=str(args.get("device", "cuda")),
            num_inference_steps=int(args.get("num_inference_steps", 10)),
        )
        if self.wrapper.is_pi05_baseline():
            raise ValueError("pi05_baseline checkpoints need an action mask; evaluate them with eval_rtac1.py")
        self.action_dim = int(self.wrapper.get_raw_action_dim())
        if int(self.wrapper.get_raw_state_dim()) != self.action_dim:
            raise ValueError(
                f"State/action raw dim mismatch: {self.wrapper.get_raw_state_dim()} vs {self.action_dim}"
            )
        self.layout = _infer_layout(self.action_dim)
        self.use_tactile_input = bool(getattr(self.wrapper.model_config, "use_tactile_input", True))
        print(
            f"[TactileVLA] Loaded {self.checkpoint_dir} (action_dim={self.action_dim}, layout={self.layout}, "
            f"action_rep={self.action_rep}, chunk_first_n={self.chunk_first_n}, prompt={self.prompt!r})",
            flush=True,
        )
        self.reset()

    def _resolve_action_rep(self, configured: str) -> str:
        if configured == "auto":
            with open(self.checkpoint_dir / "train_config.json", "r") as f:
                configured = json.load(f).get("action_joint_rep")
        action_rep = _canonicalize_action_joint_rep(configured)
        if action_rep is None:
            raise ValueError(f"Unsupported action_rep {configured!r}; set action_rep in deploy.yml")
        return action_rep

    def reset(self) -> None:
        self._history: list[tuple[np.ndarray, int, np.ndarray]] = []
        self._step = 0

    def _infer_chunk(self, observation: dict, qpos8: np.ndarray) -> np.ndarray:
        images = {"camera_ego_rgb_0": observation["head_rgb"]}
        if observation.get("wrist_rgb") is not None:
            images["right_wrist_camera_rgb_0"] = observation["wrist_rgb"]

        tactile_kwargs = {}
        if self.use_tactile_input and observation.get("tactile") is not None:
            # Tactile input must be float32 in [0, 255]: uint8 gets truncated in place during normalization.
            tactile_kwargs = {
                "tactiles": {self.tactile_key: observation["tactile"][None].astype(np.float32)},
                "tactile_function_areas": {self.tactile_key: [0, 1]},
                "tactile_sensors": {self.tactile_key: self.tactile_sensor},
            }

        # The wrapper prints a timing line on every call.
        with contextlib.redirect_stdout(io.StringIO()):
            chunk = self.wrapper.infer(
                images=images,
                state=_build_state(qpos8, self.action_dim, self.layout, self.mapping),
                prompt=self.prompt,
                **tactile_kwargs,
            )
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != self.action_dim or chunk.shape[0] <= _CHUNK_OFFSET:
            raise ValueError(f"Unexpected RTac1 chunk shape {chunk.shape}, expected (H>1, {self.action_dim})")
        return chunk

    def get_action(self, observation: dict) -> np.ndarray:
        qpos8 = np.asarray(observation["qpos"], dtype=np.float32).reshape(8)
        self._history.append((self._infer_chunk(observation, qpos8), self._step, qpos8))

        kept, candidates = [], []
        for chunk, infer_step, qpos8_base in self._history:
            idx = self._step - infer_step + _CHUNK_OFFSET
            if idx >= chunk.shape[0] or (self.chunk_first_n > 0 and idx > self.chunk_first_n):
                continue
            kept.append((chunk, infer_step, qpos8_base))
            # Delayed predictions resolve against the qpos their chunk was inferred from, not the current one.
            action8 = _extract_action8(chunk[idx], self.layout, self.mapping)
            candidates.append(_to_absolute_qpos(action8, qpos8_base, self.action_rep))
        self._history = kept
        self._step += 1

        weights = np.exp(-self.ensemble_k * np.arange(len(candidates), dtype=np.float32))
        weights /= weights.sum()
        return (np.stack(candidates) * weights[:, None]).sum(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# HTTP client used by the IsaacLab evaluation process
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _to_wire_payload(value: Any):
    """Convert tensors/arrays to JSON-safe containers for cross-env HTTP."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if isinstance(value, dict):
        return {str(k): _to_wire_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wire_payload(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return _to_wire_payload(np.asarray(value))


def _from_wire_payload(value: Any):
    """Restore JSON-safe ndarray payloads on the receiving side."""
    if isinstance(value, dict):
        if value.get("__ndarray__") is True:
            data = base64.b64decode(value["data"].encode("ascii"))
            array = np.frombuffer(data, dtype=np.dtype(value["dtype"]))
            return array.reshape(value["shape"]).copy()
        return {k: _from_wire_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_wire_payload(v) for v in value]
    return value


class _TactileVLAClient:
    def __init__(self, args: dict):
        self.repo_root = _TACTILEVLA_DIR.parents[1]
        tactilevla_repo = Path(args["tactilevla_repo"]).expanduser()
        self.host = str(args.get("tactilevla_host", os.environ.get("TACTILEVLA_HOST", "127.0.0.1")))
        self.port = int(args.get("tactilevla_port", os.environ.get("TACTILEVLA_PORT", 0)) or 0)
        self.python = Path(args.get("tactilevla_python") or tactilevla_repo / ".venv" / "bin" / "python").expanduser()
        self.startup_timeout = float(args.get("tactilevla_startup_timeout", 300))
        self.request_timeout = float(args.get("tactilevla_request_timeout", 300))
        self.authkey = str(args.get("tactilevla_authkey", os.environ.get("TACTILEVLA_AUTHKEY", "")))
        self._external = self.port != 0
        self._process: subprocess.Popen | None = None
        self._closed = False

        if not self._external:
            self.port = _find_free_port()
            self.authkey = secrets.token_hex(16)
            self._start_server()

        self.base_url = f"http://{self.host}:{self.port}"
        self._wait_until_ready()
        self._post_json("/init", {"authkey": self.authkey, "args": _to_wire_payload(args)})

    def _start_server(self) -> None:
        if not self.python.exists():
            raise FileNotFoundError(f"tactileVLA python not found: {self.python}")
        # Isaac Sim's module and library paths must not leak into the model's environment.
        env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"}}
        self._process = subprocess.Popen(
            [
                str(self.python),
                str(_TACTILEVLA_DIR / "tactilevla_server.py"),
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--authkey",
                self.authkey,
            ],
            cwd=str(self.repo_root),
            env=env,
        )

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                response = self._request("GET", "/health")
                if response.get("status") == "ok":
                    return
            except BaseException as exc:
                last_error = exc
                if self._process is not None and self._process.poll() is not None:
                    raise RuntimeError(f"tactileVLA server exited with code {self._process.returncode}") from exc
            time.sleep(0.25)
        raise TimeoutError(f"timed out waiting for tactileVLA server: {last_error}")

    def _request(self, method: str, path: str, data: bytes | None = None, content_type: str | None = None):
        headers = {"X-TactileVLA-Auth": self.authkey}
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"tactileVLA server HTTP {exc.code}: {detail}") from exc

    def _post_json(self, path: str, payload: Any):
        response = self._request(
            "POST",
            path,
            data=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
        )
        if isinstance(response, dict) and response.get("type") == "error":
            raise RuntimeError(f"tactileVLA server error:\n{response.get('error')}\n{response.get('traceback')}")
        return response

    def get_action(self, observation: dict) -> np.ndarray:
        payload = {"authkey": self.authkey, "observation": _to_wire_payload(observation)}
        response = self._post_json("/act", payload)
        return np.asarray(_from_wire_payload(response["action"]), dtype=np.float32)

    def reset(self) -> None:
        self._post_json("/reset", {"authkey": self.authkey})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._external:
                self._post_json("/reset", {"authkey": self.authkey})
            else:
                self._post_json("/shutdown", {"authkey": self.authkey})
        except Exception:
            pass
        if self._process is not None and self._process.poll() is None:
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()


class Policy(BasePolicy):
    """Deploy-time tactileVLA wrapper for UniVTAC via a local inference server."""

    def __init__(self, args: dict):
        self.task_name = args["task_name"]
        with open(_TACTILEVLA_DIR.parent / "task_settings.json", "r") as f:
            camera_type = json.load(f).get(self.task_name, {}).get("camera_type", "head")
        self.use_wrist = camera_type == "all"
        self.force_joint_targets = bool(args.get("force_joint_targets", False))
        self.model = _TactileVLAClient(args)

    def encode_obs(self, observation: dict) -> dict:
        cameras = observation["observation"]
        tactile = observation.get("tactile", {})
        left_key = "left_gsmini" if "left_gsmini" in tactile else "left_tactile"
        right_key = "right_gsmini" if "right_gsmini" in tactile else "right_tactile"
        pads = None
        if left_key in tactile and right_key in tactile:
            # Pad order is [left (thumb), right (index)], as in training.
            pads = np.stack([_resize(_to_numpy_uint8(tactile[key]["rgb_marker"])) for key in (left_key, right_key)])

        qpos8 = observation["embodiment"]["joint"][:8]
        if hasattr(qpos8, "detach"):
            qpos8 = qpos8.detach().cpu().numpy()
        return {
            "head_rgb": _resize(_to_numpy_uint8(cameras["head"]["rgb"])),
            "wrist_rgb": _resize(_to_numpy_uint8(cameras["wrist"]["rgb"])) if self.use_wrist else None,
            "tactile": pads,
            "qpos": np.asarray(qpos8, dtype=np.float32).reshape(8),
        }

    def eval(self, task, observation):
        import torch

        action_np = self.model.get_action(self.encode_obs(observation))
        action_t = torch.from_numpy(action_np).to(task.device).float()
        return task.take_action(action_t, action_type="qpos", force=self.force_joint_targets)

    def reset(self):
        self.model.reset()

    def close(self):
        self.model.close()
