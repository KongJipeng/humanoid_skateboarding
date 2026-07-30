"""Play an RL agent and plot foot/skateboard heights on exit."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType

import tyro

from mjlab_husky.scripts.play import PlayConfig
from mjlab_husky.tasks.registry import list_tasks, load_rl_cfg


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIM_PATH = PROJECT_ROOT / "test_scene" / "sim.py"
DEFAULT_SCENE = PROJECT_ROOT / "test_scene" / "mjlab_scene.xml"
DEFAULT_POLICY = PROJECT_ROOT / "ckpts" / "test.onnx"
DEFAULT_PLOT = PROJECT_ROOT / "height_plot.png"


class HeightRecorder:
  """Collect world-frame heights after every MuJoCo simulation step."""

  def __init__(self, model):
    self.left_foot_site_id = model.site("robot/left_foot").id
    self.right_foot_site_id = model.site("robot/right_foot").id
    self.skateboard_body_id = model.body("skateboard/skateboard_deck").id
    self.times: list[float] = []
    self.left_foot: list[float] = []
    self.right_foot: list[float] = []
    self.skateboard: list[float] = []

  def record(self, data) -> None:
    sim_time = float(data.time)
    if self.times and sim_time <= self.times[-1]:
      self.clear()

    self.times.append(sim_time)
    self.left_foot.append(float(data.site_xpos[self.left_foot_site_id, 2]))
    self.right_foot.append(float(data.site_xpos[self.right_foot_site_id, 2]))
    self.skateboard.append(float(data.xpos[self.skateboard_body_id, 2]))

  def clear(self) -> None:
    self.times.clear()
    self.left_foot.clear()
    self.right_foot.clear()
    self.skateboard.clear()


def ensure_mjpython_on_macos() -> None:
  """Re-launch through MuJoCo's Cocoa-compatible Python on macOS."""
  if sys.platform != "darwin" or os.environ.get("MJPYTHON_BIN"):
    return

  mjpython = Path(sys.executable).with_name("mjpython")
  if not mjpython.is_file():
    raise FileNotFoundError(
      f"MuJoCo's macOS launcher was not found in the environment: {mjpython}"
    )

  os.execv(
    str(mjpython),
    [str(mjpython), str(Path(__file__).resolve()), *sys.argv[1:]],
  )


def load_lightweight_sim() -> ModuleType:
  """Load test_scene/sim.py without running its command-line entry point."""
  spec = importlib.util.spec_from_file_location("_mjlab_lightweight_sim", SIM_PATH)
  if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load simulation module: {SIM_PATH}")

  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def run_simulation(
  sim: ModuleType,
  checkpoint_path: Path,
  cfg: PlayConfig,
) -> HeightRecorder:
  # pynput must initialize before the Cocoa-backed MuJoCo viewer on macOS.
  listener = threading.Thread(target=sim.start_listener, daemon=True)
  listener.start()

  controller = sim.RealTimePolicyController(
    xml_file=str(DEFAULT_SCENE),
    policy_path=str(checkpoint_path),
    device=cfg.device or "auto",
    policy_frequency=50,
    activation="auto",
    export_onnx=cfg.export_onnx,
  )
  recorder = HeightRecorder(controller.model)

  original_viewer_sync = controller.viewer.sync

  def record_then_sync(*sync_args, **sync_kwargs):
    recorder.record(controller.data)
    return original_viewer_sync(*sync_args, **sync_kwargs)

  # Hook the Python viewer handle rather than replacing mujoco.mj_step.
  # Replacing the native MuJoCo function causes a SIGTRAP under mjpython.
  controller.viewer.sync = record_then_sync
  try:
    controller.run()
  except KeyboardInterrupt:
    print("\n[INFO]: Simulation interrupted; saving the recorded height graph.")
  finally:
    controller.viewer.sync = original_viewer_sync

  return recorder


def save_height_plot(recorder: HeightRecorder, output_path: Path) -> Path:
  if not recorder.times:
    raise RuntimeError("No height samples were recorded; no graph was generated.")

  # Agg produces a PNG without competing with mjpython for the macOS GUI thread.
  import matplotlib

  matplotlib.use("Agg", force=True)
  import matplotlib.pyplot as plt

  output_path = output_path.expanduser().resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)

  figure, axis = plt.subplots(figsize=(10, 5))
  axis.plot(
    recorder.times,
    recorder.left_foot,
    label="Left foot",
    linewidth=1.5,
  )
  axis.plot(
    recorder.times,
    recorder.right_foot,
    label="Right foot",
    linewidth=1.5,
  )
  axis.plot(
    recorder.times,
    recorder.skateboard,
    label="Skateboard",
    linewidth=1.5,
  )
  axis.set(
    title="Foot and Skateboard Heights",
    xlabel="Simulation time (s)",
    ylabel="World-frame height (m)",
  )
  axis.grid(True, alpha=0.3)
  axis.legend()
  figure.tight_layout()
  figure.savefig(output_path, dpi=160)
  plt.close(figure)
  return output_path


def open_plot(output_path: Path) -> None:
  if sys.platform == "darwin":
    command = ["open", str(output_path)]
  elif sys.platform == "win32":
    os.startfile(output_path)  # type: ignore[attr-defined]
    return
  elif shutil.which("xdg-open") and (
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
  ):
    command = ["xdg-open", str(output_path)]
  else:
    print("[INFO]: No graphical image viewer detected; open the PNG manually.")
    return

  try:
    subprocess.Popen(
      command,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      start_new_session=True,
    )
  except OSError as error:
    print(f"[WARN]: Could not open the height graph automatically: {error}")


def run_play_graph(task_id: str, cfg: PlayConfig) -> None:
  checkpoint_path = (
    Path(cfg.checkpoint_file).expanduser().resolve()
    if cfg.checkpoint_file is not None
    else DEFAULT_POLICY
  )
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Policy file not found: {checkpoint_path}")
  if not DEFAULT_SCENE.is_file():
    raise FileNotFoundError(f"MuJoCo XML file not found: {DEFAULT_SCENE}")

  print(f"[INFO]: Task: {task_id}")
  print(f"[INFO]: Loading checkpoint: {checkpoint_path}")
  ensure_mjpython_on_macos()
  sim = load_lightweight_sim()
  recorder = run_simulation(sim, checkpoint_path, cfg)
  output_path = save_height_plot(recorder, DEFAULT_PLOT)
  print(f"[INFO]: Height graph saved to: {output_path}")
  open_plot(output_path)


def main() -> None:
  # Match play.py: parse the registered task first, then the PlayConfig flags.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  agent_cfg = load_rl_cfg(chosen_task)
  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args, agent_cfg

  run_play_graph(chosen_task, args)


if __name__ == "__main__":
  main()
