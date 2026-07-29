import argparse
import math
import re
import time
from pathlib import Path
import numpy as np
import mujoco
import torch
from rich import print
from collections import deque
import mujoco.viewer as mjv
from tqdm import tqdm
import os

try:
    import onnxruntime as ort
except ImportError:
    ort = None

class OnnxPolicyWrapper:
    """Minimal wrapper so ONNXRuntime policies mimic TorchScript call signature."""

    def __init__(self, session, input_name, output_index=0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        result = outputs[self.output_index]
        if not isinstance(result, np.ndarray):
            result = np.asarray(result, dtype=np.float32)
        return torch.from_numpy(result.astype(np.float32))


def _activation_module(name: str) -> torch.nn.Module:
    activations = {
        "elu": torch.nn.ELU,
        "selu": torch.nn.SELU,
        "relu": torch.nn.ReLU,
        "crelu": torch.nn.CELU,
        "lrelu": torch.nn.LeakyReLU,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "softplus": torch.nn.Softplus,
        "gelu": torch.nn.GELU,
        "swish": torch.nn.SiLU,
        "mish": torch.nn.Mish,
        "identity": torch.nn.Identity,
    }
    key = name.lower()
    if key not in activations:
        valid = ", ".join(sorted(activations))
        raise ValueError(f"Unsupported activation '{name}'. Expected one of: {valid}")
    return activations[key]()


def _resolve_checkpoint_activation(checkpoint_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested

    agent_cfg_path = checkpoint_path.parent / "params" / "agent.yaml"
    if agent_cfg_path.exists():
        text = agent_cfg_path.read_text(encoding="utf-8")
        match = re.search(r"(?m)^\s+activation:\s*([A-Za-z0-9_]+)\s*$", text)
        if match:
            return match.group(1)

    print(
        "[yellow]Could not determine the checkpoint activation from "
        f"{agent_cfg_path}; falling back to ELU.[/yellow]"
    )
    return "elu"


class TorchCheckpointPolicy(torch.nn.Module):
    """Rebuild a feed-forward RSL-RL actor from a training checkpoint."""

    NORMALIZATION_EPS = 1e-2

    def __init__(self, checkpoint_path: str, device: str, activation: str = "auto"):
        super().__init__()
        path = Path(checkpoint_path)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(
                f"{path} is not an RSL-RL checkpoint containing model_state_dict."
            )
        state = checkpoint["model_state_dict"]
        layer_indices = sorted(
            {
                int(match.group(1))
                for key in state
                if (match := re.fullmatch(r"actor\.(\d+)\.weight", key))
            }
        )
        if not layer_indices:
            raise ValueError(f"No feed-forward actor layers were found in {path}.")

        activation_name = _resolve_checkpoint_activation(path, activation)
        layers = []
        for position, layer_index in enumerate(layer_indices):
            weight = state[f"actor.{layer_index}.weight"]
            bias = state[f"actor.{layer_index}.bias"]
            layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(bias)
            layers.append(layer)
            if position < len(layer_indices) - 1:
                layers.append(_activation_module(activation_name))

        self.actor = torch.nn.Sequential(*layers)
        self.input_dim = int(state[f"actor.{layer_indices[0]}.weight"].shape[1])
        self.output_dim = int(state[f"actor.{layer_indices[-1]}.weight"].shape[0])

        mean = state.get("actor_obs_normalizer._mean")
        std = state.get("actor_obs_normalizer._std")
        if mean is None or std is None:
            mean = torch.zeros(1, self.input_dim)
            std = torch.ones(1, self.input_dim)
            print(
                "[yellow]Checkpoint has no actor observation normalizer; "
                "using identity normalization.[/yellow]"
            )
        self.register_buffer("obs_mean", mean.detach().clone())
        self.register_buffer("obs_std", std.detach().clone())
        self.to(device)
        self.eval()

        iteration = checkpoint.get("iter", "unknown")
        print(
            f"PyTorch policy loaded from {path} "
            f"(iteration={iteration}, activation={activation_name}, "
            f"input={self.input_dim}, actions={self.output_dim}, device={device})"
        )

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        normalized = (obs_tensor - self.obs_mean) / (
            self.obs_std + self.NORMALIZATION_EPS
        )
        return self.actor(normalized)

    def export_onnx(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros(
            1, self.input_dim, device=self.obs_mean.device, dtype=self.obs_mean.dtype
        )
        torch.onnx.export(
            self,
            dummy,
            str(output_path),
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        print(f"ONNX policy exported to {output_path}")


def resolve_device(device: str) -> str:
    """Resolve a portable inference device for the lightweight ONNX demo."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[yellow]CUDA is unavailable; falling back to CPU.[/yellow]")
        return "cpu"
    return device


def load_onnx_policy(policy_path: str, device: str) -> OnnxPolicyWrapper:
    if ort is None:
        raise ImportError("onnxruntime is required for ONNX policy inference but is not installed.")
    providers = []
    available = ort.get_available_providers()
    if device.startswith('cuda'):
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        else:
            print("CUDAExecutionProvider not available in onnxruntime; falling back to CPUExecutionProvider.")
    providers.append('CPUExecutionProvider')
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"ONNX policy loaded from {policy_path} using providers: {session.get_providers()}")
    return OnnxPolicyWrapper(session, input_name)


def load_policy(
    policy_path: str,
    device: str,
    activation: str = "auto",
    export_onnx: bool = False,
):
    suffix = Path(policy_path).suffix.lower()
    if suffix == ".onnx":
        if export_onnx:
            raise ValueError("--export-onnx is only valid for a .pt checkpoint.")
        return load_onnx_policy(policy_path, device)
    if suffix in {".pt", ".pth"}:
        policy = TorchCheckpointPolicy(policy_path, device, activation)
        if export_onnx:
            policy.export_onnx(Path(policy_path).with_suffix(".onnx"))
        return policy
    raise ValueError(
        f"Unsupported policy format '{suffix}'. Expected .onnx, .pt, or .pth."
    )


from pynput import keyboard
import threading

reset_flag = False 
pause_flag = False
V_MIN, V_MAX = 0.0, 1.5
H_MIN, H_MAX = -math.pi / 4, math.pi / 4
v = 0.0
h = 0.0


def wrap_to_pi(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def on_press(key):
    global v, h, reset_flag, pause_flag
    try:
        if key == keyboard.Key.up:
            v = round(min(v + 0.1, V_MAX), 1)
            print("v =", v, "h =", round(h, 3), "(rad)")
        elif key == keyboard.Key.down:
            v = round(max(v - 0.1, V_MIN), 1)
            print("v =", v, "h =", round(h, 3), "(rad)")
        elif key == keyboard.Key.left:
            h = round(max(h + 0.1, H_MIN), 2)
            print("v =", v, "h =", round(h, 3), "(rad)")
        elif key == keyboard.Key.right:
            h = round(min(h - 0.1, H_MAX), 2)
            print("v =", v, "h =", round(h, 3), "(rad)")
        elif key == keyboard.Key.enter:
            reset_flag = True
            print("Reset flag set! Simulation will reset...")
        elif key == keyboard.Key.space:
            pause_flag = not pause_flag
            if pause_flag:
                print("Simulation PAUSED. Press SPACE to resume.")
            else:
                print("Simulation RESUMED.")
        elif hasattr(key, "char") and key.char == "5":
            v = 0.0
            h = 0.0
            print("Commands reset: v = 0.0, h = 0.0")
    except AttributeError:
        pass

def start_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def quat_apply_np(quat, vec):
    quat = np.asarray(quat)
    vec = np.asarray(vec)
    orig_shape = vec.shape

    q = quat.reshape(-1, 4)
    v = vec.reshape(-1, 3)

    w = q[:, 0]
    qvec = q[:, 1:4]

    t = 2 * np.cross(qvec, v)
    v_rot = v + (w[:, None] * t) + np.cross(qvec, t)
    v_rot = v_rot.reshape(orig_shape)
    return v_rot

reindex_list = [15, 16, 17, 18, 19, 20, 21, 22, 0, 2, 6, 8, 12, 1, 3, 7, 9, 13, 14, 4, 5, 10, 11]

class RealTimePolicyController:
    def __init__(self, 
                 xml_file, 
                 policy_path, 
                 device='auto',
                 policy_frequency=50,
                 activation='auto',
                 export_onnx=False,
                 ):

        self.device = resolve_device(device)
        self.policy = load_policy(
            policy_path,
            self.device,
            activation=activation,
            export_onnx=export_onnx,
        )

        # Create MuJoCo sim
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.005
        self.model.opt.iterations = 10
        self.model.opt.ls_iterations = 20
        self.model.opt.ccd_iterations = 50
        
        self.data = mujoco.MjData(self.model)
        
        self.viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        self.viewer.cam.distance = 4.0
        self.viewer.cam.azimuth = 210.0
        self.viewer.cam.elevation = -10.0
        self.num_actions = 23
        self.sim_duration = 30.0
        self.sim_dt = 0.005
        self.cycle_time = 6
        self.step_dt = 1 / policy_frequency
        self.sim_decimation = int(1 / (policy_frequency * self.sim_dt))
        
        print(f"sim_decimation: {self.sim_decimation}")

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        self.robot_default_dof_pos = np.array([
            0.0, 0.0, 0.0, 0.23, -0.20, 0.0,
            -0.7, 0.0, 0.0, 1.17, -0.45, 0.0,
            0.0, 0.0, 0.0,
            -0.03, 0.45, -0.21, 1.32,
            -0.7, -0.845, 0.83, 1.19
            ])

        self.mujoco_default_dof_pos = np.concatenate([
            np.array([-0.03, 0.1, 0.78]),
            np.array([1, 0, 0, 0]),
            self.robot_default_dof_pos,
            np.array([0, 0, 0.10]),
            np.array([1, 0, 0, 0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ])
        
        self.action_scale = np.array([
            0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386, 
            0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386, 
            0.5475, 0.4386, 0.4386, 
            0.4386, 0.4386, 0.4386, 0.4386,
            0.4386, 0.4386, 0.4386, 0.4386,
        ])

        self.n_obs_single = 3 + 3 + 3 + 3*23 + 1
        self.history_len = 5
        self.total_obs_size = self.n_obs_single * (self.history_len) 
        policy_input_dim = getattr(self.policy, "input_dim", self.total_obs_size)
        if policy_input_dim != self.total_obs_size:
            raise ValueError(
                f"Policy expects {policy_input_dim} observations, but sim.py "
                f"constructs {self.total_obs_size}."
            )

        self.obs_block_dims = [2, 1, 3, 3, 23, 23, 23, 1]
        self.obs_block_starts = np.cumsum([0] + self.obs_block_dims[:-1])

        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))

    def reset_sim(self):
        """Reset simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, init_pos):
        """Reset robot to initial position"""
        self.data.qpos[:] = init_pos
        self.data.qvel[:] = 0
        self.data.ctrl[:-7] = self.robot_default_dof_pos[reindex_list]
        mujoco.mj_forward(self.model, self.data)

    def extract_data(self):
        n_robot_dof = self.num_actions

        robot_quat = self.data.qpos[3:7]
        robot_dof_pos = self.data.qpos[7:7+n_robot_dof]
        robot_ang_vel = self.data.qvel[3:6]
        robot_dof_vel = self.data.qvel[6:6+n_robot_dof] 

        return robot_dof_pos, robot_dof_vel, robot_quat, robot_ang_vel

    def run(self):
        """Main simulation loop"""
        global reset_flag, pause_flag, v, h
        print("Starting Skater simulation...")

        self.reset_sim()
        self.reset(self.mujoco_default_dof_pos)

        steps = int(self.sim_duration / self.sim_dt)
        pbar = tqdm(range(steps), desc="Simulating Skater...")

        phase_counter = 0

        try:
            for i in pbar:
                if not self.viewer.is_running():
                    print("Viewer closed, stopping simulation.")
                    break
                if reset_flag:
                    self.reset_sim()
                    self.reset(self.mujoco_default_dof_pos)
                    reset_flag = False
                    phase_counter = 0
                    print("Simulation RESET!")
                if pause_flag:
                    time.sleep(0.01) 
                    continue
                t_start = time.time()

                # Match training: a standing command keeps the gait phase at zero.
                if v < 0.1:
                    phase_counter = 0
                else:
                    phase_counter += 1

                phase = ((phase_counter * self.step_dt / self.cycle_time)) % 1.0
                phase = torch.tensor(phase)
                phase = torch.clip(phase, 0.0, 1.0)

                robot_dof_pos, robot_dof_vel, robot_quat, robot_ang_vel = self.extract_data()

                gravity_orientation = get_gravity_orientation(robot_quat)

                sensor_id = self.model.sensor("robot/imu_ang_vel").id
                sensor_adr = self.model.sensor_adr[sensor_id]
                sensor_dim = self.model.sensor_dim[sensor_id]
                sensor_ang_vel = self.data.sensordata[sensor_adr : sensor_adr + sensor_dim]

                forward_w = quat_apply_np(robot_quat, np.array([1, 0, 0]))
                heading = np.array([np.arctan2(forward_w[1], forward_w[0])])

                obs_proprio = np.concatenate([
                    np.array([v, h], dtype=np.float32) * [2.0, 1.0],
                    heading * 1.0 / math.pi,
                    sensor_ang_vel * 0.25,
                    gravity_orientation,
                    (robot_dof_pos - self.robot_default_dof_pos),
                    robot_dof_vel * 0.05,
                    self.last_action,
                    np.array([phase], dtype=np.float32),
                ])

                self.proprio_history_buf.append(obs_proprio)
                history_array = np.array(self.proprio_history_buf)
                
                obs_buf_parts = []
                for i, (start, dim) in enumerate(zip(self.obs_block_starts, self.obs_block_dims)):
                    obs_block = history_array[:, start:start+dim]
                    obs_buf_parts.append(obs_block.flatten())
                    
                obs_buf = np.concatenate(obs_buf_parts)

                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                
                self.last_action = raw_action
                scaled_actions = raw_action * self.action_scale

                pd_target_robot = (scaled_actions + self.robot_default_dof_pos)

                viewer_closed = False
                for _ in range(self.sim_decimation):
                    if not self.viewer.is_running():
                        viewer_closed = True
                        break
                    self.data.ctrl[:-7] = pd_target_robot[reindex_list]
                    mujoco.mj_step(self.model, self.data)
                    pelvis_pos = self.data.xpos[self.model.body("robot/pelvis").id]
                    self.viewer.cam.lookat = pelvis_pos
                    self.viewer.sync()
                if viewer_closed:
                    break

                dt = self.model.opt.timestep * self.sim_decimation
                sleep = dt - (time.time() - t_start)
                if sleep > 0:
                    time.sleep(sleep)
                    
        except Exception as e:
            print(f"Error in run: {e}")
            import traceback
            traceback.print_exc()
        finally:
            
            if self.viewer:
                self.viewer.close()
            print("Simulation finished.")


def main():
    parser = argparse.ArgumentParser(description='Run skater policy in simulation')
    parser.add_argument('--xml', type=str, default='mjlab_scene.xml',
                        help='Path to MuJoCo XML file')
    parser.add_argument('--policy', type=str, required=True,
                        help='Path to a skater ONNX policy or RSL-RL .pt checkpoint')
    parser.add_argument('--device', type=str, 
                        default='auto',
                        help='Device to run policy on (auto/cuda/cpu)')
    parser.add_argument(
        "--activation",
        default="auto",
        help="Actor activation for .pt checkpoints (auto reads params/agent.yaml)",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export a loaded .pt checkpoint beside it as ONNX",
    )
    parser.add_argument("--policy_frequency", help="Policy frequency", default=50, type=int)
    args = parser.parse_args()
    
    if not os.path.exists(args.policy):
        print(f"Error: Policy file {args.policy} does not exist")
        return
    
    if not os.path.exists(args.xml):
        print(f"Error: XML file {args.xml} does not exist")
        return
    
    print(f"Starting skater simulation controller...")
    print(f"  XML file: {args.xml}")
    print(f"  Policy file: {args.policy}")
    print(f"  Device: {args.device}")

    listener_thread = threading.Thread(target=start_listener, daemon=True)
    listener_thread.start()

    controller = RealTimePolicyController(
        xml_file=args.xml,
        policy_path=args.policy,
        device=args.device,
        policy_frequency=args.policy_frequency,
        activation=args.activation,
        export_onnx=args.export_onnx,
    )
    controller.run()


if __name__ == "__main__":
    main()
