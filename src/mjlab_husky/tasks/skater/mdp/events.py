"""Skateboard-specific domain-randomization events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _resolve_env_ids(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
) -> torch.Tensor:
  all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if env_ids is None:
    return all_env_ids
  if isinstance(env_ids, slice):
    return all_env_ids[env_ids]
  return env_ids.to(device=env.device, dtype=torch.long)


def _validate_range(name: str, value_range: tuple[float, float]) -> None:
  lower, upper = value_range
  if lower > upper:
    raise ValueError(f"{name} lower bound {lower} exceeds upper bound {upper}.")
  if lower <= 0.0:
    raise ValueError(f"{name} values must be positive, got {value_range}.")


@requires_model_fields("body_pos")
def randomize_skateboard_deck_height(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  asset_cfg: SceneEntityCfg,
  deck_half_thickness: float,
) -> None:
  """Randomize the physical deck-surface height and reset the board on the ground.

  Moving the front and rear truck bodies changes the wheel-to-deck geometry, so the
  randomized height remains after the wheels settle. This is different from only
  perturbing the skateboard freejoint height, which would fall back to one fixed
  equilibrium height.
  """
  env_ids = _resolve_env_ids(env, env_ids)
  height_range = env.cfg.skateboard_deck_height_range
  _validate_range("skateboard_deck_height_range", height_range)

  min_height, max_height = height_range
  deck_height = min_height + torch.rand(
    len(env_ids), device=env.device
  ) * (max_height - min_height)

  skateboard: Entity = env.scene[asset_cfg.name]
  body_ids = skateboard.indexing.body_ids[asset_cfg.body_ids]
  body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
  env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")

  default_body_pos = env.sim.get_default_field("body_pos")
  default_truck_z = default_body_pos[body_ids, 2]
  # Infer the nominal deck-surface height from the asset instead of coupling it
  # to the randomization range through a manually maintained reference value.
  nominal_deck_height = (
    skateboard.data.default_root_state[env_ids, 2] + deck_half_thickness
  )
  truck_z_offset = nominal_deck_height - deck_height
  env.sim.model.body_pos[env_grid, body_grid, 2] = (
    default_truck_z.unsqueeze(0) + truck_z_offset.unsqueeze(1)
  )

  # Place the deck center at surface height minus the collision box half-thickness.
  root_state = skateboard.data.default_root_state[env_ids].clone()
  root_state[:, :3] += env.scene.env_origins[env_ids]
  root_state[:, 2] = (
    deck_height - deck_half_thickness + env.scene.env_origins[env_ids, 2]
  )
  skateboard.write_root_state_to_sim(root_state, env_ids=env_ids)

  if not hasattr(env, "skateboard_deck_height"):
    env.skateboard_deck_height = (
      skateboard.data.default_root_state[:, 2] + deck_half_thickness
    )
  env.skateboard_deck_height[env_ids] = deck_height


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def randomize_skateboard_roll_pd(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Randomize board roll stiffness and damping with one coherent hardness sample.

  MuJoCo position actuators encode Kp in gainprm[..., 0] and biasprm[..., 1],
  while Kd is encoded in biasprm[..., 2]. All selected board/truck actuators use
  the same per-environment hardness sample so the front and rear trucks remain
  mechanically symmetric.
  """
  env_ids = _resolve_env_ids(env, env_ids)
  stiffness_range = env.cfg.skateboard_roll_stiffness_scale_range
  damping_range = env.cfg.skateboard_roll_damping_scale_range
  _validate_range("skateboard_roll_stiffness_scale_range", stiffness_range)
  _validate_range("skateboard_roll_damping_scale_range", damping_range)

  hardness = torch.rand(len(env_ids), device=env.device)
  stiffness_scale = stiffness_range[0] + hardness * (
    stiffness_range[1] - stiffness_range[0]
  )
  damping_scale = damping_range[0] + hardness * (
    damping_range[1] - damping_range[0]
  )

  skateboard: Entity = env.scene[asset_cfg.name]
  assert skateboard.indexing.ctrl_ids is not None
  actuator_ids = skateboard.indexing.ctrl_ids[asset_cfg.actuator_ids]
  actuator_ids = torch.as_tensor(
    actuator_ids, device=env.device, dtype=torch.long
  )
  env_grid, actuator_grid = torch.meshgrid(
    env_ids, actuator_ids, indexing="ij"
  )

  default_gain = env.sim.get_default_field("actuator_gainprm")
  default_bias = env.sim.get_default_field("actuator_biasprm")
  kp_scale = stiffness_scale.unsqueeze(1)
  kd_scale = damping_scale.unsqueeze(1)

  env.sim.model.actuator_gainprm[env_grid, actuator_grid, 0] = (
    default_gain[actuator_ids, 0].unsqueeze(0) * kp_scale
  )
  env.sim.model.actuator_biasprm[env_grid, actuator_grid, 1] = (
    default_bias[actuator_ids, 1].unsqueeze(0) * kp_scale
  )
  env.sim.model.actuator_biasprm[env_grid, actuator_grid, 2] = (
    default_bias[actuator_ids, 2].unsqueeze(0) * kd_scale
  )

  if not hasattr(env, "skateboard_roll_stiffness_scale"):
    env.skateboard_roll_stiffness_scale = torch.ones(
      env.num_envs, device=env.device
    )
    env.skateboard_roll_damping_scale = torch.ones(
      env.num_envs, device=env.device
    )
  env.skateboard_roll_stiffness_scale[env_ids] = stiffness_scale
  env.skateboard_roll_damping_scale[env_ids] = damping_scale
