"""HUSKY extensions for MjLab."""

import sys


if sys.platform == "darwin":
  # The pinned MjLab/MuJoCo-Warp commits reference a legacy MuJoCo flag that
  # was removed from newer macOS nightly wheels. HUSKY leaves multiccd disabled,
  # so a zero-valued compatibility member is sufficient for module import.
  import mujoco

  if not hasattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD"):
    setattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD", 0)
