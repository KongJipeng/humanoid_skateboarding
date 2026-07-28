SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)

if [ -z "$1" ]; then
  echo "Usage: $0 path_to_policy.onnx"
  exit 1
fi

ckpt_path=$1

if [ "$(uname -s)" = "Darwin" ]; then
  python_runner="mjpython"
else
  python_runner="python"
fi

uv run "${python_runner}" "${SCRIPT_DIR}/sim.py" \
    --xml "${SCRIPT_DIR}/mjlab_scene.xml" \
    --policy "${ckpt_path}" \
    --device auto \
    --policy_frequency 50 
