#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-root@116.238.240.2}"
# REMOTE_HOST="${REMOTE_HOST:-root@10.127.48.252}"
REMOTE_PORT="${REMOTE_PORT:-30961}"
REMOTE_ROOT="${REMOTE_ROOT:-/data_sjy/kjp/Humanoid/humanoid_skateboarding/logs/rsl_rl/g1_skater}"
LOCAL_ROOT="${LOCAL_ROOT:-${PROJECT_ROOT}/logs/rsl_rs/g1_skater}"
WEIGHT_GLOB="model_*.pt"

# macOS limits Unix-domain socket paths to 104 bytes. Always use a short path
# under /tmp instead of the usually much longer per-user $TMPDIR.
control_dir="$(mktemp -d "/tmp/g1s.XXXXXX")"
ssh_common=(
  -o ControlMaster=auto
  -o ControlPersist=120
  -o "ControlPath=${control_dir}/ssh-%C"
)
ssh_opts=(-n -p "${REMOTE_PORT}" "${ssh_common[@]}")
scp_opts=(-P "${REMOTE_PORT}" -p "${ssh_common[@]}")

cleanup() {
  ssh "${ssh_opts[@]}" -O exit "${REMOTE_HOST}" >/dev/null 2>&1 || true
  rm -f -- "${control_dir}"/ssh-*
  rmdir "${control_dir}" 2>/dev/null || true
}
trap cleanup EXIT

remote_quote() {
  printf "%q" "$1"
}

copy_remote_file() {
  local remote_file="$1"
  local local_file="$2"

  mkdir -p "$(dirname "${local_file}")"
  echo "  <- ${remote_file}"
  scp "${scp_opts[@]}" "${REMOTE_HOST}:${remote_file}" "${local_file}" </dev/null
}

mkdir -p "${LOCAL_ROOT}"

remote_root_q="$(remote_quote "${REMOTE_ROOT}")"
if ! ssh "${ssh_opts[@]}" "${REMOTE_HOST}" \
  "test -d ${remote_root_q}"; then
  echo "Could not access remote directory: ${REMOTE_HOST}:${REMOTE_ROOT}" >&2
  echo "Check the SSH connection, credentials, and remote path." >&2
  exit 1
fi

experiment_count=0

while IFS= read -r -d '' remote_experiment; do
  experiment_count=$((experiment_count + 1))
  experiment_name="$(basename "${remote_experiment}")"
  local_experiment="${LOCAL_ROOT}/${experiment_name}"
  remote_experiment_q="$(remote_quote "${remote_experiment}")"

  echo
  echo "[${experiment_count}] Syncing experiment: ${experiment_name}"
  mkdir -p "${local_experiment}"

  # RSL-RL checkpoints use model_<iteration>.pt. Natural version sorting makes
  # model_10000.pt newer than model_900.pt without relying on modification time.
  latest_weight="$(
    ssh "${ssh_opts[@]}" "${REMOTE_HOST}" \
      "find ${remote_experiment_q} -maxdepth 1 -type f -name '${WEIGHT_GLOB}' -printf '%f\n' \
        | LC_ALL=C sort -V \
        | tail -n 1"
  )"

  if [[ -z "${latest_weight}" ]]; then
    echo "  No ${WEIGHT_GLOB} checkpoint found; skipping."
    continue
  fi

  remote_weight="${remote_experiment}/${latest_weight}"
  local_weight="${local_experiment}/${latest_weight}"
  temporary_weight="${local_weight}.part.$$"

  echo "  Latest checkpoint: ${latest_weight}"
  copy_remote_file "${remote_weight}" "${temporary_weight}"
  mv -f -- "${temporary_weight}" "${local_weight}"
done < <(
  ssh "${ssh_opts[@]}" "${REMOTE_HOST}" \
    "find ${remote_root_q} -mindepth 1 -maxdepth 1 -type d -print0"
)

echo
if [[ "${experiment_count}" -eq 0 ]]; then
  echo "No experiment directories found under ${REMOTE_HOST}:${REMOTE_ROOT}"
else
  echo "Done. Synced ${experiment_count} experiment(s) to:"
  echo "  ${LOCAL_ROOT}"
fi
