#!/usr/bin/env bash
# Launch bimanual Push-T teleoperation (requires X11 display via MobaXterm).
#
# Usage:
#   ./launch_bimanual_teleop.sh                               # plain teleop
#   ./launch_bimanual_teleop.sh --dataset_file datasets/pusht_bimanual.hdf5
#   ./launch_bimanual_teleop.sh --num_envs 1 --dataset_file datasets/pusht_bimanual.hdf5
#
# All extra args are forwarded to the teleop script.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Launching bimanual Push-T teleop…"
echo "  Keys (always): WASD=left arm, IJKL=right arm, Q/E=Z-height, [/]=speed"
echo "  Keys (recording, if --dataset_file given): P=start, O=save, R=discard"
echo ""

cd "$SCRIPT_DIR"
python scripts/environments/teleoperation/teleop_pusht_bimanual.py "$@"
