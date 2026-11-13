#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo 'Usage: bash simulation/ns3/run_ns3.sh /path/to/ns-3 /absolute/allocations.csv /absolute/ns3_flows.csv [slotSeconds]'
  exit 2
fi
NS3_DIR="$(realpath "$1")"
INPUT="$(realpath "$2")"
OUTPUT="$(realpath -m "$3")"
SLOT="${4:-1.0}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gameedgeallocai-ns3.cc"
cp "$SRC" "$NS3_DIR/scratch/gameedgeallocai-ns3.cc"
(
 cd "$NS3_DIR"
 ./ns3 run "gameedgeallocai-ns3 --input=$INPUT --output=$OUTPUT --slotSeconds=$SLOT"
)
