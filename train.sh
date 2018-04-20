#!/usr/bin/env bash

# Kill children when script finishes or exits
trap 'trap - SIGTERM && kill 0' SIGINT SIGTERM EXIT

# Check parameters
if [ $# -lt 2 ]; then
  echo "Usage: train.sh game vrep_path (main_arg_1 main_arg_2 ... main_arg_n)"
  exit 1
fi

# Get parameters
game=$1
rom="roms/$game.bin"
vrep=$2
current="$(cd "$(dirname "$0")"; pwd -P)"
shift 2

# Constants
port=-25000 # Use negative port numbers for shared memory.
enable_sync=TRUE # Use TRUE or FALSE to enable synchronous communication

# Execute V-REP
cd $vrep
cmd="./vrep.sh ${current}/scenes/main.ttt -h -gREMOTEAPISERVERSERVICE_${port}_TRUE_${enable_sync}"
$cmd &

# Execute main script
cd $current
python3 src/main.py $rom --environment robot "$@" --port $port --enable_sync $enable_sync
