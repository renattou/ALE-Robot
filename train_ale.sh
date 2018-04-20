#!/usr/bin/env bash

# Check parameters
if [ $# -lt 1 ]; then
  echo "Usage: train_ale.sh game (main_arg_1 main_arg_2 ... main_arg_n)"
  exit 1
fi

# Get parameters
game=$1
rom="roms/$game.bin"
shift 1

# Execute main script
python3 src/main.py $rom --environment ale --target_fps -1 --exploration_decay_steps 1000000 "$@"
