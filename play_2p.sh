#!/usr/bin/env bash

# Usage: play_2p.sh (main_arg_1 main_arg_2 ... main_arg_n)

# Get parameters
game=pong
rom="roms/$game.bin"

# Execute main script
python3 src/main.py $rom --environment ale --frame_skip 1 --game_mode 2 --two_player true "$@"
