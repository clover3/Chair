#!/usr/bin/env bash

sbatch -p m40-short --gres=gpu:1 /home/youngwookim/code/Chair/src/adhoc_main.sh