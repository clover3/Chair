#!/usr/bin/env bash

sbatch -p titanx-short --gres=gpu:1 adhoc_main.sh