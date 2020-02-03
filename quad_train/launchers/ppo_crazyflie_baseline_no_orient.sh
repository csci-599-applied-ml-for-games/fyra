#!/bin/bash
parallel ./train_quad.py config/ppo__crazyflie_baseline_no_orient.yml _results_temp/ppo_crazyflie_baseline_no_orient_fixed \
--seed {1} \
::: {1..5}
