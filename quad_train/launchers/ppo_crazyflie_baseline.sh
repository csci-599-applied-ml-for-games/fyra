#!/bin/bash
parallel --files ./train_quad.py config/ppo__crazyflie_baseline.yml _results_temp/ppo_crazyflie_baseline \
--seed {1} \
::: {1..2}
