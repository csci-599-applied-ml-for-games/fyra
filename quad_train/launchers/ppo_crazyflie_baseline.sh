#!/bin/bash
parallel ./train_quad.py config/ppo_crazyflie_baseline.yml _results_temp/ppo_crazyflie_baseline \
--seed {1} \
::: {1..3}
