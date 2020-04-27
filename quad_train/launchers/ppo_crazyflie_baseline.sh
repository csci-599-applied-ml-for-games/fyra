#!/bin/bash
parallel -k --lb ./train_quad.py config/ppo_crazyflie_baseline.yml _results_temp/ppo_crazyflie_velocity_0.5_1.0 \
--seed {1} \
::: {1..3}
