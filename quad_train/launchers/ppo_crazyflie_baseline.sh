#!/bin/bash
parallel -k --lb ./train_quad.py config/ppo_crazyflie_baseline.yml _results_temp/ppo_crazyflie_simplified_epsilon_2 \
--seed {1} \
::: {1..3}
