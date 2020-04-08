#!/bin/bash
parallel ./train_quad_ddpg.py config/ddpg_crazyflie_baseline.yml _results_temp/ddpg_crazyflie_baseline \
--seed {1} \
::: {1..5}
