#!/bin/bash
parallel ./train_quad.py config/ppo__crazyflie_baseline.yml _results_temp/ppo_crazyflie_simplified_goal \
--seed {1} \
-p env_param.rew_coeff.orient \
-pv {2} \
::: {1..3} ::: 0.5 0.2 0.1
