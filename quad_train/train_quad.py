#!/usr/bin/env python
"""
This is a parametrized script to run TRPO/PPO 
with a custom env
"""
import os

os.environ["OMP_NUM_THREADS"]="6"
os.environ["KMP_AFFINITY"]="none"

import argparse
import sys

import datetime, time
import itertools
import os.path as osp
import uuid
import copy

import numpy as np

import dateutil.tz
import yaml




########################################################################
## ARGUMENTS
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("config_file", help='yaml file with default settings of parameters')
parser.add_argument("log_dir", default='_results_temp/trpo_ppo_last', help='Directory to log into')
parser.add_argument("--seed", '-s', default="1", help='list of seeds to use separated by comma (or a single seed w/o comma). If None seeds from config_file will be used')
parser.add_argument("--n_parallel", '-n', type=int, default=4, help='Number of parallel workers to run a single task')
parser.add_argument("--snapshot_mode", '-snm', default='last', help='Snapshot mode. Opt: last')
parser.add_argument("--plot", '-plt', action="store_true", help='Plotting')
parser.add_argument("--param_name", '-p', help='task hyperparameter names separated by comma')
parser.add_argument("--param_val", '-pv', help='task hyperparam values.'+ 
                    ' For a single par separated by comma.' +
                    ' For adjacent params separated by double comma.' +
                    '   Ex: \"-p par1,par2 -pv pv11,pv12,,pv21,pv22\"' + 
                    '   where pv11,pv12 - par values for par1 , pv21,pv22 - par values for par2')
parser.add_argument("--nodisp", action="store_true", help="use virtual display")
args = parser.parse_args()

if args.nodisp:
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(1400,900))
    display.start()

import gym

from garage.envs import normalize
from garage.experiment import run_experiment

# Custom stuff
import quad_train.config.config_loader as conf
import quad_train.misc.variants_utils as vu

########################################################################
## PARAMETERS (non grid)
# Loading parameters not specified in the arguments
print('Reading parameter file %s ...' % args.config_file)
params = conf.trpo_ppo_default_params()
yaml_stream = open(args.config_file, 'r')
params_new = yaml.load(yaml_stream)
params.update(params_new)
print('###############################################################')
print('### PARAMETERS LOADED FROM CONFIG FILES (Later will be updated by arguments provided)')
print(params)

## Get a grid of task variations and put it into list as parameter dictionaries
## WARN: when you add more parameters to add_arguments you will have to modify grid_of_variants()
variants_list = vu.grid_of_variants(args, params)

## Saving command line executing the script
cmd = " ".join(sys.argv)
if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)
with open(args.log_dir + os.sep + "cmd.sh", "w") as cmdfile:
    cmdfile.write("#!/usr/bin/bash\n")
    cmdfile.write(cmd)


def run_task(task_param):
    """
    Wrap PPO training task in the run_task function.

    :param _:
    :return:
    """
    from garage.tf.baselines import GaussianMLPBaseline
    from garage.tf.envs import TfEnv
    from garage.tf.policies import GaussianMLPPolicy, DeterministicMLPPolicy, GaussianGRUPolicy, GaussianLSTMPolicy
    
    from quad_train.algos.cem import CEM
    from quad_train.algos.cma_es import CMAES
    from quad_train.algos.ppo import PPO
    from quad_train.algos.trpo import TRPO

    if task_param["env"] == "QuadrotorEnv":
        from quad_sim.quadrotor import QuadrotorEnv
        print(task_param)
        env = TfEnv(QuadrotorEnv(**task_param["env_param"]))
        del task_param["env_param"]
    else:
        env = TfEnv(normalize(gym.make(task_param["env"])))
    del task_param["env"]
    
    print(task_param["policy_param"])
    policy = locals()[task_param["policy_class"]](env_spec=env.spec, **task_param["policy_param"])
    del task_param["policy_class"]
    del task_param["policy_param"]

    if task_param["alg_class"] != "CEM" and task_param["alg_class"] != "CMAES":
        baseline = locals()[task_param["baseline_class"]](env_spec=env.spec, **task_param["baseline_param"])
        del task_param["baseline_class"]
        del task_param["baseline_param"]

        algo = locals()[task_param["alg_class"]](
            env=env,
            policy=policy,
            baseline=baseline,
            **task_param["alg_param"])
    else:
        algo = locals()[task_param["alg_class"]](
            env=env,
            policy=policy,
            **task_param["alg_param"])

    del task_param["alg_class"]
    del task_param["alg_param"]

    # Check that we used all parameters:
    # It helps revealing situations where you thought you set certain parameter
    # But in fact made spelling mistake and it failed
    del task_param["exp_name"] #This is probably generated by garage
    assert task_param == {}, "ERROR: Some of parameter values were not used: %s" % str(task_param)

    algo.train()

start_time = time.time()
for var in variants_list:
    ## Dumping config
    with open(var["log_dir"] + os.sep + "config.yml", 'w') as yaml_file:
        yaml_file.write(yaml.dump(var, default_flow_style=False))

    ## Running
    run_experiment(
        run_task,
        **var
    )

end_time = time.time()
print("##################################################")
print("Total Runtime: ", end_time - start_time)
