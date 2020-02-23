#!/usr/bin/env python
"""
This is a parametrized script to run TRPO/PPO 
with a custom env
"""
import argparse
import sys
import os
import datetime, time
import itertools
import os.path as osp
import uuid
import copy

import numpy as np

import dateutil.tz
import yaml

import gym

from garage.envs import normalize
from garage.experiment import run_experiment

# Custom stuff
import quad_train.config.config_loader as conf
import quad_train.misc.variants_utils as vu


########################################################################
## ARGUMENTS
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("config_file", help='yaml file with default settings of parameters')
parser.add_argument("log_dir", default='_results_temp/trpo_ppo_last', help='Directory to log into')
parser.add_argument("--cont", "-c", action="store_true", help='Continue from params.pkl. The params.pkl must exist in the log folders')
parser.add_argument("--seed", '-s', default=None, help='list of seeds to use separated by comma (or a single seed w/o comma). If None seeds from config_file will be used')
parser.add_argument("--n_parallel", '-n', type=int, default=1, help='Number of parallel workers to run a single task')
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

########################################################################
## PARAMETERS (non grid)
# Loading parameters not specified in the arguments
print('Reading parameter file %s ...' % args.config_file)
params = conf.trpo_ppo_default_params()
yaml_stream = open(args.config_file, 'r')
params_new = yaml.load(yaml_stream)
params.update(params_new)
print('###############################################################')
print('### PARAMETERS LOADED FROM CONFIG FILES (Later updated by arguments provided)')
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

def run_task_continue(task_param):
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

    import sys
    import os

    import garage.misc.logger as logger
    import joblib

    pkl_file = logger.get_snapshot_dir().rstrip(os.sep) + os.sep + "params.pkl"
    if os.path.isfile(pkl_file):
        print("WARNING: Loading and continuing from %s snapshot ..." 
            % logger.get_snapshot_dir().rstrip(os.sep))
    else:
        raise ValueError("ERROR: params.pkl not found in %s" % pkl_file)

    import tensorflow as tf

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Unpack the snapshot
        snapshot = joblib.load(pkl_file)
        env = snapshot["env"]
        policy = snapshot["policy"]
        itr = snapshot["itr"]
        task_param["alg_param"]["start_itr"] = itr+1

        del task_param["env"]
        del task_param["env_param"]
        del task_param["policy_class"]
        del task_param["policy_param"]

        if task_param["alg_class"] != "CEM" and task_param["alg_class"] != "CMAES":
            baseline = snapshot["baseline"]
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

        algo.train(sess=sess)

def run_task_new(task_param):
    """
    Wrap PPO training task in the run_task function.

    :param _:
    :return:
    """
    from garage.tf.baselines import GaussianMLPBaseline
    from garage.tf.envs import TfEnv
    from garage.tf.policies import GaussianMLPPolicy, DeterministicMLPPolicy, GaussianGRUPolicy, GaussianLSTMPolicy
    
    from quad_dynalearn.algos.cem import CEM
    from quad_dynalearn.algos.cma_es import CMAES
    from quad_dynalearn.algos.ppo import PPO
    from quad_dynalearn.algos.trpo import TRPO

    import sys
    import os

    import garage.misc.logger as logger
    import joblib

    if os.path.isfile(logger.get_snapshot_dir().rstrip(os.sep) + os.sep + "params.pkl"):
        print("WARNING: found params file from the previous iteration. They will be erased !!!!")

    if task_param["env"] == "QuadrotorEnv":
        # from gym_art.quadrotor.quadrotor_control import *
        # from gym_art.quadrotor.quadrotor_modular import QuadrotorEnv
        try:
            from gym_art.quadrotor.quadrotor import QuadrotorEnv
            env = TfEnv(QuadrotorEnv(**task_param["env_param"]))
        except:
            print("WARNING: Couldn't load quadrotor.py, using quadrotor_modular.py")
            from gym_art.quadrotor.quadrotor_modular import QuadrotorEnv
            env = TfEnv(QuadrotorEnv(**task_param["env_param"]))
        del task_param["env_param"]
    else:
        env = TfEnv(normalize(gym.make(task_param["env"])))
    del task_param["env"]
    
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

if args.cont:
    run_task = run_task_continue
else:
    run_task = run_task_new

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
