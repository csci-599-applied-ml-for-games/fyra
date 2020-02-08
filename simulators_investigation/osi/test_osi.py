import sys
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import time

from gym_art.quadrotor.quad_utils import R2quat
from gym_art.quadrotor.quad_models import *
from gym_art.quadrotor.quadrotor_randomization import sample_random_dyn

from quad_dynalearn.osi.deterministic_mlp_osi import DeterministicMlpOSI
from quad_dynalearn.osi.categorical_mlp_osi import CategoricalMlpOSI
from quad_dynalearn.osi.deterministic_lstm_osi import DeterministicLSTMOSI

from simulators_investigation.networks.lstm import lstm_policy
from simulators_investigation.utils import *

def test_rollout(osi_param_file, policy_param_file=None, 
        traj_file=None,
        render=False, 
        rollouts_num=5,
        render_each = 2,
        ep_time=2, 
        use_noise=False, # if it's true, use what the env already has
        random_init=False,   # if it's true, use what the env already has
        random_quad=False, #False, # if it's true, use what the env already has
        use_osi_output=False,
        excite=False,
        save=False, 
        plot=False):
    import tqdm

    if traj_file != None:
        print("Reading trajectory...")
        traj = np.loadtxt(traj_file, delimiter=',')
        traj_freq = 1 ## every 1 time step(s), the goal is set to the next point in the traj
        # rollouts_num = 1

    import tensorflow as tf
    tf.reset_default_graph()
    with tf.Session() as sess:
        print("extrating osi parameters from file %s ..." % osi_param_file)
        osi_params = joblib.load(osi_param_file)
        env = osi_params['env'].env
        osi = osi_params['osi']

        # Initialize some missing variables
        uninitialized_vars = []
        for var in tf.all_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                # print("Uninitialized var: ", var)
                uninitialized_vars.append(var)
        init_new_vars_op = tf.initialize_variables(uninitialized_vars)
        sess.run(init_new_vars_op)
        # sess.run(tf.global_variables_initializer())

        if policy_param_file == None:
            policy = osi_params['policy']
        else:
            print("extracting policy parameters from file %s ..." % policy_param_file)
            policy_params = joblib.load(policy_param_file)
            policy = policy_params['policy']

        ## input to the osi
        obs_tsteps = osi.obs_tsteps
        obs_comp = osi.obs_comp
        obs_size = osi.obs_size
        out_comp = osi.out_comp
        print(obs_comp)
        print(out_comp)

        ## modify the environment
        if not use_noise:
            ## do not use noise
            env.update_sense_noise(sense_noise=None)
        if not random_init:
            ## set init random state to False
            env.init_random_state = False
            init_pos = np.array([0, 0, 0.05])
            init_vel = np.array([0, 0, 0])
            init_rot = rpy2R(0, 0, 0) # np.eye(3)
            init_omega = np.array([0, 0, 0])
        if not random_quad:
            ## no random quad
            env.dynamics_randomize_every = None
            ## set up a quad
            env.update_dynamics(dynamics_params=crazyflie_params())
        
        print(env.dynamics_params)

        dt = env.dt
        env.excite = False
        sim_steps = env.sim_steps
        env.ep_len = int(ep_time / (dt * sim_steps))
        ## output import info about the environment
        print('#############################')
        print('Episode time: {}'.format(ep_time))
        print('Integration step: {}'.format(dt))
        print('Simulation step: {}'.format(sim_steps))
        print('#############################')
        ## ========================================

        if isinstance(osi, DeterministicLSTMOSI):
            lstm_osi = lstm_policy(osi._regressor)

        ## Diagnostics
        observations = []
        pred_error = []
        dyna_params_stats = []

        for rollouts_id in tqdm.tqdm(range(rollouts_num)):
            s = env.reset()
            policy.reset()

            ## reset the goal to x:0, y:0 z:0
            env.goal = np.array([0., 0., 1])

            if isinstance(osi, DeterministicLSTMOSI):
                lstm_osi.reset_internal_states()

            dynamics = env.dynamics

            ## set the initial state
            if not random_init:
                env.dynamics.set_state(init_pos, init_vel, init_rot, init_omega)
                env.dynamics.reset()
                env.scene.reset(env.goal, env.dynamics)
                s = env.state_vector(env)
     
            ## initial inputs for the osi
            osi_obs = [np.zeros(obs_size) for _ in range(obs_tsteps)]

            t = 0
            traj_ptr = 0
            done = False
            osi_outputs = []
            pred = {var: 0 for var in out_comp}
            scaling = {'t2w': [1.5, 3.5], 't2t': [0.005, 0.05]}
            while True:
                # =================================
                if render and (t % render_each == 0): env.render()

                ## use the osi outputs in the state
                if t > 0 and use_osi_output:
                    for idx, osi_output in enumerate(osi_outputs[::-1]):
                        s[-1-idx] = osi_output

                if traj_file != None:
                    if traj_ptr < traj.shape[0]:
                        if t % traj_freq == 0:
                            env.goal = traj[traj_ptr][:3]
                            traj_ptr += 5   ## need to adjust this parameter according to the trajectory file frequency
                        action = policy.get_action(s)[1]['mean']
                        s, r, _, info = env.step(action)
                    else:
                        done = True
                elif excite and t % 250 == 0:
                    ## change the goal every 5 time step
                    env.goal = np.concatenate([
                        np.random.uniform(low=-0.5, high=0.5, size=(2,)),
                        np.random.uniform(low=0.5, high=1.5, size=(1,))
                    ])    
                    action = policy.get_action(s)[1]['mean'] 
                    s, r, done, info = env.step(action)
                else:
                    action = policy.get_action(s)[1]['mean']
                    s, r, done, info = env.step(action)

                if done: break
                t += 1
                # ========== Diagnostics ==========
                ## osi prediction
                obs_comp_values = []
                for key in obs_comp:
                    obs_comp_values.append(np.array(info['obs_comp'][key][0]).reshape([1, -1]))
                    
                osi_obs = np.roll(osi_obs, shift=1, axis=0)
                osi_obs[0] = np.concatenate(obs_comp_values, axis=1)

                if isinstance(osi, DeterministicMlpOSI):
                    osi_outputs = osi.predict(np.concatenate(osi_obs, axis=0).reshape([1, -1]))
                elif isinstance(osi, DeterministicLSTMOSI):
                    osi_outputs = lstm_osi.get_action(osi_obs[0].reshape([1, 1, -1])).flatten()
                elif isinstance(osi, CategoricalMlpOSI):
                    # output = osi.predict(np.concatenate(osi_obs, axis=0).reshape([1, -1]))
                    ## not supported now
                    return

                pred_gt = info['dyn_params']

                ## rescaling the outputs
                for idx, output_variable in enumerate(out_comp):
                    # pred_gt[output_variable] = pred_gt[output_variable] * (scaling[output_variable][1]-scaling[output_variable][0])+scaling[output_variable][0]
                    # osi_outputs[idx] = osi_outputs[idx] * (scaling[output_variable][1]-scaling[output_variable][0])+scaling[output_variable][0]
                    pred[output_variable] = osi_outputs[idx]

                pred_error.append([np.abs(pred_gt[var]-pred[var]) for var in out_comp])
                if t == 1:
                    dyna_params_stats.append([pred_gt[var] for var in out_comp])

                real_pos = env.state_vector(env)
                pos = real_pos[0:3] + env.goal
                vel = real_pos[3:6]
                quat = R2quat(real_pos[6:15])
                # reformat to [x, y, z, w]
                quat[0], quat[1], quat[2], quat[3] = quat[1], quat[2], quat[3], quat[0]
                rpy = R2rpy(real_pos[6:15])
                omega = real_pos[15:18]

                # obs = np.concatenate([[t * dt * sim_steps], 
                #     pos, rpy, vel, omega, env.goal, action, \
                #     np.ravel([[pred[var], pred_gt[var]] for var in out_comp])])
                obs = np.concatenate([[t * dt * sim_steps], 
                    pos, rpy, vel, omega, env.goal, info['obs_comp']['ClippedAct'][0], \
                    np.ravel([[pred[var], pred_gt[var]] for var in out_comp])])
                observations.append(obs)
                
        ## print avg prediction error
        avg_error = np.mean(pred_error, axis=0).flatten()
        std_error = np.std(pred_error, axis=0).flatten()
        dyna_params_mean = np.mean(dyna_params_stats, axis=0).flatten()
        dyna_params_std = np.std(dyna_params_stats, axis=0).flatten()
        dyna_params_max = np.max(dyna_params_stats, axis=0).flatten()
        dyna_params_min = np.min(dyna_params_stats, axis=0).flatten()
        for idx, var in enumerate(out_comp):
            print('Mean of {} is {}, std is {}, max is {}, min is {}'.format(
                var, dyna_params_mean[idx], dyna_params_std[idx], dyna_params_max[idx], dyna_params_min[idx]))
            print('Avg error on {} prediction is {}, std is {}.'.format(var, avg_error[idx], std_error[idx]))
        
        if save == True:
            save_path = './test_tmp/'
            try:
                os.makedirs(save_path, exist_ok=True)
            except FileExistsError:
                # directory already exists
                pass
            np.savetxt(save_path + 'observations.csv', observations, delimiter=',')

        if plot == True:
            TIME = 0 
            X, Y, Z = 1, 2, 3
            Roll, Pitch, Yaw = 4, 5, 6
            VX, VY, VZ = 7, 8, 9 
            Roll_rate, Pitch_rate, Yaw_rate = 10, 11, 12
            Xt, Yt, Zt = 13, 14, 15
            t0, t1, t2, t3 = 16, 17, 18, 19
            pred_obs_idx = [19+i for i in range(1, 2*len(out_comp), 2)]

            plot_comp = {
                'Omega': {'roll rate': Roll_rate, 'pitch rate': Pitch_rate, 'yaw rate': Yaw_rate},
                'Position': {'X': X, 'Target X': Xt, 'Y': Y, 'Target Y': Yt, 'Z': Z, 'Target Z': Zt},
                'Velocity': {'VX': VX, 'VY': VY, 'VZ': VZ}, 
                # 'Orientation': {'qx': QX, 'qy': QY, 'qz': QZ, 'qw': QW}, 
                'Orientation': {'R': Roll, 'P': Pitch, 'Y': Yaw}, 
                'Actions': {'t0': t0, 't1': t1, 't2': t2, 't3': t3},
            }
            ## add osi outputs for plotting
            for idx, var in enumerate(out_comp):
                plot_comp[var] = {'predicted': pred_obs_idx[idx], 'ground_truth': pred_obs_idx[idx]+1}
            
            observations = np.array(observations)

            total_subplots = len(plot_comp)
            current_plot = 1
            for obs_comp in plot_comp:
                plt.subplot(total_subplots, 1, current_plot)
                # if current_plot == 1:
                #     plt.title('Avg t2w prediction error: {}, Avg t2t prediction error {}'.format(avg_error[0], avg_error[1]))
                for comp in plot_comp[obs_comp]:
                    plt.plot(observations[:,plot_comp[obs_comp][comp]], '-', label=comp)
                plt.xlabel('Time [s]')
                plt.ylabel(obs_comp)    
                plt.legend(loc=9, ncol=3, borderaxespad=0.)
                
                current_plot += 1

            plt.show()
        

    print("##############################################################")


def main(argv):
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'osi_param_file',
        type=str,
        help="provide a osi params.pkl file"
    )

    parser.add_argument(
        '-policy_param_file', 
        type=str,
        default=None,
        help="provide a policy params.pkl file"
    )

    parser.add_argument(
        '-traj',
        type=str,
        default=None,
        help='a trajectory file'    
    )

    parser.add_argument(
        '-rollouts_num', 
        type=int,
        default=5,
        help="number of trajectories to simulate"
    )    

    parser.add_argument(
        '-ep_time', 
        type=int,
        default=2,
        help="length of a trajectory"
    )    

    parser.add_argument(
        '--random_quad', 
        action='store_true',
        help='whether to use the env quad parameter'
    )

    parser.add_argument(
        '--use_osi_output', 
        action='store_true',
        help='whether to feed the osi outputs to the controller'
    )

    parser.add_argument(
        '--use_noise', 
        action='store_true',
        help='whether to use noise'
    )

    parser.add_argument(
        '--render', 
        action='store_true',
        help='whether to render'
    )

    parser.add_argument(
        '--random_init', 
        action='store_true',
        help='whether to randomly initialize the quad'
    )   

    parser.add_argument(
        '--excite', 
        action='store_true',
        help='whether to perturb the quad'
    )

    parser.add_argument(
        '--plot', 
        action='store_true',
        help='whether to plot'
    )    

    parser.add_argument(
        '--save', 
        action='store_true',
        help='whether to record flight'
    )    

    args = parser.parse_args()

    print('Running test rollout...')
    test_rollout(
        args.osi_param_file, 
        args.policy_param_file, 
        traj_file=args.traj,
        ep_time=args.ep_time,
        use_noise=args.use_noise,
        render=args.render, 
        rollouts_num=args.rollouts_num,
        random_init=args.random_init,
        random_quad=args.random_quad,
        use_osi_output=args.use_osi_output,
        excite=args.excite,
        save=args.save, 
        plot=args.plot
    )


if __name__ == '__main__':
	main(sys.argv)
