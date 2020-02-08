from gym_art.quadrotor.quad_utils import R2quat
from gym_art.quadrotor.quad_models import *
from gym_art.quadrotor.quadrotor_full_goal_state import *
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import time

def test_rollout(
        param_file, 
        traj_file=None, 
        render=True, 
        rollouts_num=1,
        ep_time = 7.0,
        render_each=2,
        use_noise=True, # if it's true, use what the env already has
        random_init=False,  # if it's true, use what the env already has
        random_quad=False,  # if it's true, use what the env already has
        plot_step=None,  
        save=False, 
        plot=False
    ):
    import tqdm

    if traj_file != None:
        print("Reading trajectory...")
        traj = np.loadtxt(traj_file, delimiter=',')
        traj_freq = 1 ## every n time step(s), the goal is set to the next point in the traj

    import tensorflow as tf
    with tf.Session() as sess:
        print("extrating parameters from file %s ..." % param_file)
        params = joblib.load(param_file)

        policy = params['policy']
        env = params['env'].env

        ## create a new environment with the full goal state
        obs_repr = env.obs_repr
        env = QuadrotorEnv(
            dynamics_params=crazyflie_params(), 
            raw_control=True, 
            raw_control_zero_middle=True, 
            dynamics_randomize_every=None, 
            dynamics_randomization_ratio=None, 
            sim_freq=100,
            sim_steps=1,
            sense_noise="default", 
            init_random_state=False, 
            obs_repr=obs_repr)

        dt = env.dt
        sim_steps = env.sim_steps
        env.ep_len = int(ep_time / (dt * sim_steps))
        ## output import info about the environment
        print('#############################')
        print('Episode time: {}'.format(ep_time))
        print('Integration step: {}'.format(dt))
        print('Simulation step: {}'.format(sim_steps))
        print('#############################')
        ## ========================================

        dynamics = env.dynamics
        ## disable thrust noise
        dynamics.thrust_noise.sigma = 0.0
        ## modify motor delays
        dynamics.motor_damp_time_up = 0.0
        dynamics.motor_damp_time_down = 0.0
        print("thrust to weight ratio set to: {}, and max thrust is {}".format(dynamics.thrust_to_weight, dynamics.thrust_max))
        
        rollouts_id = 0
        for rollouts_id in tqdm.tqdm(range(rollouts_num)):
            s = env.reset()
            policy.reset()

            ## set the initial state
            if not random_init:
                dynamics.set_state(init_pos, init_vel, init_rot, init_omega)
                dynamics.reset()
                env.scene.reset(env.goal, dynamics)
                s = env.state_vector(env)

            rollouts_id += 1

            ## Diagnostics
            observations = []

            t = 0
            done = False
            traj_ptr = 0
            while True:
                # =================================
                if render and (t % render_each == 0): env.render()

                if traj_file != None:
                    if traj_ptr < traj.shape[0]:
                        if t % traj_freq == 0:
                            env.goal = Goal(traj[traj_ptr][0:3], vel=traj[traj_ptr][3:6], omega=np.radians(traj[traj_ptr][6:9]))
                            traj_ptr += 1
                        action = policy.get_action(s)[1]['mean']
                        s, r, _, info = env.step(action)
                    else:
                        done = True
                else:
                    action = policy.get_action(s)[1]['mean']
                    s, r, done, info = env.step(action)

                if done: break
                t += 1

                # ========== Diagnostics ==========
                real_pos = dynamics.state_vector()
                pos = real_pos[0:3]
                vel = real_pos[3:6]
                quat = R2quat(real_pos[6:15])
                # reformat to [x, y, z, w]
                quat[0], quat[1], quat[2], quat[3] = quat[1], quat[2], quat[3], quat[0]
                omega = real_pos[15:18]
                
                real_pos = np.concatenate([[t * dt * sim_steps], pos, quat, vel, omega, action])
                observations.append(real_pos)

        if save == True:
            save_path = './quadrotor_env_data/'
            try:
                os.makedirs(save_path, exist_ok=True)
            except FileExistsError:
                # directory already exists
                pass
            np.savetxt(save_path + 'observations.csv', observations, delimiter=',')

        if plot == True:
            TIME, X, Y, Z, QX, QY, QZ, QW, VX, VY, VZ, Roll_rate, Pitch_rate, Yaw_rate, t0, t1, t2, t3 = \
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
            
            observations = np.array(observations)

            plot_comp = {
                'Omega': {'roll rate': Roll_rate, 'pitch rate': Pitch_rate, 'yaw rate': Yaw_rate},
                'Position': {'X': X, 'Y': Y, 'Z': Z},
                'Velocity': {'VX': VX, 'VY': VY, 'VZ': VZ}, 
                'Orientation': {'qx': QX, 'qy': QY, 'qz': QZ, 'qw': QW}, 
                'Actions': {'t0': t0, 't1': t1, 't2': t2, 't3': t3}
            }

            total_subplots = len(plot_comp)
            current_plot = 1
            for obs_comp in plot_comp:
                plt.subplot(total_subplots, 1, current_plot)
                for comp in plot_comp[obs_comp]:
                    plt.plot(observations[:,TIME], observations[:,plot_comp[obs_comp][comp]], '-', label=comp)
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
        'param_file',
        type=str,
        help="provide a param.pkl file"
    )

    parser.add_argument(
        '-policy_type', 
        type=str,
        default='lstm',
        help='the type of network'
    )

    parser.add_argument(
        '-traj',
        type=str,
        default=None,
        help='a trajectory file'    
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
    test_rollout(args.param_file, traj_file=args.traj, policy_type=args.policy_type, save=args.save, plot=args.plot)


if __name__ == '__main__':
	main(sys.argv)
