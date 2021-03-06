import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import time
import csv


def test_rollout(
        param_file, 
        traj_file=None,
        repeat_traj=1,
        render=False, 
        rollouts_num=1,
        dt=0.005,
        sim_steps=2,
        ep_time=7.0,
        render_each=3,
        use_noise=False, # if it's true, use what the env already has
        random_init=False,  # if it's true, use what the env already has
        random_quad=False,  # if it's true, use what the env already has
        excite=False, # whether to perturb the quad
        save=False, 
        plot=False,
        nodisp=False,
    ):

    # moved these here so that pyvirtualdisplay is imported before them

    from quad_sim.quad_utils import R2quat
    from quad_sim.quad_models import crazyflie_params
    from quad_sim.quadrotor import QuadrotorEnv
    from quad_sim.quadrotor_randomization import R2quat
    from quad_gen.utils import rpy2R, R2rpy
    import tqdm

    quad_traj = []


    if traj_file != None:
        print("Reading trajectory...")
        traj = np.loadtxt(traj_file, delimiter=',')
        num_goals_render = 32
        render_goals = np.array(traj[0:num_goals_render]).flatten()
        # traj_freq = 1 ## every 1 time step(s), the goal is set to the next point in the traj

    import tensorflow as tf
    with tf.Session() as sess:
        print("extrating parameters from file %s ..." % param_file)
        params = joblib.load(param_file)

        env = QuadrotorEnv(num_goals=2, obs_repr="nxyz_vxyz_R_omega_reached", goal_tolerance=0.1, rew_type="default", traj=render_goals)
        #env = QuadrotorEnv(num_goals=2, obs_repr=)

        policy = params['policy']
        num_goals = 2 #params['env'].num_goals
        
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

        env.dt = dt
        env.sim_steps = sim_steps
        env.ep_len = int(ep_time / (dt * sim_steps))
        ## output import info about the environment
        print('#############################')
        print('Episode time: {}'.format(ep_time))
        print('Integration step: {}'.format(dt))
        print('Simulation step: {}'.format(sim_steps))
        print('#############################')
        ## ========================================
        
        ## Diagnostics
        observations = []
        traj_ptr = 0
        for rollouts_id in tqdm.tqdm(range(rollouts_num)):
            s = env.reset()
            policy.reset()
            
            ## reset the goal to x:0, y:0 z:0
            # env.goal = np.array([0., 0., 1])
            
            dynamics = env.dynamics
            print("thrust to weight ratio set to: {}, and max thrust is {}".format(dynamics.thrust_to_weight, dynamics.thrust_max))

            ## set the initial state
            if not random_init:
                dynamics.set_state(init_pos, init_vel, init_rot, init_omega)
                dynamics.reset()
                if traj_file is not None:
                    env.scene.reset(env.traj, dynamics)
                else:
                    env.scene.reset(env.goal, dynamics)
                s = env.state_vector(env)
                 
            times_traj_completed = 0
            t = 0
            done = False
            while True:
                # =================================
                if render and (t % render_each == 0): env.render()

                if traj_file != None:
                    env.goal = np.array([traj[i % len(traj)] for i in range(traj_ptr, traj_ptr + num_goals)]).flatten()
                    if env.curr_goal == num_goals_render:
                        env.curr_goal = 0
                        env.scene.recolor_goals()
                    # if t * dt >= ep_time:
                    #     # update goal
                    #     done = True
                    
                    action = policy.get_action(s)[1]['mean']
                    s, r, _, info = env.step(action)

                    state = env.state_vector(env)
                    reached_one_goal = state[-1 * num_goals]
                    done = state[-1]

                    if done:
                        times_traj_completed += 1
                        if times_traj_completed < repeat_traj:
                            done = False

                    if reached_one_goal:
                        if traj_ptr < (repeat_traj * traj.shape[0] - num_goals):
                            traj_ptr += 1
                            env.reached = np.zeros(num_goals)
                            reached_one_goal = False

                    quad_traj.append(env.dynamics.pos)

                elif excite and t % 1000 == 0:
                    ## change the goal every 100 time step
                    env.goal = np.concatenate([
                        np.random.uniform(low=-1, high=1, size=(2,)),
                        np.random.uniform(low=1, high=1, size=(1,))
                    ])
                    action = policy.get_action(s)[1]['mean']
                    s, r, done, info = env.step(action)
                else:
                    action = policy.get_action(s)[1]['mean']
                    s, r, done, info = env.step(action)

                if done: break  
                t += 1
                
                # ========== Diagnostics ==========
                real_pos = env.state_vector(env)
                pos = real_pos[0:3] + env.goal[:3]
                vel = real_pos[3:6]
                quat = R2quat(real_pos[6:15])
                # reformat to [x, y, z, w]
                quat[0], quat[1], quat[2], quat[3] = quat[1], quat[2], quat[3], quat[0]
                rpy = R2rpy(real_pos[6:15])
                omega = real_pos[15:18]
                
                # real_pos = np.concatenate([[t * dt * sim_steps], pos, quat, vel, omega, action])
                # real_pos = np.concatenate([[t * dt * sim_steps], 
                #     pos, quat, info['obs_comp']['Vxyz'][0], omega, info['obs_comp']['Act'][0], [r]])
                real_pos = np.concatenate([[t * dt * sim_steps], 
                    pos, rpy, vel, omega, env.goal, action])
                observations.append(real_pos)
        
        save_traj_to_csv(quad_traj, traj_file.split('/')[-1][:-4])

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
            
            observations = np.array(observations)

            plot_comp = {
                'Omega': {'roll rate': Roll_rate, 'pitch rate': Pitch_rate, 'yaw rate': Yaw_rate},
                'Position': {'X': X, 'Target X': Xt, 'Y': Y, 'Target Y': Yt, 'Z': Z, 'Target Z': Zt},
                'Velocity': {'VX': VX, 'VY': VY, 'VZ': VZ}, 
                # 'Orientation': {'qx': QX, 'qy': QY, 'qz': QZ, 'qw': QW}, 
                'Orientation': {'R': Roll, 'P': Pitch, 'Y': Yaw}, 
                'Actions': {'t0': t0, 't1': t1, 't2': t2, 't3': t3},
                'Reward' : {'reward': reward}
            }

            total_subplots = len(plot_comp)
            current_plot = 1
            for obs_comp in plot_comp:
                plt.subplot(total_subplots, 1, current_plot)
                for comp in plot_comp[obs_comp]:
                    plt.plot(observations[:,plot_comp[obs_comp][comp]], '-', label=comp)
                plt.xlabel('Time [s]')
                plt.ylabel(obs_comp)    
                plt.legend(loc=9, ncol=3, borderaxespad=0.)
                
                current_plot += 1

            plt.show()
        

    print("##############################################################")

def save_traj_to_csv(traj, file_name):
    with open(file_name + '_quad_traj.csv', 'w') as csvfile:
        traj_file = csv.writer(csvfile)
        for pos in traj:
            traj_file.writerow(pos)


def main(argv):
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'param_file',
        type=str,
        help="provide a param.pkl file"
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
        default=1,
        help='number of rollouts'    
    )

    parser.add_argument(
        '-ep_time',
        type=int,
        default=7,
        help='episode time'    
    )

    parser.add_argument(
        '-dt',
        type=float,
        default=0.005,
        help='time step size'    
    )

    parser.add_argument(
        '-sim_steps',
        type=int,
        default=2,
        help='controller step'    
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
        '--random_quad', 
        action='store_true',
        help='whether to use the env quad parameter'
    ) 

    parser.add_argument(
        '--use_noise', 
        action='store_true',
        help='whether to use noise'
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

    parser.add_argument(
        '--nodisp',
        action='store_true',
        help='use virtual display, render won\'t show'
    )

    parser.add_argument(
        '-repeat_traj',
        type=int,
        default=1,
        help='number of times to repeat the trajectory'
    )
    args = parser.parse_args()

    if args.nodisp:
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1400,900))
        display.start()

    print('Running test rollout...')
    test_rollout(
        args.param_file, 
        traj_file=args.traj,
        repeat_traj=args.repeat_traj,
        render=args.render, 
        rollouts_num=args.rollouts_num,
        dt=args.dt,
        sim_steps=args.sim_steps,
        ep_time=args.ep_time,
        use_noise=args.use_noise,
        random_init=args.random_init,
        random_quad=args.random_quad,
        excite=args.excite,
        save=args.save, 
        plot=args.plot
    )


if __name__ == '__main__':
	main(sys.argv)
