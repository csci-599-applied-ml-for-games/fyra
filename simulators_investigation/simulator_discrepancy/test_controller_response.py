##
## Test the network response of a single variable
##
from gym_art.quadrotor.quad_utils import R2quat
from gym_art.quadrotor.quad_models import *
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import time

def test_response(
        param_file, 
        target_variable, ## x, y, z, vx, vy, vy, roll, pitch, yaw
        freq=1, ## Hz
        amp=1,
        dt=0.005,
        sim_steps=2,
        ep_time=7,
        save=True, 
        plot=True):

    state_variables = {'x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw'}
    assert target_variable in state_variables, "%s is not a state" % target_variable

    import tensorflow as tf
    with tf.Session() as sess:
        print("extrating parameters from file %s ..." % param_file)
        params = joblib.load(param_file)

        policy = params['policy']

        ## generate a sinusoidal signal 
        samples = np.linspace(0, ep_time, int(ep_time/(dt*sim_steps)), endpoint=False)
        signal = amp * np.sin(freq * 2 * np.pi * samples)

        ## create quadrotor state
        states = dict()
        for variable in state_variables:
            states[variable] = np.zeros(len(samples))
        states[target_variable] = signal

        ## Diagnostics
        observations = []

        for t in range(len(samples)):
            pos = np.array([states['x'][t], states['y'][t], states['z'][t]])
            vel = np.array([states['vx'][t], states['vy'][t], states['vz'][t]])
            rot = np.eye(3).flatten()
            omega = np.array([states['roll'][t], states['pitch'][t], states['yaw'][t]])

            s = np.concatenate([pos, vel, rot, omega])

            action = policy.get_action(s)[1]['mean']

            real_pos = np.concatenate([[t], pos, vel, omega, action])
            observations.append(real_pos)

        if plot == True:
            TIME, X, Y, Z, VX, VY, VZ, Roll_rate, Pitch_rate, Yaw_rate, t0, t1, t2, t3 = \
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
            
            observations = np.array(observations)

            plot_comp = {
                'Omega': {'roll rate': Roll_rate, 'pitch rate': Pitch_rate, 'yaw rate': Yaw_rate},
                'Position': {'X': X, 'Y': Y, 'Z': Z},
                'Velocity': {'VX': VX, 'VY': VY, 'VZ': VZ}, 
                'Actions': {'t0': t0, 't1': t1, 't2': t2, 't3': t3}
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
    
def main(argv):
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'param_file',
        type=str,
        help="provide a param.pkl file"
    )

    parser.add_argument(
        'variable',
        type=str,
        help="which variable to test"
    )

    parser.add_argument(
        '-freq', 
        type=float, 
        default=1, ## Hz
        help="sinusoid frequency"
    )

    parser.add_argument(
        '-amplitude',
        type=float,
        default=1, 
        help="sinusoid amplitude"
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
        default=0.01,
        help='time step size'    
    )

    parser.add_argument(
        '-sim_steps',
        type=int,
        default=1,
        help='controller step'    
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
    test_response(
        args.param_file, 
        args.variable,
        freq=args.freq,
        amp=args.amplitude,
        dt=args.dt,
        sim_steps=args.sim_steps,
        ep_time=args.ep_time,
        save=args.save, 
        plot=args.plot
    )


if __name__ == '__main__':
	main(sys.argv)