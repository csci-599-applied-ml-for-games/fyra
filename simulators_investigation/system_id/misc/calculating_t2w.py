#!/usr/bin/env python

import argparse
import numpy as np
import os
import sys

from flight_data_reader import reader
import matplotlib.pyplot as plt
from scipy import signal

def normalize(v):
	norm = np.linalg.norm(v)
	if norm == 0: 
		return v
	return v / norm

def scale_action(action):
	"""
	scaled and clip the actions and use as training data
	"""
	action_dim = action.shape[1]
	action_low = np.zeros(action_dim) 
	action_high = np.ones(action_dim)
	action_scale = 0.5 
	action_bias = 1.0

	## scaled and clip the action and use as training data
	action = action_scale * (action + action_bias)
	action = np.clip(action, a_min=action_low, a_max=action_high)

	return action

def load_data(real_flight_file):
	assert os.path.isfile(real_flight_file) == True, "data path doesn't exist"

	log_data = reader.decode(real_flight_file, plot=False)

	"""
	clean the data such that 
	it only contains valid flight data
	"""
	## eliminate ground effect and clean data
	z_threshold = 0.05
	start = end = 0
	found_start = False
	for z in log_data['ctrltarget.z']:
		if z > z_threshold and found_start == False:
			found_start = True
		if found_start == False:
			start += 1
		if z <= z_threshold and found_start == True:
			break
		end += 1

	for key in log_data:
		log_data[key] = log_data[key][start:end]

	b, a = signal.butter(8, 0.01)

	## linear velocity
	abs_vx, abs_vy, abs_vz = log_data['stateEstimate.vx'], log_data['stateEstimate.vy'], log_data['stateEstimate.vz']
	## orientation
	qx, qy, qz, qw = log_data['stateEstimate.qx'], log_data['stateEstimate.qy'], log_data['stateEstimate.qz'], log_data['stateEstimate.qw'] 
	## neural network raw output
	thrust0, thrust1, thrust2, thrust3 = log_data['ctrlNN.out0'], log_data['ctrlNN.out1'], log_data['ctrlNN.out2'], log_data['ctrlNN.out3']

	abs_v = np.column_stack([abs_vx, abs_vy, abs_vz])
	actions = scale_action(np.column_stack([thrust0, thrust1, thrust2, thrust3]))
	rots = quat2R(np.column_stack([qx, qy, qz, qw]))

	return signal.filtfilt(b, a, abs_v, axis=0, padlen=150), signal.filtfilt(b, a, actions, axis=0, padlen=150) ##, rots

def load_data_sim(real_flight_file):
	assert os.path.isfile(real_flight_file) == True, "data path doesn't exist"

	TIME = 0 
	X, Y, Z = 1, 2, 3
	Roll, Pitch, Yaw = 4, 5, 6
	VX, VY, VZ = 7, 8, 9 
	Roll_rate, Pitch_rate, Yaw_rate = 10, 11, 12
	Xt, Yt, Zt = 13, 14, 15
	t0, t1, t2, t3 = 16, 17, 18, 19
	# pred_obs_idx = [19+i for i in range(1, 2*len(out_comp), 2)]

	log_data = np.loadtxt(real_flight_file, delimiter=',')

	## eliminate ground effect and clean data
	z_threshold = 0.1
	start = end = 0
	found_start = False
	for z in log_data[:, Z]:
		if z > z_threshold and found_start == False:
			found_start = True
		if found_start == False:
			start += 1
		if z <= z_threshold and found_start == True:
			break
		end += 1

	log_data = log_data[start:end]

	abs_v = log_data[:, VX:VZ+1]
	actions = scale_action(log_data[:, t0:t3+1])
	# rots = quat2R(log_data[:, QX:QW+1]) 

	return abs_v, actions ##, rots

def plot_t2w(abs_v, actions):
	assert abs_v.shape[0] == actions.shape[0], "mismatch length"
	# assert abs_v.shape[0] == rots.shape[0], "mismatch length"

	## the frequency at which the data was collected
	dt = 0.01
	GRAV = 9.81

	## diagonsis
	t2w_overtime = []
	acceleration = []
	action_sum = []

	for i in range(1, abs_v.shape[0]):
		## calculate the velocity change
		v_diff = abs_v[i] - abs_v[i-1]
		acc = v_diff / dt

		## calculate the normalized thrust
		action = np.sum(actions[i-1]) / 4
		action_sum.append(action)

		## rotation matrix
		# rot = np.array(rots[i]).reshape((3, 3))

		## acceleration after compensating for gravity
		g = np.array([0., 0., -GRAV])
		acc = acc - g

		## total acceleration in the direction of the total thrust applied
		acc = np.linalg.norm(acc)
		acceleration.append(acc)

		## assuming thrust and action is linearly related
		t2w = acc / (action * GRAV)

		t2w_overtime.append(t2w)

	t2w_overtime = np.array(t2w_overtime)
	print(np.mean(t2w_overtime, axis=0), np.std(t2w_overtime, axis=0))

	plt.figure(0)
	# plt.plot(np.linspace(1, len(t2w_overtime),len(t2w_overtime)), t2w_overtime[:, 0], label='t2w x')
	# plt.plot(np.linspace(1, len(t2w_overtime),len(t2w_overtime)), t2w_overtime[:, 1], label='t2w y')
	plt.plot(np.linspace(1, len(t2w_overtime),len(t2w_overtime)), t2w_overtime, label='t2w z')
	plt.plot(np.linspace(1, len(t2w_overtime),len(t2w_overtime)), abs_v[1:,0], label='vx')
	plt.plot(np.linspace(1, len(t2w_overtime),len(t2w_overtime)), abs_v[1:,1], label='vy')
	plt.plot(np.linspace(1, len(t2w_overtime),len(t2w_overtime)), abs_v[1:,2], label='vz')
	plt.plot(np.linspace(1, len(t2w_overtime),len(t2w_overtime)), action_sum, label='action sum')
	plt.plot(np.linspace(1, len(t2w_overtime),len(t2w_overtime)), acceleration, label='acceleration')
	plt.legend()
	plt.show()


def main(argv):
	
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument(
		'real_flight_file', 
		type=str,
		help='provide real flight data for training auxiliary dynamics'
	)

	parser.add_argument(
		'--sim',
		action='store_true',
		help='if it is simulation data'
	)

	args = parser.parse_args()

	if args.sim:
		abs_v, actions = load_data_sim(args.real_flight_file)
	else:
		abs_v, actions = load_data(args.real_flight_file)

	plot_t2w(abs_v, actions)

if __name__ == '__main__':
	main(sys.argv)