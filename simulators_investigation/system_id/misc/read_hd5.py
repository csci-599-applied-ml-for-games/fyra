import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

from calculating_t2w import scale_action
from scipy import signal

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
	'file', 
	type=str,
	help='data file'
)

args = parser.parse_args()

file_path = args.file

obs_comp = {"Omega", "Vxyz", "R22", "R", "Inertial", \
			"Act", "ClippedAct", "TorqueAct", "dt", \
			"MaxThrust"}

f = h5py.File(file_path, 'r')

group = f['traj_data']['0000']

## convert the file data to a dictionary for easier access
## structure of the h5d file
## '0000'->itr->traj_id->env_infos->obs_comp

data = dict()

for itr in group.keys():
	# print('=====itr: '+itr+'=====')
	itr_data = group[itr]
	data[itr] = dict()
	for traj_id in itr_data.keys():
		# print('=== traj id:'+traj_id+'===')
		traj_data = itr_data[traj_id]
		data[itr][traj_id] = dict()
		for attr in traj_data['env_infos']['obs_comp']:
			attr_data = traj_data['env_infos']['obs_comp'][attr]
			data[itr][traj_id][attr] = attr_data.value
		for attr in traj_data['env_infos']['dyn_params']:
			attr_data = traj_data['env_infos']['dyn_params'][attr]
			data[itr][traj_id][attr] = attr_data.value

## plot some trajectories
plot_comp = {
	'Omega': {'roll rate': 0, 'pitch rate': 1, 'yaw rate': 2},
	'Vxyz': {'VX': 0, 'VY': 1, 'VZ': 2}, 
	'R22': {'R22': 0}, 
	'TorqueAct': {'t0': 0, 't1': 1, 't2': 2, 't3': 3},
	#'ClippedAct': {'t0': 0, 't1': 1, 't2': 2, 't3': 3},
	#'Act': {'t0': 0, 't1': 1, 't2': 2, 't3': 3},
	# 't2w': {'t2w':0}
}

b, a = signal.butter(8, 0.01)

for itr in data.keys():
	for traj_id in data[itr].keys():
		traj_data = data[itr][traj_id]

		total_subplots = len(plot_comp) # + 4
		current_plot = 1
		for obs_comp in plot_comp:
			traj_data[obs_comp] = traj_data[obs_comp].squeeze()
			print(traj_data[obs_comp])
			plt.subplot(total_subplots, 1, current_plot)
			for comp in plot_comp[obs_comp]:
				try:
					plt.plot(traj_data[obs_comp][:, plot_comp[obs_comp][comp]], '-', label=comp)
				except:
					plt.plot(traj_data[obs_comp], '-', label=comp)
				plt.xlabel('Time [s]')
				plt.ylabel(obs_comp)
				plt.legend(loc=9, ncol=3, borderaxespad=0.)

			current_plot += 1

		"""
		t2w_overtime = []
		t2w_gt = []
		acceleration = []
		action_sum = []

		# traj_data['Act'] = signal.filtfilt(b, a, traj_data['Act'], axis=0, padlen=150)
		# traj_data['Vxyz'] = signal.filtfilt(b, a, traj_data['Vxyz'], axis=0, padlen=150)
		for i in range(20, len(traj_data['Vxyz'])):
			## calculate the velocity change
			v_diff = traj_data['Vxyz'][i] - traj_data['Vxyz'][i-1]
			acc = v_diff / 0.01

			## calculate the normalized thrust
			action = np.sum(traj_data['ClippedAct'][i-1].reshape(1, 4)) / 4 #np.sum(scale_action(traj_data['ClippedAct'][i-1].reshape(1, 4))) / 4
			action_sum.append(action)

			## acceleration after compensating for gravity
			g = np.array([0., 0., 9.81])
			acc = acc - g

			## total acceleration in the direction of the total thrust applied
			acc = np.linalg.norm(acc)
			acceleration.append(acc)

			## assuming thrust and action is linearly related
			t2w = acc / (action * 9.81)

			t2w_overtime.append(t2w)
			t2w_gt.append(traj_data['t2w'][i])

		t2w_comp = {'t2w': t2w_overtime, 't2w_gt': t2w_gt, 'acc': acceleration, 'action': action_sum}
		for comp in t2w_comp:
			plt.subplot(total_subplots, 1, current_plot)
			plt.plot(t2w_comp[comp], '-', label=comp)
			plt.xlabel('Time [s]')
			plt.ylabel(comp)
			plt.legend(loc=9, ncol=3, borderaxespad=0.)
			current_plot += 1

		print('t2w_gt: {}, mean predicted: {}'.format(t2w_gt[0], np.mean(t2w_overtime)))
		"""
		plt.show()
		# break
	break