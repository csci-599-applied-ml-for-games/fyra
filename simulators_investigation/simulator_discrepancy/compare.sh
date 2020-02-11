#!/usr/bin/env bash

# extract data that was collected during a Ros run
if [[ $# -eq 1 ]]; then
	cd ./extract_data
	python extract_data_from_ros.py $1
	cd ..
fi

# compare the ros output with the model output by feeding into the same states
# there are two dummy arguments to the script
python test_controller.py 1 ../models/ppo_crazyflieperturb_0.1_noisy_nodamp_yaw0_rand_0.3/params.pkl ./extract_data/controller_input.csv ./extract_data/controller_output.csv



