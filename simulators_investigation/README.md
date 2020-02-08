# simulators_investigation
This repo is for testing trained policies (controller and OSI) in the Python simulation.
Follow the instructions in [here](https://github.com/amolchanov86/quad_dynalearn) to set up your environment.

## Activate your virtual environment (the same one you use to train your networks)

## networks/
This folder contains implementation of MLP and LSTM neural network architectures. Running OSI where you will most likely encounter LSTM networks, due to some reasons (that I don't remember) related to TensorFlow you will have to use the LSTM implementation. Otherwise, you will rarely need this folder. I admit this is not a perfect solution, but it works.

## osi/
This folder currently contains only one script which is used to test trained OSI networks. It works as follow:
1. Have a trained OSI network ready. It should be in a pickle file, i.e params.pkl)
2. Run `python test_osi.py [params.pkl]`. Replace the `[params.pkl]` with the actual path to the file.
3. You can specify a number of options. I will explain a couple here. You can see all avaliable options by running `python test_osi.py -h`.

### `test_osi.py` options
1. `-policy_param_file`: with this option, you can specify an alternative control network to use. By default, `params.pkl` should have a policy and `test_osi.py` automatically loads that policy. But you can specify one if you'd like.
2. `-traj`: you can provide a trajectory file such that the quad will follow the trajectory. Some exmaple trajectory files are located under `/simulator_discrepancy/trajs`. This is extremely useful when you want to see how the OSI performs when flying a trajectory, i.e a figure 8.
3. `--use_osi_output`: this option tells the script to replace the last couple elements of the returned state with the OSI prediction. When you don't specify enable this option, the state contains the ground truth of what the OSI is trying to predict. Note: if you network doesn't expect the OSI predictions, this option will very likely cause the quad to crash miserably.
4. `--random_quad`: this is not a good name. By specify this option, you tell the script to use whatever dynamics parameters the environment used to train your OSI (you most likely randomize some dynamics parameters, this is where the name comes from). Otherwise, it will use the crazyflie dynamics parameters.
5. `--plot`: please use this option, otherwise the script won't show you any outputs. By enabling this option, the script will output a plot showing the states of the quad and the OSI prediction against the ground truth for all the trajectories it flew.
6. An example: `python test_osi.py [path to params.pkl] --plot --rollouts_num 10 --random_init --random_quad --ep_time 5`. This changes the episode time to 5 seconds and fly 10 trajectories (randomly initialize the quad 10 times and each time it flies for 5 seconds). And finally output a plot showing the OSI prediction results.

## simulator_discrepancy
This folder contains many files but the most important one is the `test_controller.py`.
1. Have a trained controller network ready. It should be in a pickle file, i.e params.pkl)
2. Run `python test_controller.py [params.pkl]`. Replace the `[params.pkl]` with the actual path to the file.
3. You can specify a number of options. I will explain a couple here. You can see all avaliable options by running `python test_controller.py -h`.

### `test_controller.py` options
1. `-dt`: this determines the simulation frequency. Default to 0.005 (200 Hz). It overrides the training environment parameters. 
2. `-sim_steps`: this argument determines the controller frequency. Default to 2 (100 Hz), meaning that every control command the environment step two time steps (dt)

# Please create a new branch to modify the scripts if you want to plot different outputs.
