import numpy as np

class Policy():
    """
    re-implement neural network structures to make sure 
    we have the correct computation graph
    """
    def __init__(self, plot=False):
        self.plot = plot
        self.step = 0
        self.average_action = np.zeros(4)
        self.previous_action = np.zeros(4)

    def extract_parameters(self, tf_parameters):
        raise NotImplementedError

    def network_evaluate(self, obs):
        raise NotImplementedError

    def get_action(self, obs):
        self.action = self.network_evaluate(obs)
        self.obs = obs

        if self.plot:
            self.plot_states()

        ## calculate the average actions
        # self.step += 1
        # self.average_action = ((self.step - 1) * self.average_action + self.action) / self.step

        # ## calculate the action change
        # action_change = np.abs(self.previous_action - self.action)
        # self.previous_action = self.action 

        return self.action #, self.average_action, action_change

    def plot_states(self):
        raise NotImplementedError