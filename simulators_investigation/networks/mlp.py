import numpy as np
import matplotlib.pyplot as plt

from simulators_investigation.networks.policy import Policy

class mlp_policy(Policy):
    """
    Assumes to be a 2 layer network
    """
    def __init__(self, policy, plot=False):
        super().__init__(plot=plot)
        # tf_parameters is a list containing all the variables in order
        tf_parameters = policy.get_params()

        self.extract_parameters(tf_parameters)

    def extract_parameters(self, tf_parameters):
        self.hidden_1_w = tf_parameters[0].eval()
        self.hidden_1_b = tf_parameters[1].eval()
        self.hidden_2_w = tf_parameters[2].eval()
        self.hidden_2_b = tf_parameters[3].eval()
        self.output_w = tf_parameters[4].eval()
        self.output_b = tf_parameters[5].eval()

    def network_evaluate(self, obs):
        return np.tanh(np.tanh(obs @ self.hidden_1_w + self.hidden_1_b) @ self.hidden_2_w + self.hidden_2_b) @ self.output_w + self.output_b

    def plot_states(self):
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.bar(np.arange(18), self.obs)
        plt.ylabel('observation')

        plt.subplot(2, 1, 2)
        plt.bar(np.arange(4), self.action)
        plt.ylabel('action')

        plt.show(block=False)
        plt.pause(0.0001)