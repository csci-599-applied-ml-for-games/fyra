import matplotlib.pyplot as plt
import numpy as np

from simulators_investigation.networks.policy import Policy

class lstm_policy(Policy):
    """
    Assuming the following machanism:
        Incoming gate:    i(t) = sigmoid(x(t) @ W_xi + h(t-1) @ W_hi + b_i)
        Forget gate:      f(t) = sigmoid(x(t) @ W_xf + h(t-1) @ W_hf + b_f)
        Cell gate:        c(t) = f(t) * c(t - 1) + i(t) * tanh(x(t) @ W_xc + h(t-1) @ W_hc + b_c)
        Out gate:         o(t) = sigmoid(x(t) @ W_xo + h(t-1) @ W_ho + b_o)
        New hidden state: h(t) = o(t) * tanh(c(t))
    Args:
        policy: the lstm policy containing all the tensors
    """
    def __init__(self, policy, plot=False):
        super().__init__(plot=plot)
        # tf_parameters is a list containing all the variables in order
        tf_parameters = policy.get_params()

        self.extract_parameters(tf_parameters)

    def network_evaluate(self, obs):
        incoming_gate = self.sigmoid(obs @ self.W_xi + self.hidden_state @ self.W_hi + self.b_i)
        forget_gate = self.sigmoid(obs @ self.W_xf + self.hidden_state @ self.W_hf + self.b_f + 1.0)
        self.cell_state = forget_gate * self.cell_state + incoming_gate * np.tanh(obs @ self.W_xc + self.hidden_state @ self.W_hc + self.b_c)
        out_gate = self.sigmoid(obs @ self.W_xo + self.hidden_state @ self.W_ho + self.b_o)
        self.hidden_state = out_gate * np.tanh(self.cell_state)

        action = self.hidden_state @ self.W + self.b

        return action


    def extract_parameters(self, tf_parameters):
        self.hidden_state = tf_parameters[0].eval()
        self.cell_state = tf_parameters[1].eval()
        self.W_xi = tf_parameters[2].eval()
        self.W_hi = tf_parameters[3].eval()
        self.b_i = tf_parameters[4].eval()
        self.W_xf = tf_parameters[5].eval()
        self.W_hf = tf_parameters[6].eval()
        self.b_f = tf_parameters[7].eval()
        self.W_xc = tf_parameters[8].eval()
        self.W_hc = tf_parameters[9].eval()
        self.b_c = tf_parameters[10].eval()
        self.W_xr = tf_parameters[11].eval()
        self.W_hr = tf_parameters[12].eval()
        self.b_r = tf_parameters[13].eval()
        self.W_xo = tf_parameters[14].eval()
        self.W_ho = tf_parameters[15].eval()
        self.b_o = tf_parameters[16].eval()
        self.W = tf_parameters[17].eval()
        self.b = tf_parameters[18].eval()

        self.num_units = self.hidden_state.shape[0]

        self.init_hidden_state = self.hidden_state
        self.init_cell_state = self.cell_state

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x)) 

    def reset_internal_states(self):
        self.hidden_state = self.init_hidden_state
        self.cell_state = self.init_cell_state

    def plot_states(self):
        plt.clf()
        plt.subplot(4, 1, 1)
        plt.bar(np.arange(18), self.obs)
        plt.ylabel('observation')

        plt.subplot(4, 1, 2)
        plt.bar(np.arange(self.num_units), self.hidden_state)
        plt.ylabel('hidden state')

        plt.subplot(4, 1, 3)
        plt.bar(np.arange(self.num_units), self.cell_state)
        plt.ylabel('cell state')

        plt.subplot(4, 1, 4)
        plt.bar(np.arange(4), self.action)
        plt.ylabel('action')

        plt.show(block=False)
        plt.pause(0.0001)