import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # if you want to run this on a cpu instead

class DQN:
    """
    Essentially, Deep Q-Learning tries to emulate a Q-table for complex situations.
    A Q-table is a map of all possible states of the environment, with the immediate and long-term rewards that
    could be gained with each action for each state.
    In other words, a Q-table complete provides the absolute best move for every possible situation.
    However, for more complex games with a large order of magnitude of states, generating a Q-table is time-consuming
    and resource-intensive, so we try to get a good enough approximation instead.
    """
    def __init__(self, learning_rate=0.5, discount=0.95, exploration_rate=1.0, iterations=10000, layer_size=32, filepath=None):
        """
        Q-table formula approximated through Deep Q-Learning:
        Q(s, a) = Q(s, a) + learning_rate * [reward + discount * max_expected_Q(s', a) - Q(s, a)]

        Here, s is state, a is action, reward is the immediate reward from the action, Q(s', a) is the future reward
        that the agent would get from the optimal choice of actions from those made available after taking action a.
        Q(s, a) is simply the current Q-table value of taking action a during state s.

        :param exploration_rate: fraction chance that agent will take random action to explore environment
        :param iterations: number of experiences over which exploration rate is decreased to 0
        :param layer_size: number of neurons in each hidden layer of the neural network
        """
        self.learning_rate = learning_rate
        # a higher discount rate allows rewards of a good action to "seep through" to the actions that led to it,
        # which is important in a game like pong where only a few frames really decide whether a series of actions as
        # a whole were good or not.
        self.discount = discount
        self.exploration_rate = exploration_rate
        self.iterations = iterations  # only defining under self so that we can save and load models in properly later
        self.exploration_delta = exploration_rate / self.iterations  # the amount that exploration rate is reduced by after each experience

        self.input_count = 6
        self.output_count = 3  # up, stay, or down; all mutually exclusive outputs

        if filepath:
            self.model = keras.models.load_model(filepath)
        else:
            # defining the neural network - two hidden layers, each of layer_size neurons
            self.model = keras.Sequential()
            initializer = keras.initializers.Zeros()  # for initializing weights and biases to 0
            self.model.add(keras.layers.Dense(layer_size, input_shape=(self.input_count,), kernel_initializer=initializer,
                                              bias_initializer=initializer, activation="sigmoid"))
            self.model.add(keras.layers.Dense(layer_size, kernel_initializer=initializer, bias_initializer=initializer,
                                              activation="sigmoid"))
            self.model.add(keras.layers.Dense(self.output_count, kernel_initializer=initializer, bias_initializer=initializer))

            optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)
            self.model.compile(loss="mse", optimizer=optimizer)  # builds the neural network

    # Ask model to estimate Q value for specific state (inference)
    def get_Q(self, state):
        """
        Used to return the approximated Q-table value from the neural network more conveniently
        :param state: list of the inputs used to define each "state" of the environment
        :return: list containing Q-table values for each action (i.e. list of 3 values for up, stay and down respectively).
                 Obviously, using argmax determines which of the three indexes has a higher value and should be picked.
        """
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        return np.array(self.model(state_tensor)[0])

    def get_next_action(self, state):
        if random.random() > self.exploration_rate:
            return np.argmax(self.get_Q(state))  # chooses the best option from knowledge gained so far
        else:
            # 0 is up, 1 is stay, 2 is down
            return random.randrange(0, 3)  # explores more by selecting a random move

    def train(self, old_state, action, reward, new_state):
        """
        Essentially calculates what the Q-table value would have been given the above params, then runs the optimizer
        that was defined in the constructor of the class to make the values of the Deep Q-Learning neural net approximate
        the actual Q-table better and better.
        :param old_state: state before the action was taken
        :param action: action taken (up, stay or down, i.e. 0, 1 or 2)
        :param reward: immediate reward for the action taken
        :param new_state: state after the action, used to calculate potential long-term rewards of the action
        """
        old_state_Q_values = self.get_Q(old_state)
        new_state_Q_values = self.get_Q(new_state)

        # Real Q value for the action we took. This is what we will train towards.
        # Recall the Q-table formula from the constructor!
        # This is slightly modified as the neural network itself takes care of applying the learning rate so we don't have to.
        old_state_Q_values[action] = reward + self.discount * np.amax(new_state_Q_values)

        # Outputs of training_input are optimized to move towards target_output (stored in old_state_Q_values)
        self.model.fit(np.array([old_state]), np.array([old_state_Q_values]), verbose=0)

    def update(self, old_state, new_state, action, reward):
        # Train our model with the results from the action taken
        self.train(old_state, action, reward, new_state)

        # Shift our exploration_rate toward zero so it eventually only takes the best options.
        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta
            self.iterations -= 1
