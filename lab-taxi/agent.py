import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.02
        self.gamma = 0.98
        self.epsilon = 0.99
        self.min_epsilon = 0.1
        self.e_decay_rate = 0.9

    def greedy_policy(self, state):
        """ Policy that returns the best action according to q values.
        """
        if state not in self.Q:
            self.Q[state] = [0.0] * self.nA
        return int(np.argmax(self.Q[state]))

    def e_greedy_policy(self, state, epsilon):
        """ Policy that returns the best action according to q values with
        (epsilon/#action) + (1 - epsilon) probability and any other action with
        probability episolon/#action.
        """
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.nA)
        else:
            return self.greedy_policy(state)

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return self.e_greedy_policy(state, 0.01)


    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        action = self.e_greedy_policy(state, self.epsilon)
        greedy_action = self.greedy_policy(next_state)
        tde = reward+self.gamma*self.Q[next_state][greedy_action]
        self.Q[state][action] += self.alpha*(tde-self.Q[state][action])
        self.epsilon = max(self.epsilon*self.e_decay_rate, self.min_epsilon)