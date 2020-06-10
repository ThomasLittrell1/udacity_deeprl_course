import numpy as np
from collections import defaultdict


class Agent:
    def __init__(self, nA=6, gamma=1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1
        self.alpha = 0.1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy = self.get_policy_for_state(state)
        return np.random.choice(np.arange(self.nA), p=policy)

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
        # policy = self.get_policy_for_state(state)
        # expected_Q = (policy * self.Q[next_state]).sum()
        # new_estimate = reward + self.gamma * expected_Q
        best_action = self.Q[next_state].argmax()
        new_estimate = reward + self.gamma * self.Q[next_state][best_action]
        self.Q[state][action] += self.alpha * (new_estimate - self.Q[state][action])

    def get_policy_for_state(self, state):
        result = np.full(self.nA, self.epsilon / self.nA)
        best_action = self.Q[state].argmax()
        result[best_action] += 1 - self.epsilon
        return result
