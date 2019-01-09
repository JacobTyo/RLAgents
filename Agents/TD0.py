import numpy as np
import random


class TD0:
    # Temporal Difference Learning (TD(0)) with epsilon-greedy policy
    def __init__(self, state_space, action_space, epsilon, alpha, gamma):
        # inputs
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor

        # needed learning variables
        self.estimated_values = np.zeros(16)
        self.estimated_values[15] = 1
        self.current_state = np.zeros(2)
        self.previous_state = -1

    def act(self, observation, reward, new_episode):
        update = True
        if new_episode:
            self.estimated_values[self.previous_state] = self.estimated_values[self.previous_state] + self.alpha * \
                                                         (reward + self.gamma * 1 -
                                                          self.estimated_values[self.previous_state])
            update = False

        if self.previous_state >= 0 and update:
            self.estimated_values[self.previous_state] = self.estimated_values[self.previous_state] + self.alpha * \
                                                         (reward + self.gamma*self.estimated_values[observation] -
                                                          self.estimated_values[self.previous_state])
        self.previous_state = observation

        self.current_state = self.state_to_vec(observation)

        # to get the next action we should take, look at the value of all available next states
        max_action_value = np.zeros(2)
        best_action_found = False

        best_action_found, max_action_value = self.check_state_left(best_action_found,
                                                                    max_action_value)
        best_action_found, max_action_value = self.check_state_down(best_action_found,
                                                                    max_action_value)
        best_action_found, max_action_value = self.check_state_right(best_action_found,
                                                                     max_action_value)
        best_action_found, max_action_value = self.check_state_up(best_action_found,
                                                                  max_action_value)

        action = max_action_value[0]

        # if an action was not found, or if we are under epsilon, select an action randomly
        if not best_action_found or random.random() < self.epsilon:
            # select an action randomly
            action = random.randint(0, 3)

        return action, self.estimated_values

    @staticmethod
    def state_to_vec(observation):
        current_state = np.zeros(2)
        current_state[0] = int(observation / 4)
        current_state[1] = observation % 4
        return current_state

    def check_state_left(self, best_action_found, max_action_value):
        # test left
        if self.current_state[0] - 1 >= 0:
            # we can go left
            check_this_state = (self.current_state[0] - 1) * 4 + self.current_state[1]
            checked_value = self.estimated_values[int(check_this_state)]
            if checked_value > max_action_value[1]:
                max_action_value = [0, checked_value]
                best_action_found = True

        return best_action_found, max_action_value

    def check_state_down(self, best_action_found, max_action_value):
        # test down
        if self.current_state[1] + 1 <= 3:
            # we can go down
            check_this_state = self.current_state[0] * 4 + (self.current_state[1] + 1)
            checked_value = self.estimated_values[int(check_this_state)]
            if checked_value > max_action_value[1]:
                max_action_value = [1, checked_value]
                best_action_found = True

        return best_action_found, max_action_value

    def check_state_right(self, best_action_found, max_action_value):
        # test right
        if self.current_state[0] + 1 <= 3:
            # we can go right
            check_this_state = (self.current_state[0] + 1) * 4 + self.current_state[1]
            checked_value = self.estimated_values[int(check_this_state)]
            if checked_value > max_action_value[1]:
                max_action_value = [2, checked_value]
                best_action_found = True

        return best_action_found, max_action_value

    def check_state_up(self, best_action_found, max_action_value):
        # test down
        if self.current_state[1] - 1 >= 0:
            # we can go down
            check_this_state = self.current_state[0] * 4 + (self.current_state[1] - 1)
            checked_value = self.estimated_values[int(check_this_state)]
            if checked_value > max_action_value[1]:
                max_action_value = [3, checked_value]
                best_action_found = True

        return best_action_found, max_action_value
