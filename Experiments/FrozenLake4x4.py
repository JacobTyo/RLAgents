import gym
from Agents.TD0 import TD0
import matplotlib.pyplot as plt
from tqdm import tqdm


num_episodes = 10000
iterations = 1000

env = gym.make('FrozenLake-v0')

# initialize agent
agent = TD0(state_space=env.observation_space, action_space=env.action_space, epsilon=0.1, alpha=0.1, gamma=0.99)

observation = env.reset()
reward = 0

data = {'episode': [],
        'iteration': [],
        'action': [],
        'reward': [],
        'average_reward': [],
        'episode_length': [],
        'success': []}
new_episode = False

this_is_a_sloppy_workaround = True
estimates = []

# for each episode
for episode in tqdm(range(num_episodes)):
    # for each iteration
    for i in range(iterations):
        action, estimates = agent.act(observation, reward, new_episode)
        new_episode = False
        observation, reward, done, _ = env.step(action)
        data['episode'].append(episode)
        data['iteration'].append(i)
        data['action'].append(action)
        data['reward'].append(reward)
        if this_is_a_sloppy_workaround:
            data['average_reward'].append(reward)
            this_is_a_sloppy_workaround = False
        else:
            data['average_reward'].append(data['average_reward'][-1] + 1/len(data['average_reward']) *
                                          (reward - data['average_reward'][-1]))
        if done:
            # episode is over, reset and continue
            if reward > 0:
                data['success'].append(1)
            else:
                data['success'].append(0)
            observation = env.reset()
            reward = 0
            new_episode = True
            data['episode_length'].append(i)
            break

print("Average Episode Length: {}".format(sum(data['episode_length'])/len(data['episode_length'])))
print("Success Rate: {}".format(sum(data['success'])/len(data['success'])))
print(estimates)

plt.plot(data['average_reward'])
plt.ylabel('average reward')
plt.xlabel('steps')
plt.show()
