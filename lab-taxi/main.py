from agent import Agent
from monitor import interact
import gym
import numpy as np

# # Option 1: Use Taxi-v3 (recommended)
# env = gym.make('Taxi-v3')

# Option 2: Manually register Taxi-v2 (uncomment if you need Taxi-v2)
from gym.envs.toy_text.taxi import TaxiEnv
gym.envs.registration.register(
    id='Taxi-v2',
    entry_point='gym.envs.toy_text:TaxiEnv',
    max_episode_steps=200,
    reward_threshold=8.0,
)
env = gym.make('Taxi-v2')

agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)

# Print results
print("Average Rewards:", avg_rewards)
print("Best Average Reward:", best_avg_reward)