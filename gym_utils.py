import gym
import numpy as np

def query_environment(name):

  env = gym.make(name)
  spec = gym.spec(name)
  env_info = {"Name" : name,
              "Action Space" : env.action_space,
              "Observation Space" : env.observation_space,
              "Max Episode Steps" : spec.max_episode_steps,
              "Nondeterministic"  : spec.nondeterministic,
              "Reward Range"      : env.reward_range,
              "Reward Threshold"  : spec.reward_threshold}
  print(f"Action Space: {env.action_space}")
  print(f"Observation Space: {env.observation_space}")
  print(f"Max Episode Steps: {spec.max_episode_steps}")
  print(f"Nondeterministic: {spec.nondeterministic}")
  print(f"Reward Range: {env.reward_range}")
  print(f"Reward Threshold: {spec.reward_threshold}")

  return env, env_info




