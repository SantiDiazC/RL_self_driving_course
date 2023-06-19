import gym
import numpy as np

import gym_donkeycar

env = gym.make("donkey-warren-track-v0")
obs = env.reset()
try:
    for _ in range(100):
        # drive straight with small speed
        action = np.array([0.0, 0.5])
        # execute the action
        obs, reward, done, info = env.step(action)
        env.render()
except KeyboardInterrupt:
    # You can kill the program using ctrl+c
    pass

    # Exit the scene
env.close()
