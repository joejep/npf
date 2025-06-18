import gymnasium as gym
import gymnasium_robotics

from env.AddImageObsWrapper import AddImageObsWrapper
from env.custom_fetch_and_place import MujocoFetchPickAndPlaceEnv
import cv2

env = MujocoFetchPickAndPlaceEnv(render_mode="rgb_array")
env=AddImageObsWrapper(env)
# Reset the environment
observation, info = env.reset()

# Run the environment loop
for step in range(10000):  # Added a limit to prevent infinite loop
    # Sample a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # img=env.render()
    cv2.imwrite("test_2.jpg",obs)
    if terminated or truncated:
        observation, info = env.reset()
        print(f"Episode finished at step {step}, resetting...")

env.close()