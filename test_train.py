import gymnasium as gym
import gymnasium_robotics
from env.AddImageObsWrapper import AddImageObsWrapper
from env.custom_fetch_and_place import MujocoFetchPickAndPlaceEnv
import numpy as np
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers.time_limit import TimeLimit
import torch
import cv2
import os

class ImageObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to convert the environment to use image observations for CNN policy
    """
    def __init__(self, env, image_size=(160, 120)):
        super().__init__(env)
        self.image_size = image_size
        
        # Update observation space to be image-based
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(3,image_size[0], image_size[1],), 
            dtype=np.uint8
        )
    
    def observation(self, obs):
        # Render the environment to get image
        img = self.env.render()
        if img is not None:
            # Resize image to desired size
            img_resized = cv2.resize(img, self.image_size,interpolation=cv2.INTER_AREA)
            # cv2.imwrite("test.jpg",img_resized)
            img_resized=img_resized.transpose(2,0,1)
            return img_resized
        else:
            # Return zeros if render fails
            return np.zeros((3,self.image_size[0], self.image_size[1],), dtype=np.uint8)

def create_env():
    """Create and wrap the environment"""
    # Create base environment
    env = MujocoFetchPickAndPlaceEnv(render_mode="rgb_array")
    env = AddImageObsWrapper(env)
    env = ImageObservationWrapper(env, image_size=(64, 64))
    env = Monitor(env)
    env =TimeLimit(env,max_episode_steps=2000)
    
    return env

def make_vec_env(n_envs=1):
    """Create vectorized environment"""
    def _init():
        return create_env()
    
    # Create vectorized environment
    vec_env = DummyVecEnv([_init for _ in range(n_envs)])
    # Normalize observations and rewards
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    return vec_env

def train_model():
    """Train the model using CNN policy"""
    
    # Create environment
    env = make_vec_env(n_envs=1)  # Use 4 parallel environments
    # eval_env = make_vec_env(n_envs=1)
    
    # Create model with CNN policy
    # SAC works well with continuous control tasks like robotic manipulation
    model = SAC(
        "CnnPolicy",
        env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=256,
        # tau=0.005,
        # gamma=0.98,
        # train_freq=1,
        # gradient_steps=1,
        ent_coef=0.1,
        target_update_interval=1,
        target_entropy='auto',
        use_sde=False,
        sde_sample_freq=-1,
        use_sde_at_warmup=False,
        
        tensorboard_log="./tensorboard_logs/",
        policy_kwargs=dict(
            net_arch=[256, 256],  # CNN feature extractor + MLP layers
            activation_fn=torch.nn.ReLU,
            normalize_images=False
        )
    )
    
    # Alternative: You can also try TD3 or PPO
    # model = TD3(
    #     "CnnPolicy",
    #     env,
    #     verbose=1,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    #     learning_rate=1e-3,
    #     buffer_size=1000000,
    #     learning_starts=10000,
    #     batch_size=100,
    #     tau=0.005,
    #     gamma=0.98,
    #     train_freq=(1, "episode"),
    #     gradient_steps=-1,
    #     policy_delay=2,
    #     target_policy_noise=0.2,
    #     target_noise_clip=0.5,
    #     tensorboard_log="./tensorboard_logs/",
    # )
    
    # # Set up callbacks
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
    # eval_callback = EvalCallback(
    #     eval_env, 
    #     best_model_save_path='./logs/',
    #     log_path='./logs/', 
    #     eval_freq=10000,
    #     deterministic=True, 
    #     render=False,
    #     callback_on_new_best=stop_callback
    # )
    
    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=1000000,
        # callback=eval_callback,
        log_interval=4,
        progress_bar=True
    )
    
    # Save the final model
    model.save("fetch_pick_place_cnn_final")
    
    # Save VecNormalize statistics
    env.save("vec_normalize.pkl")
    
    return model, env


if __name__ == "__main__":
    # Choose what to run
    model, env = train_model()
    print("Training completed!")
    