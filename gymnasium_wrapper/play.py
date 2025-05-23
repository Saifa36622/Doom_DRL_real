import os

from gymnasium.utils import EzPickle
import time
from vizdoom import scenarios_path
from base_gymnasium_env import VizdoomEnv
import wandb
import torch
import gc
from stable_baselines3 import PPO,SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

class VizdoomScenarioEnv(VizdoomEnv, EzPickle):
    """Basic ViZDoom environments which reside in the `scenarios` directory"""

    def __init__(
        self, scenario_file, frame_skip=1, max_buttons_pressed=1, render_mode=None
    ):
        EzPickle.__init__(
            self, scenario_file, frame_skip, max_buttons_pressed, render_mode
        )
        super().__init__(
            os.path.join(scenarios_path, scenario_file),
            frame_skip,
            max_buttons_pressed,
            render_mode,
        )


# Initialize Weights & Biases
# name = "Viz_Doom_PPO_4_entropy"
# wandb.init(project="doom_DRL_real", sync_tensorboard=True, name=name)

# Specify the scenario file
scenario_file = "deadly_corridor.cfg"  # Replace with your desired scenario file
scenario_path = os.path.join(scenarios_path, scenario_file)

# Create the environment
env = VizdoomScenarioEnv(scenario_file=scenario_file, render_mode="human")

# Clear GPU cache and collect garbage
torch.cuda.empty_cache()
gc.collect()


# Define the path to your trained model
model_path = "ViZDoom-master\gymnasium_wrapper\Viz_Doom_PPO_3__100000_steps"

# Load the trained PPO model
model = PPO.load(model_path, env=env)


terminated = False
truncated = False
obs,info = env.reset()
while not (terminated or truncated):
    action , _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    # time.sleep(0.5)

print("done")