import os

from gymnasium.utils import EzPickle

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
name = "Viz_Doom_PPO_8_dont_know"
wandb.init(project="doom_DRL_real", sync_tensorboard=True, name=name)

# Specify the scenario file
scenario_file = "deadly_corridor.cfg"  # Replace with your desired scenario file
scenario_path = os.path.join(scenarios_path, scenario_file)

# Create the environment
env = VizdoomScenarioEnv(scenario_file=scenario_file, render_mode="human")

# Clear GPU cache and collect garbage
torch.cuda.empty_cache()
gc.collect()

learning_rate=3e-4
n_steps=500
batch_size=500
n_epochs=10
gamma=0.99
gae_lambda=0.95
clip_range=0.2
clip_range_vf=None
normalize_advantage=True
ent_coef=0.01
vf_coef=0.5
max_grad_norm=0.5
use_sde=False
sde_sample_freq=-1
rollout_buffer_class=None
rollout_buffer_kwargs=None
target_kl=0.01
stats_window_size=100


# Define and initialize the PPO model
model = PPO(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=learning_rate,
    n_steps=n_steps,
    batch_size=batch_size,
    n_epochs=n_epochs,
    gamma=gamma,
    gae_lambda=gae_lambda,
    clip_range=clip_range,
    clip_range_vf=None,
    normalize_advantage=normalize_advantage,
    ent_coef=ent_coef,
    vf_coef=vf_coef,
    
    max_grad_norm=max_grad_norm,
    sde_sample_freq=sde_sample_freq,
    rollout_buffer_class=None,
    rollout_buffer_kwargs=None,
    target_kl=None,
    stats_window_size=100,
    use_sde=False,  # Disable gSDE
    tensorboard_log="./tensorboard_logs/",
    policy_kwargs=None,
    verbose=0,
    seed=42,
    device="auto",
    _init_setup_model=True,
)

# model = SAC(
#     policy="MultiInputPolicy",
#     env= env,
#     learning_rate =3.0e-4 ,
#     buffer_size = 500,
#     batch_size=500,
#     ent_coef='auto',
#     gamma=0.99,
#     gradient_steps= 1 ,
#     action_noise= None,
#     learning_starts = 500,
#     tau =  0.005,
#     device='auto',
#     optimize_memory_usage = False,
#     # ent_coef = 'auto',
#     tensorboard_log ="./tensorboard_logs/",
# )

checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path=f"./model_save/{name}",
    name_prefix=f"{name}_"
)


callback = CallbackList([checkpoint_callback])
# Train the model
model.learn(
    total_timesteps=100000,
    tb_log_name=name,
    # log_interval=10,
    callback=callback,
)

# Save the trained model
model.save(f"Doom_model_{name}_done")
wandb.finish()
print("done")