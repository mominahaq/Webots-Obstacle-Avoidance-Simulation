import sys
import os

sys.path.append('/usr/local/webots/lib/controller/python')
sys.path.append(os.path.join(os.path.dirname(__file__), 'controllers', 'robot_controller'))

from robot_controller import RobotNavEnv
from stable_baselines3 import PPO

env = RobotNavEnv()

print("Starting training - Seed 2...")
model = PPO("MlpPolicy", env, tensorboard_log="./ppo_logs/", verbose=1, seed=2)

print("Training started...")
model.learn(total_timesteps=500_000, reset_num_timesteps=True, tb_log_name="run_seed2")
model.save("robot_nav_ppo_seed2")
print("Model saved!")
env.close()
