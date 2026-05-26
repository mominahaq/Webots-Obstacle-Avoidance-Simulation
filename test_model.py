import sys
import os

sys.path.append('/usr/local/webots/lib/controller/python')
sys.path.append(os.path.join(os.path.dirname(__file__), 'controllers', 'robot_controller'))

from robot_controller import RobotNavEnv
from stable_baselines3 import PPO

env = RobotNavEnv()

print("Loading model...")
model = PPO.load("robot_nav_ppo_fixed_500k", env=env)

print("Testing model — watch Webots!")
episodes = 10
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"Episode {ep+1}: Steps={steps}, Total Reward={total_reward:.1f}, {'✅ GOAL' if terminated else '❌ TIMEOUT'}")

env.close()
print("Done!")
