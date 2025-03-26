import gymnasium as gym
from ddpg import Actor, QNetwork, make_env
import torch
import numpy as np

env_id = 'InvertedPendulum-v4'
env = gym.make(env_id, render_mode="human")
device = torch.device("cpu")
model_path = "runs//InvertedPendulum-v4__rs-ddpg__1__1743012371/rs-ddpg.cleanrl_model"
capture_video = True


envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, "eval")])
actor = Actor(envs).to(device)
qf = QNetwork(envs).to(device)
actor_params, qf_params = torch.load(model_path, map_location=device)
actor.load_state_dict(actor_params)
actor.eval()
qf.load_state_dict(qf_params)
qf.eval()

observation, info = env.reset()

episode_over = False
i = 0
while not episode_over and i < 100:
    i += 1
    print(i)
    # action = env.action_space.sample()  # agent policy that uses the observation and info
    with torch.no_grad():
        action = actor(torch.Tensor(observation).to(device))
        action = action.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)
        action = action.squeeze()

    if action.shape == ():
        action = np.array([action])

    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

    env.render()

env.close()