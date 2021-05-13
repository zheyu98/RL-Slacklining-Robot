import argparse
import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='slackline',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/a2c',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic, obs_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"),
                        map_location='cpu')

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.obs_rms = obs_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()

if render_func is not None:
    render_func('human')

count = 1
ob = obs.clone().detach() 
t = 0
ob = np.append(ob,t)[np.newaxis, :]

while count:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)

    masks.fill_(0.0 if done else 1.0)

    if render_func is not None:
        render_func('human')
        time.sleep(0.03)

    t += 0.01
    cc = obs.clone().detach() 
    cc = np.append(cc, t)
    cc = cc[np.newaxis, :]
    ob = np.append(ob, cc, axis=0)
    if done:
        count = 0
print(np.mean(np.rad2deg(ob[:,2])))
print(np.mean(np.rad2deg(ob[:,3])))
print(np.std(np.rad2deg(ob[:,2])))
print(np.std(np.rad2deg(ob[:,3])))
plt.plot(ob[:,-1], np.rad2deg(ob[:,2]),'b',label='phib')
plt.plot(ob[:,-1], np.rad2deg(ob[:,3]),color='orange',label='phit')
plt.xlim(0,4)
plt.xlabel('Time in s')
plt.ylabel('Angles in deg')
plt.title('phib and phit with small initial deviation')
plt.legend(loc='lower left')
plt.grid()
plt.show()

   

