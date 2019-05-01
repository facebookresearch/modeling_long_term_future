# Copyright (c) Facebook, Inc. and its affiliates.
#!/usr/bin/env python3

import argparse
import gym
import time

import babyai.utils as utils
from pyvirtualdisplay import Display
import scipy.optimize
import random
import scipy.misc
import torch
from scripts.rl_zforcing import ZForcing
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


display_ = Display(visible=0, size=(550, 500))
display_.start()



# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model REQUIRED)")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (default: 0 if model agent, 1 if demo agent)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin REQUIRED)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--pause", type=float, default=0.1,
                    help="the pause between two consequent actions of an agent")

args = parser.parse_args()

def load_param(model, model_file_name):
    model.load_state_dict(torch.load(model_file_name))
    return model

if args.seed is None:
    args.seed = 1

# Set seed for all randomness sources

utils.seed(args.seed)
num_actions = 7

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)
for _ in range(args.shift):
    env.reset()

# Define agent
random.seed(20)

agent = utils.load_agent(args, env)

zf_model = 'BabyAI-UnlockPickup-v0_model/zforce_2opt_room_10_lr0.0001_bwd_w_1.0_aux_w_1e-06_kld_w_0.0_491.pkl'
#zf_model = 'BabyAI-UnlockPickup-v0_model/zforce_2opt_room_10_lr0.0001_bwd_w_0.0_aux_w_0.0_kld_w_0.0_976.pkl'
#model_name = 'zforce_2opt_room_10_lr0.0001_bwd_w_0.0_aux_w_0.0_kld_w_0.0_976' + str(random.randint(1,5000))
model_name = 'zforce_2opt_room_10_lr0.0001_bwd_w_1.0_aux_w_1e-06_kld_w_0.0_491_' + str(random.randint(1,5111))
image_dir = 'zf_rendered_image'

import ipdb; ipdb.set_trace()

model_image_dir = os.path.join(image_dir, model_name)
os.mkdir(model_image_dir)

zf = ZForcing(emb_dim=512, rnn_dim=512, z_dim=256,
            mlp_dim=256, out_dim=num_actions , z_force=True, cond_ln=True, return_loss=True)
zf = load_param(zf, zf_model)
# Run the agent

done = True
import cv2
import numpy as np
episode = 0
step = 0
logs = {"num_frames_per_episode": [], "return_per_episode": []}
returnn = 0

zf.float().cuda()
hidden = zf.init_hidden(1)
obs = env.reset()
num_frames = 0
step = 0
model_image_dir_episode = os.path.join(model_image_dir, 'episode_0')
os.mkdir(model_image_dir_episode)

num_episode = 200

aux_loss = []


def plot_loss(aux_loss, image_dir):
    plt.plot(aux_loss)
    plt_file = os.path.join(image_dir, 'aux_loss.pdf')
    plt.savefig(plt_file)
start_time = time.time()
while True:
    time.sleep(args.pause)
    image = env.render("rgb_array")
    image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    epi_file_name = 'episodes_' +  str(episode) +'_step_' +str(step) + '.png'
    file_name = os.path.join(model_image_dir_episode, epi_file_name) 
    #file_name = 'zf_rendered_image/episodes_'+str(episode) + '_step_' +str(step) + '.png'
    #image = np.transpose(image, (2, 0, 1))
    cv2.imwrite(file_name, image[:,:,::-1])
    
    obs_image = np.expand_dims(obs['image'], 0)
    mask = torch.ones(obs_image.shape).unsqueeze(0)
    obs_image = torch.from_numpy(obs_image).unsqueeze(0).permute(0,1,4,2,3)
    action, hidden, aux_nll = zf.generate_onestep(obs_image.float().cuda(), mask.cuda(), hidden) 
    aux_loss.append(aux_nll) 
    if done:
        obs = env.reset()
        plot_loss(aux_loss, model_image_dir_episode)
        print("Mission: {}".format(obs["mission"]))
        episode += 1
        obs = env.reset()
        done = False
        num_frames = 0
        returnn = 0
        step = 0
        hidden = zf.init_hidden(1)
        model_image_dir_episode = os.path.join(model_image_dir, 'episode_'+str(episode))
        os.mkdir(model_image_dir_episode)
        aux_loss = []
    action = agent.get_action(obs)
    obs, reward, done, _ = env.step(action)
    agent.analyze_feedback(reward, done)
    num_frames += 1
    step += 1
    returnn += reward
    if done:
        print("Reward:", reward)
        logs["num_frames_per_episode"].append(num_frames)
        logs["return_per_episode"].append(returnn)
    
    if episode > num_episode:
        break
    #if image.window is None:
    #    break
import datetime
end_time = time.time()
num_frames = sum(logs["num_frames_per_episode"])                                                                                                                                                     
fps = num_frames/(end_time - start_time)                                                                                                                                                             
ellapsed_time = int(end_time - start_time)                                                                                                                                                           
duration = datetime.timedelta(seconds=ellapsed_time)                                                                                                                                                 
return_per_episode = utils.synthesize(logs["return_per_episode"])                                                                                                                                    
num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])                                                                                                                            
                                                                                                                                                                                                                     
log_line = ("F {} | FPS {:.0f} | D {} | R:x̄σmM {:.2f} {:.2f} {:.2f} {:.2f} | F:x̄σmM {:.1f} {:.1f} {} {}".format(num_frames, fps, duration, *return_per_episode.values(),*num_frames_per_episode.values()))
print("F {} | FPS {:.0f} | D {} | R:x̄σmM {:.2f} {:.2f} {:.2f} {:.2f} | F:x̄σmM {:.1f} {:.1f} {} {}"                                                                                                   
                    .format(num_frames, fps, duration,                                                                                                                                                               
                    *return_per_episode.values(),                                                                                                                                                                    
                    *num_frames_per_episode.values()))     
#print(log_line)
