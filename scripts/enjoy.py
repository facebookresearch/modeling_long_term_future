#!/usr/bin/env python3

import argparse
import gym
import time

import babyai.utils as utils
from pyvirtualdisplay import Display

display_ = Display(visible=0, size=(550, 500))
display_.start()



# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin REQUIRED)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model REQUIRED)")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (default: 0 if model agent, 1 if demo agent)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--pause", type=float, default=0.1,
                    help="the pause between two consequent actions of an agent")

args = parser.parse_args()

assert args.model is not None or args.demos_origin is not None, "--model or --demos-origin must be specified."
if args.seed is None:
    args.seed = 0 if args.model is not None else 1

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)
for _ in range(args.shift):
    env.reset()

# Define agent

agent = utils.load_agent(args, env)

# Run the agent

done = True
import cv2
import numpy as np
episode = 0
step = 0
while True:
    time.sleep(args.pause)
    image = env.render("rgb_array")
    image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    #image = np.transpose(image, (2, 0, 1))
    file_name = 'rendered_image/episodes_'+str(episode) + '_step_' +str(step) + '.png'
    cv2.imwrite(file_name, image[:,:,::-1])
    step += 1
    if done:
        obs = env.reset()
        print("Mission: {}".format(obs["mission"]))
        episode += 1
        step = 0
    action = agent.get_action(obs)
    obs, reward, done, _ = env.step(action)
    agent.analyze_feedback(reward, done)

    if done:
        print("Reward:", reward)

    #if image.window is None:
    #    break
