#!/usr/bin/env python3

import argparse
import cv2
import gym
import time
import datetime
import pickle
import babyai.utils as utils
from itertools import count
import scipy.optimize
from scripts.rl_zforcing_dec import ZForcing
import random
import scipy.misc
import torch
import numpy as np
# Parse arguments
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re
from pyvirtualdisplay import Display

display_ = Display(visible=0, size=(550, 500))
display_.start()

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--eval-episodes", type=int, default=1000,
                    help="number of episodes of evaluation (default: 1000)")
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument("--eval-interval", type=int, default=100,
                    help="how often to evaluate the student model")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (default: 0 if model agent, 1 if demo agent)")
parser.add_argument("--room", type=int, default=15,
                    help="room size")
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument('--aux-weight-start', type=float, default=0.,
                    help='start weight for auxiliary loss')
parser.add_argument('--l2-weight', type=float, default=1.,
                    help='weight for l2 loss')
parser.add_argument('--aux-weight-end', type=float, default=0.,
                    help='end weight for auxiliary loss')
parser.add_argument('--bwd-weight', type=float, default=0.,
                    help='weight for bwd teacher forcing loss')
parser.add_argument('--kld-weight-start', type=float, default=0.,
                    help='start weight for kl divergence between prior and posterior z loss')
parser.add_argument('--kld-step', type=float, default=1e-6,
                    help='step size to anneal kld_weight per iteration')
parser.add_argument('--aux-step', type=float, default=1e-6,
                    help='step size to anneal aux_weight per iteration')
parser.add_argument("--datafile", default=None,
                    help="name and location of the expert trajectory data file to load")

def pad(array, length):
    return array + [np.zeros_like(array[-1])] * (length - len(array))

def front_pad(array, length):
    return [np.zeros_like(array[-1])] * (length - len(array)) + array

def max_length(arrays):
    return max([len(array) for array in arrays])

def save_param(model, model_file_name):
    torch.save(model.state_dict(), model_file_name)

def load_param(model, model_file_name):
    model.load_state_dict(torch.load(model_file_name))
    return model

def write_samples(all_samples_obs, all_samples_actions, filename):
    # write to pickle file
    all_data = list(zip(all_samples_obs, all_samples_actions))
    output = open(filename, "wb")
    pickle.dump(all_data, output)
    output.close()
    return True

def load_samples(filename):
    output = open(filename, "rb")
    all_data = pickle.load(output)
    return all_data


def evaluate_student(agent, env, episodes):
    logs = {"num_frames_per_episode": [], "return_per_episode": []}
    reward_batch = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        num_frames = 0
        returnn = 0
        hidden = zf.init_hidden(1)

        while not (done):
            #action = agent(obs['image'])
            image = np.expand_dims(obs['image'], 0)
            mask = torch.ones(image.shape).unsqueeze(0)
            image = torch.from_numpy(image).unsqueeze(0).permute(0,1,4,2,3)
            action, hidden = zf.generate_onestep(image.float().cuda(), mask.cuda(), hidden)
            obs, reward, done, _ = env.step(action)
            num_frames += 1
            returnn += reward
        logs["num_frames_per_episode"].append(num_frames)
        logs["return_per_episode"].append(returnn)
        reward_batch.append(returnn)
    #log_line = 'test reward is '+ str(np.asarray(reward_batch).mean()) +'\n'
    #log_line = 'test reward std is ' + str(np.asarray(reward_batch).std()) + '\n'
    #print (log_line)
    #log_line = 'test reward is '+ str(np.asarray(reward_batch).mean())
    #with open(log_file, 'a') as f:
    #    f.write(log_line)
    return logs

def analysis_zf(agent, env, episode, iteration, episodes):
    logs = {"num_frames_per_episode": [], "return_per_episode": []}
    print ('analyzing model')
    reward_batch = []

    curr_dir = os.path.join(model_dir, 'episode_'+str(episode) + '_iter_' + str(iteration))
    os.mkdir(curr_dir)

    for iter_ in range(episodes):
        iter_dir = os.path.join(curr_dir, 'iter_'+str(iter_))
        os.mkdir(iter_dir)
        obs = env.reset()
        done = False
        num_frames = 0
        returnn = 0
        hidden = zf.init_hidden(1)
        test_images = []
        test_actions = []
        episode_images = []
        episode_actions = []
        images = []
        step = 0
        while not (done):
            #action = agent(obs['image'])
            image = env.render("rgb_array")
            image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
            file_name = os.path.join(iter_dir, 'iter_' +  str(iter_) +'_step_' +str(step) + '.png')
            cv2.imwrite(file_name, image[:,:,::-1])


            obs_image = np.expand_dims(obs['image'], 0)

            episode_images.append(obs_image)
            mask = torch.ones(obs_image.shape).unsqueeze(0)

            zf_image = torch.from_numpy(obs_image).unsqueeze(0).permute(0,1,4,2,3)
            action, hidden = zf.generate_onestep(zf_image.float().cuda(), mask.cuda(), hidden)
            episode_actions.append(action.item())
            obs, reward, done, _ = env.step(action)
            num_frames += 1
            step += 1
            returnn += reward
        # after gathering all observation images, run them through the ZForcing model and print predication cose
        obs_image = np.expand_dims(obs['image'], 0)
        episode_images.append(obs_image)

        test_images.append(episode_images)
        test_actions.append(episode_actions)

        images_max_len = max_length(test_images)
        actions_max_len = max_length(test_actions)
        images_mask = [[1] * (len(array) - 1) + [0] * (images_max_len - len(array))
                   for array in test_images]
        fwd_images = [pad(array[:-1], images_max_len - 1) for array in test_images]
        bwd_images = [front_pad(array[1:], images_max_len - 1) for array in test_images]
        bwd_images_target = [front_pad(array[:-1], images_max_len - 1) for array in test_images]
        training_actions = [pad(array, actions_max_len) for array in test_actions]

        fwd_images = np.array(list(zip(*fwd_images)), dtype=np.float32)
        bwd_images = np.array(list(zip(*bwd_images)), dtype=np.float32)
        bwd_images_target = np.array(list(zip(*bwd_images_target)), dtype=np.float32)
        images_mask = np.array(list(zip(*images_mask)), dtype=np.float32)
        test_actions = np.array(list(zip(*test_actions)), dtype=np.float32)
        x_fwd = torch.from_numpy(fwd_images.squeeze(1)).permute(0,1,4,2,3).cuda()
        x_bwd = torch.from_numpy(bwd_images.squeeze(1)).permute(0,1,4,2,3).cuda()
        y_bwd = torch.from_numpy(bwd_images_target.squeeze(1)).permute(0,1,4,2,3).cuda()
        #y_bwd = torch.from_numpy(fwd_images.squeeze(1)).permute(0,1,4,2,3).cuda()
        y = torch.from_numpy(test_actions).cuda()
        x_mask = torch.from_numpy(images_mask).cuda()

        fwd_nll, bwd_nll, aux_nlls, klds, log_pz, bwd_l2_loss = zf(x_fwd, x_bwd, y, y_bwd, x_mask, hidden, return_per_step=True)
        aux_nlls = aux_nlls.data.cpu().numpy().reshape(-1)
        plt.plot(aux_nlls, label='auxillary cost changes')
        plt.legend(loc='upper right')
        filename = os.path.join(iter_dir, 'aux_cost.pdf')
        plt.savefig(filename)
        plt.close()
        logs["num_frames_per_episode"].append(num_frames)
        logs["return_per_episode"].append(returnn)
        test_images = []
        test_actions = []
        reward_batch.append(returnn)
    return logs







if __name__ == "__main__":
    args = parser.parse_args()
    lr = args.lr

    if args.seed is None:
        args.seed = 0 # if args.model is not None else 1

    model_name = 'zforce_2opt_room_' + str(args.room) + '_lr'+ str(args.lr) + '_bwd_w_' + str(args.bwd_weight) +'_l2_w_' + str(args.l2_weight) + '_aux_w_' + str(args.aux_weight_start) + '_kld_w_' + str(args.kld_weight_start) + '_' + str(random.randint(1,1000))

    model_dir = os.path.join(args.env+'-model', model_name)

    os.mkdir(model_dir)
    zf_name = model_name + '.pkl'
    zf_file = os.path.join(model_dir, zf_name)

    log_name = model_name +'.log'
    log_file = os.path.join(model_dir, log_name)
    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Generate environment

    env = gym.make(args.env)
    env.seed(args.seed)


    # Run the agent

    start_time = time.time()

    # load expert data samples

    end_time = time.time()

    # Print logs

    '''num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames/(end_time - start_time)
    ellapsed_time = int(end_time - start_time)
    duration = datetime.timedelta(seconds=ellapsed_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    print("F {} | FPS {:.0f} | D {} | R:x̄σmM {:.2f} {:.2f} {:.2f} {:.2f} | F:x̄σmM {:.1f} {:.1f} {} {}"
          .format(num_frames, fps, duration,
                  *return_per_episode.values(),
                  *num_frames_per_episode.values()))

    indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
    n = 10
    print("{} worst episodes:".format(n))
    for i in indexes[:n]:
        print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
    '''
    # Train a student policy
    num_actions = 7
    zf = ZForcing(emb_dim=512, rnn_dim=512, z_dim=256,
            mlp_dim=256, out_dim=num_actions , z_force=False, cond_ln=False, use_l2=True)
    data_file = args.datafile #'data/BabyAI-UnlockPickup-v0start_flag_room_10_10000_samples.dat'
    all_data = load_samples(data_file)
    all_samples_obs, all_samples_actions = [list(t) for t in zip(*all_data)]

    fwd_param = []
    bwd_param = []

    hist_return_mean = 0.0

    for param_tuple in zf.named_parameters():
        name = param_tuple[0]
        param = param_tuple[1]
        if 'bwd' in name:
            bwd_param.append(param)
        else:
            fwd_param.append(param)

    zf_fwd_param = (n for n in fwd_param)
    zf_bwd_param = (n for n in bwd_param)
    fwd_opt = torch.optim.Adam(zf_fwd_param, lr = lr, eps=1e-5)
    bwd_opt = torch.optim.Adam(zf_bwd_param, lr = lr, eps=1e-5)

    kld_weight = args.kld_weight_start
    aux_weight = args.aux_weight_start
    bwd_weight = args.bwd_weight
    zf.float()
    zf.cuda()

    num_samples = len(all_samples_obs)

    batch_size = 32

    num_episodes = 50

    for episode in range(num_episodes):
        for i in range(int(num_samples/ batch_size)):
            training_images = all_samples_obs[i * batch_size : (i + 1) * batch_size]
            training_actions = all_samples_actions[i * batch_size : (i + 1) * batch_size]
            images_max_len = max_length(training_images)
            actions_max_len = max_length(training_actions)
            images_mask = [[1] * (len(array) - 1) + [0] * (images_max_len - len(array))
                   for array in training_images]

            fwd_images = [pad(array[:-1], images_max_len - 1) for array in training_images]


            bwd_images = [front_pad(array[1:], images_max_len - 1) for array in training_images]
            bwd_images_target = [front_pad(array[:-1], images_max_len - 1) for array in training_images]
            training_actions = [pad(array, actions_max_len) for array in training_actions]

            fwd_images = np.array(list(zip(*fwd_images)), dtype=np.float32)
            bwd_images = np.array(list(zip(*bwd_images)), dtype=np.float32)
            bwd_images_target = np.array(list(zip(*bwd_images_target)), dtype=np.float32)
            images_mask = np.array(list(zip(*images_mask)), dtype=np.float32)
            training_actions = np.array(list(zip(*training_actions)), dtype=np.float32)
            x_fwd = torch.from_numpy(fwd_images).permute(0,1,4,2,3).cuda()
            x_bwd = torch.from_numpy(bwd_images).permute(0,1,4,2,3).cuda()
            y_bwd = torch.from_numpy(bwd_images_target).permute(0,1,4,2,3).cuda()
            y = torch.from_numpy(training_actions).cuda()
            x_mask = torch.from_numpy(images_mask).cuda()

            zf.float().cuda()
            hidden = zf.init_hidden(batch_size)

            fwd_opt.zero_grad()
            bwd_opt.zero_grad()

            fwd_nll, bwd_nll, aux_nll, kld, bwd_l2_loss = zf(x_fwd, x_bwd, y, y_bwd, x_mask, hidden)
            #bwd_nll = (aux_weight > 0.) * (bwd_weight * bwd_nll)
            bwd_nll = bwd_weight * bwd_nll
            aux_nll = aux_weight * aux_nll
            all_loss = fwd_nll + bwd_nll + aux_nll + kld_weight * kld + args.l2_weight * bwd_l2_loss
            fwd_loss = fwd_nll + aux_nll + kld_weight * kld
            bwd_loss = args.l2_weight * bwd_l2_loss + 0.0 * bwd_nll

            kld_weight += args.kld_step
            kld_weight = min(kld_weight, 1.)
            if args.aux_weight_start < args.aux_weight_end:
                aux_weight += args.aux_step
                aux_weight = min(aux_weight, args.aux_weight_end)
            else:
                aux_weight -= args.aux_step
                aux_weight = max(aux_weight, args.aux_weight_end)
            log_line ='Episode: %d, Iteration: %d, All loss is %.3f , forward loss is %.3f, backward loss is %.3f, l2 loss is %.3f, aux loss is %.3f, kld is %.3f' % (
                episode, i,
                all_loss.item(),
                fwd_nll.item(),
                bwd_nll.item(),
                bwd_l2_loss.item(),
                aux_nll.item(),
                kld.item()
            ) + '\n'
            #print(log_line)
            with open(log_file, 'a') as f:
                f.write(log_line)

            fwd_loss.backward()
            bwd_loss.backward()

            torch.nn.utils.clip_grad_norm_(zf.parameters(), 100.)
            #opt.step()
            fwd_opt.step()
            bwd_opt.step()

            if (i + 1) % (args.eval_interval) == 0:
                logs = evaluate_student(zf, env, args.eval_episodes)
                # Print logs
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
                with open(log_file, 'a') as f:
                    f.write(log_line)
                if return_per_episode['mean'] > hist_return_mean:
                    save_param(zf, zf_file)
                    hist_return_mean = return_per_episode['mean']
            if (i + 1 ) % 200 == 0:
                analysis_zf(zf, env, episode, i, 10)





