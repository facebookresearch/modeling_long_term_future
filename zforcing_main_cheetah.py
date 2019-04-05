import readline
import argparse
from itertools import count
import pickle 

#import free_mjc
import gym
import scipy.optimize
import numpy as np
import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from rl_zforcing_cheetah import ZForcing
import cv2
import random
import scipy.misc
import os

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

from pyvirtualdisplay import Display
display_ = Display(visible=0, size=(550, 500))
display_.start()

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env-name', default="HalfCheetah-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=5, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--aux-weight-start', type=float, default=0.,
                    help='start weight for auxiliary loss')
parser.add_argument('--aux-weight-end', type=float, default=0.,
                    help='end weight for auxiliary loss')
parser.add_argument('--bwd-weight', type=float, default=0.,
                    help='weight for bwd teacher forcing loss')
parser.add_argument('--bwd-l2-weight', type=float, default=1e-3,
                    help='weight for bwd l2 decoding loss')
parser.add_argument('--l2-weight', type=float, default=1.,
                    help='weight for fwd l2 decoding loss')
parser.add_argument('--fwd-ll-weight', type=float, default=1.,
                    help='weight for fwd likelihood loss')
parser.add_argument('--kld-weight-start', type=float, default=0.,
                    help='start weight for kl divergence between prior and posterior z loss')
parser.add_argument('--kld-step', type=float, default=1e-6,
                    help='step size to anneal kld_weight per iteration')
parser.add_argument('--aux-step', type=float, default=1e-6,
                    help='step size to anneal aux_weight per iteration')

parser.add_argument('--eval-interval', type=int, default=100, metavar='N',
                    help='evaluation interaval (default: 50)')

parser.add_argument('--val-batch-size', type=int, default=20, metavar='N',
                    help='random seed (default: 1)')

args = parser.parse_args()
lr = args.lr
env = gym.make(args.env_name)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)


filename = args.env_name + '_0.2_chunk/zforce_reacher_model_base_10k_' +  '_lr'+ str(args.lr) + '_fwd_nll_' + str(args.fwd_ll_weight) + '_fwd_l2w_' + str(args.l2_weight) + '_aux_w_' + str(args.aux_weight_start) + '_kld_w_' + str(args.kld_weight_start) + '_' + str(random.randint(1,500))
os.makedirs(filename, exist_ok=True)
train_folder = os.path.join(filename, 'train')
test_folder = os.path.join(filename, 'test')
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
zforce_filename = os.path.join(filename, 'student.pkl')
log_file = os.path.join(filename, 'log.txt')

train_on_image = True

def save_param(model, model_file_name):
    torch.save(model.state_dict(), model_file_name)

def load_param(model, model_file_name):
    model.load_state_dict(torch.load(model_file_name))
    return model

def load_samples(filename, idx=0 ,batch_size=10):
    images, actions = [], []
    filename = filename + str(idx) + '.pkl'
    with open(filename, 'rb') as f:
        for _ in range(1000):
            data = pickle.load(f)  
            images.append(data[0]) 
            actions.append(data[1])
        return (images, actions)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def write_samples(all_samples_obs, all_samples_actions, filename):
    # write to pickle file
    all_data = list(zip(all_samples_obs, all_samples_actions))
    output = open(filename, "wb")
    pickle.dump(all_data, output)
    output.close()
    return True

def print_norm(rnn):
    param_norm = []
    
    for param_tuple in zf.named_parameters():
        name = param_tuple[0]
        param = param_tuple[1]
        if 'bwd' in name:
            norm = param.grad.norm(2).data[0]/np.sqrt(np.prod(param.size()))
            param_norm.append(norm)
    return param_norm

def evaluate_(model):
    # evaluate how well model does 
    num_episodes = 0
    reward_batch = 0
    model.cuda()
    hidden = zf.init_hidden(1)
    all_action_diff = 0
    action = np.zeros(num_actions)
    # Each iteration first collect #batch_size episodes
    while num_episodes < args.val_batch_size:
        #print(num_episodes)
        state = env.reset()
        reward_sum = 0

        for t in range(1000):
            image = env.render(mode="rgb_array") 
            #image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC) 
            if num_episodes % 10 == 0:
                image_file =  os.path.join(filename, 'test/episode_'+ str(num_episodes) +  '_t_' + str(t)+'.jpg')
                scipy.misc.imsave(image_file, image)
            #image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)  
            action = torch.from_numpy(action).float().cuda()
            image = np.asarray(np.transpose(image, (2,0,1)), dtype=float)
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
            mask = torch.ones([1,1])
            with torch.no_grad():
                action_mu, action_var, hidden = zf.generate_onestep(image, mask, hidden, action = action.unsqueeze(0).unsqueeze(0)) 
            
                action_mu = action_mu.squeeze(0).squeeze(0)
                action_logvar = action_var.squeeze(0).squeeze(0)
                std = action_logvar.mul(0.5).exp_()
                eps = std.data.new(std.size()).normal_()
            
            action = eps.mul(std).add_(action_mu)
            
            action = action.cpu().data.numpy()

            expert_action = select_action(state).data.numpy()
            
            action_diff_norm =  (np.linalg.norm(expert_action - action))
            
            all_action_diff += action_diff_norm

            next_state, reward, done, _ = env.step(action)
            
            state = running_state(next_state)
            reward_sum += reward
            
            if done:
                break

        num_episodes += 1
        reward_batch += reward_sum

    print ('test reward is ', reward_batch/ num_episodes)
    print ('average action diff norm is ', all_action_diff / num_episodes / 50)
    log_line = 'test_reward is , ' + str(reward_batch/ num_episodes)
    with open(log_file, 'a') as f:
        f.write(log_line)
    return (reward_batch/num_episodes) 

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
running_state_test = ZFilter((num_inputs,), clip=5)

load_param(policy_net, "Cheetah_policy.pkl")
load_param(value_net, "Cheetah_value.pkl")

def pad(array, length):
    return array + [np.zeros_like(array[-1])] * (length - len(array))
def max_length(arrays):
    return max([len(array) for array in arrays])

zf = ZForcing(emb_dim=512, rnn_dim=512, z_dim=256,
              mlp_dim=256, out_dim=num_actions * 2, z_force=True, cond_ln=True)

bwd_param, fwd_param, hist_return_mean, hist_test_reward, hist_l2_loss = [], [], 0.0, -30.0, 5000.0 

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
bwd_l2_weight = args.bwd_l2_weight

zf.float().cuda()

def image_resize(image):
    image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    image = np.transpose(image, (2, 0, 1))
    return image


zf.train() 

num_samples, num_episodes, idx= 1000, 50, 0
data_file = 'data_cheetah/HalfCheetah-v2num_samples_10000_'


def prepare_data(train_images, train_actions, batch_size, i=0, chunk_len = 250):
    
    
    images_max_len = max_length(train_images)
    actions_max_len = max_length(train_actions)
    
    # divide the samples into k chunks
        
        
    start_idx, end_idx = i * chunk_len, min(actions_max_len, (i + 1) * chunk_len)
         
    images_mask = [[1] * (chunk_len)
                   for array in train_images]
        
    fwd_images = [pad(array[start_idx: end_idx], chunk_len) for array in train_images]
    bwd_images = [pad(array[start_idx + 1: end_idx + 1], chunk_len) for array in train_images]
        
    train_actions = [pad(array[start_idx:end_idx], chunk_len) for array in train_actions]
        
    fwd_images = np.array(list(zip(*fwd_images)), dtype=np.float32)
    bwd_images = np.array(list(zip(*bwd_images)), dtype=np.float32)
    images_mask = np.array(list(zip(*images_mask)), dtype=np.float32)
    train_actions = np.array(list(zip(*train_actions)), dtype=np.float32)

    x_fwd = torch.from_numpy(fwd_images).cuda()
    x_bwd = torch.from_numpy(bwd_images).cuda()
    y = torch.from_numpy(train_actions).cuda()
    x_mask = torch.from_numpy(images_mask).cuda()

    
    #images_mask = [[1] * (len(array) - 1) + [0] * (images_max_len - len(array))
    #              for array in train_images]

    #fwd_images = [pad(array[:-1], images_max_len - 1) for array in train_images]
    #bwd_images = [pad(array[1:], images_max_len - 1) for array in train_images]
    #training_actions = [pad(array, actions_max_len) for array in train_actions]
    #fwd_images = np.array(list(zip(*fwd_images)), dtype=np.float32)
    #bwd_images = np.array(list(zip(*bwd_images)), dtype=np.float32)
    #images_mask = np.array(list(zip(*images_mask)), dtype=np.float32)
    #train_actions = np.array(list(zip(*train_actions)), dtype=np.float32)

    #x_fwd = torch.from_numpy(fwd_images).cuda()
    #x_bwd = torch.from_numpy(bwd_images).cuda()
    #y = torch.from_numpy(train_actions).cuda()
    #x_mask = torch.from_numpy(images_mask).cuda()
 
    return (x_fwd, x_bwd, y, x_mask)

batch_size = args.batch_size

for episode in range(num_episodes):
    for idx in range(5):
        all_train_images, all_train_actions = load_samples(data_file, idx)
        for iteration in range(int(num_samples/ args.batch_size)):
                
            index = np.random.randint(num_samples - batch_size, size=1)[0]
            train_images = all_train_images[index : index + batch_size]
            train_actions = all_train_actions[index : index + batch_size]
            
            hidden = zf.init_hidden(args.batch_size)
            for k in range(4):
                # needs to re-init hidden if k > 0
                x_fwd, x_bwd, y, x_mask = prepare_data(train_images, train_actions, args.batch_size, k)

                fwd_opt.zero_grad()
                bwd_opt.zero_grad()
                fwd_nll, bwd_nll, aux_nll, kld, l2_loss, aux_fwd_l2, hidden = zf(x_fwd, x_bwd, y, x_mask, hidden)

                bwd_nll = (aux_weight > 0.) * (bwd_weight * bwd_nll)
                aux_nll = aux_weight * aux_nll
                fwd_nll = args.fwd_ll_weight * fwd_nll
                kld = kld_weight * kld
                l2_loss = args.bwd_l2_weight * l2_loss

                all_loss = fwd_nll + bwd_nll + aux_nll + kld + 1. * aux_fwd_l2 +  l2_loss
                fwd_loss = (fwd_nll + aux_nll + kld) + aux_fwd_l2
                bwd_loss = bwd_nll + l2_loss
            
                # anneal kld cost
                kld_weight += args.kld_step
                kld_weight = min(kld_weight, 1.)
                aux_loss = l2_loss

                if args.aux_weight_start < args.aux_weight_end:
                    aux_weight += args.aux_step
                    aux_weight = min(aux_weight, args.aux_weight_end)
                else:
                    aux_weight -= args.aux_step
                    aux_weight = max(aux_weight, args.aux_weight_end)
                fwd_loss.backward()
                bwd_loss.backward()

                torch.nn.utils.clip_grad_norm_(zf.parameters(), 100.)
                bwd_opt.step()
                fwd_opt.step()
            
            log_line ='Episode: %d, Index: %d, Iteration: %d, BN, fwd-ll-weight is %.2f, All loss is %.3f , foward loss is %.3f, fwd decoding loss is %.3f, backward loss is %.3f, aux loss is %.3f, kld is %.3f, l2 loss is %.3f' % (
                episode,
                idx, 
                iteration,
                args.fwd_ll_weight,
                all_loss.item(),
                fwd_nll.item(),
                aux_fwd_l2.item(),
                bwd_nll.item(),
                aux_nll.item(),
                kld.item(),
                l2_loss.item()
            ) + '\n'
            print(log_line)
            with open(log_file, 'a') as f:
                f.write(log_line)
            if np.isnan(all_loss.item()) or np.isinf(all_loss.item()):
                continue

            #print('norm is ', np.asarray(print_norm(zf)).mean())
            # backward propagation
    
        #if (iteration + 1 ) % args.eval_interval == 0:
        test_reward = evaluate_(zf)
        if (-test_reward) < (-hist_test_reward):
            hist_test_reward = test_reward
        if l2_loss <= hist_l2_loss:
            hist_l2_loss = l2_loss 
            save_param(zf, zforce_filename) 
