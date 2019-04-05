import argparse
from itertools import count

#import free_mjc
import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from rl_zforcing import ZForcing
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
parser.add_argument('--env-name', default="Reacher-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--aux-weight-start', type=float, default=0.,
                    help='start weight for auxiliary loss')
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

parser.add_argument('--eval-interval', type=int, default=50, metavar='N',
                    help='evaluation interaval (default: 50)')

parser.add_argument('--val-batch-size', type=int, default=50, metavar='N',
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


filename = args.env_name + '_model/zforce_reacher_lr'+ str(args.lr) + '_aux_w_' + str(args.aux_weight_start) + '_kld_w_' + str(args.kld_weight_start) + '_' + str(random.randint(1,500))
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

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def evaluate_(model):
    # evaluate how well model does 
    num_episodes = 0
    reward_batch = 0
    model.cuda()
    hidden = zf.init_hidden(1)
    all_action_diff = 0

    # Each iteration first collect #batch_size episodes
    while num_episodes < args.val_batch_size:
        #print(num_episodes)
        state = env.reset()
        reward_sum = 0
        for t in range(10000):
            image = env.render(mode="rgb_array") 
            if num_episodes % 5 == 0:
                image_file =  os.path.join(filename, 'test/episode_'+ str(num_episodes) +  '_t_' + str(t)+'.jpg')
                scipy.misc.imsave(image_file, image)

            image = image_resize(image)
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
            mask = torch.ones([1,1])
            action_mu, action_var, hidden = zf.generate_onestep(image, mask, hidden) 
            
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

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
running_state_test = ZFilter((num_inputs,), clip=5)

load_param(policy_net, "Reacher_policy.pkl")
load_param(value_net, "Reacher_value.pkl")

policy_net#.cuda()
value_net#.cuda()

def pad(array, length):
    return array + [np.zeros_like(array[-1])] * (length - len(array))
def max_length(arrays):
    return max([len(array) for array in arrays])

zf = ZForcing(emb_dim=512, rnn_dim=512, z_dim=256,
              mlp_dim=256, out_dim=num_actions * 2, z_force=True, cond_ln=True)

opt = torch.optim.Adam(zf.parameters(), lr=lr, eps=1e-5)

kld_weight = args.kld_weight_start
aux_weight = args.aux_weight_start
bwd_weight = args.bwd_weight

#import ipdb.set_trace()
#zf = load_param(zf, 'zforce_reacher_64.pkl')
zf.float()
zf.cuda()

#evaluate_(zf)
#import ipdb; ipdb.set_trace()

def image_resize(image):
    image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    image = np.transpose(image, (2, 0, 1))
    return image


def expert_sample():
    import ipdb; ipdb.set_trace()


for iteration in count(1):
    training_images = []
    training_actions = []
    
    num_episodes = 0
    reward_batch = 0
    zf.train() 
    # Each iteration first collect #batch_size episodes
    while num_episodes < args.batch_size:
        #print(num_episodes)
        episode_images = []
        episode_actions = []
        state = env.reset() 
        state = running_state_test(state)
        reward_sum = 0
        
        for t in range(10000):
            
            action = select_action(state)
            action = action.data[0].numpy()
            
            if train_on_image:
                image = env.render(mode="rgb_array")
                image_filename = os.path.join(filename, 'train/episode_'+ str(num_episodes) + '_t_' + str(t)+'.jpg')
                scipy.misc.imsave(image_filename, image)
                image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
                image = np.transpose(image, (2, 0, 1))
            

            next_state, reward, done, _ = env.step(action)
            
            reward_sum += reward
            next_state = running_state_test(next_state)
            
            episode_images.append(image)
            episode_actions.append(action)
            
            state = next_state
            if done:
                break
        
        num_episodes += 1
        reward_batch += reward_sum
        
        image = env.render(mode="rgb_array")
        image = image_resize(image)

        episode_images.append(image)
        
        training_images.append(episode_images)
        training_actions.append(episode_actions)
        
    print (reward_batch/ num_episodes)
    
    # After having #batch_size trajectories, make the python array into numpy array
    images_max_len = max_length(training_images)
    actions_max_len = max_length(training_actions)
    images_mask = [[1] * (len(array) - 1) + [0] * (images_max_len - len(array))
                   for array in training_images]
    
    # Here's something a little twisted, we want the trajectories in one batch to be the same
    # length. So we want to pad zero to the ends of short trajectories. However, the forward
    # and backward trajectories are shifted by one. So we need to create and pad the fwd/bwd
    # trajectories individually and pass them to the zforcing model.
    fwd_images = [pad(array[:-1], images_max_len - 1) for array in training_images]
    bwd_images = [pad(array[1:], images_max_len - 1) for array in training_images]
    training_actions = [pad(array, actions_max_len) for array in training_actions]
    fwd_images = np.array(list(zip(*fwd_images)), dtype=np.float32)
    bwd_images = np.array(list(zip(*bwd_images)), dtype=np.float32)
    images_mask = np.array(list(zip(*images_mask)), dtype=np.float32)
    training_actions = np.array(list(zip(*training_actions)), dtype=np.float32)
    
    x_fwd = torch.from_numpy(fwd_images).cuda()
    x_bwd = torch.from_numpy(bwd_images).cuda()
    y = torch.from_numpy(training_actions).cuda()
    x_mask = torch.from_numpy(images_mask).cuda()
    
    zf.float().cuda()
    hidden = zf.init_hidden(args.batch_size)


    opt.zero_grad()
    fwd_nll, bwd_nll, aux_nll, kld = zf(x_fwd, x_bwd, y, x_mask, hidden)
    bwd_nll = (aux_weight > 0.) * (bwd_weight * bwd_nll)
    aux_nll = aux_weight * aux_nll
    all_loss = fwd_nll + bwd_nll + aux_nll + kld_weight * kld

    # anneal kld cost
    kld_weight += args.kld_step
    kld_weight = min(kld_weight, 1.)
    if args.aux_weight_start < args.aux_weight_end:
        aux_weight += args.aux_step
        aux_weight = min(aux_weight, args.aux_weight_end)
    else:
        aux_weight -= args.aux_step
        aux_weight = max(aux_weight, args.aux_weight_end)
    log_line ='Iteration: %d, All loss is %.3f , foward loss is %.3f, backward loss is %.3f, aux loss is %.3f, kld is %.3f' % (
            iteration,
            all_loss.item(),
            fwd_nll.item(),
            bwd_nll.item(),
            aux_nll.item(),
            kld.item()
        ) + '\n'
    print(log_line)
    with open(log_file, 'a') as f:
        f.write(log_line)
    if np.isnan(all_loss.item()) or np.isinf(all_loss.item()):
        continue

    # backward propagation
    all_loss.backward()
    torch.nn.utils.clip_grad_norm_(zf.parameters(), 100.)

    opt.step()
    if (iteration + 1 ) % args.eval_interval == 0:
        save_param(zf, zforce_filename) 
        evaluate_(zf)


