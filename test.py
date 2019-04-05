import argparse
from itertools import count

import cv2

import gym
import scipy.optimize
import pickle
import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from rl_zforcing import ZForcing
import scipy.misc


torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

from pyvirtualdisplay import Display
display_ = Display(visible=0, size=(600, 550))
display_.start()

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--write-data', type=bool, default=False, 
                    help='whether to write the data (state, action) pairs to file')

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--aux-weight-start', type=float, default=0.,
                    help='start weight for auxiliary loss')
parser.add_argument('--aux-weight-end', type=float, default=0.,
                    help='end weight for auxiliary loss')
parser.add_argument('--bwd-weight', type=float, default=0.,
                    help='weight for bwd teacher forcing loss')
parser.add_argument('--kld-weight-start', type=float, default=0.,
                    help='start weight for kl divergence between prior and posterior z loss')
parser.add_argument('--kld-step', type=float, default=5e-5,
                    help='step size to anneal kld_weight per iteration')
parser.add_argument('--aux-step', type=float, default=5e-5,
                    help='step size to anneal aux_weight per iteration')



args = parser.parse_args()

env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

def pad(array, length):
    return array + [np.zeros_like(array[-1])] * (length - len(array))
def max_length(arrays):
    return max([len(array) for array in arrays])


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

def write_data(data, file_name):
    with open(file_name, "wb") as fp:
        pickle.dump(l, fp)

def load_data(data, file_name):
    with open(file_name, "rb") as fp:   
        b = pickle.load(fp)

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)


reload = True
if reload:
    policy_net = load_param(policy_net, 'Reacher_policy.pkl')
    value_net = load_param(value_net, 'Reacher_value.pkl')


zf = ZForcing(emb_dim=512, rnn_dim=512, z_dim=256,
              mlp_dim=256, out_dim=num_actions, z_force=True, cond_ln=True)

opt = torch.optim.Adam(zf.parameters(), lr=args.lr, eps=1e-5)

kld_weight = args.kld_weight_start
aux_weight = args.aux_weight_start
bwd_weight = args.bwd_weight



import ipdb; ipdb.set_trace()

# data is the list of state_action pairs
data = []
for i_episode in count(1):
    
    training_images = []
    training_actions = [] 
    episode = []
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    
    while num_steps < args.batch_size:
        state = env.reset()
        state = running_state(state)
        episode_images = []
        episode_actions = []


        reward_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            image = env.render(mode="rgb_array")
            scipy.misc.imsave('Reacher_image/episode_'+ str(num_episodes) + '_batch_' + str(num_steps) + '_t_' + str(t)+'.jpg', image)
            image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            image = np.transpose(image, (2, 0, 1))

            reward_sum += reward
            next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0
            #memory.push(state, np.array([action]), mask, next_state, reward)
            if args.render:
                env.render()
            if done:
                break

            state = next_state
        
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    reward_batch /= num_episodes

    #if i_episode % args.log_interval == 0:
    print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
            i_episode, reward_sum, reward_batch))
