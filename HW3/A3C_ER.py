import gym  # openai gym library
import torch
import torch.nn as nn  # Linear
import torch.nn.functional as F  # relu, softmax
import torch.optim as optim  # Adam Optimizer
from torch.distributions import Categorical  # Categorical import from torch.distributions module
import torch.multiprocessing as mp  # multi processing
import time
from matplotlib import pyplot as plt  ###for plot
import numpy as np

from models import ReplayBuffer2
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.integration.wandb import wandb_mixin
import wandb
from tqdm import trange
import sys, yaml

# Hyperparameters
n_train_processes = 3  # number of independent agents. (for training)
learning_rate = 0.0002
update_interval = 5  # Collect data during 'update_interval steps' and proceed with training.
gamma = 0.98
max_train_ep = 300  # max episode for training.
max_test_ep = 400  # max episode for test.
alg = 'A3C-ER'

# This class is equivalent to Actor-Critic. (pi, v)
class ActorCritic(nn.Module):  # ActorCritic Class - Created by inheriting nn.Module (provided by Pytorch) Class.
    def __init__(self, observations, actions):  # constructor - initializer(__init__): Object creation and variable initialization.
        super(ActorCritic, self).__init__()  # Calling the constructor of the inherited parent Class(nn.Module).
        self.fc1 = nn.Linear(observations, 256)  # Fully Connected: input   4 --> output 256
        self.fc_pi = nn.Linear(256, actions)  # Fully Connected: input 256 --> output   2
        self.fc_v = nn.Linear(256, 1)  # Fully Connected: input 256 --> output   1

    def pi(self, x, softmax_dim=0):  # In the case of batch processing, softmax_dim becomes 1. (default 0)
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        # x    = tensor([ [-0.1170, -0.1489],
        #                 [-0.0939, -0.1290],
        #                  ...,
        #                 [-0.1168, -0.1281] ], grad_fn=<AddmmBackward0>)
        prob = F.softmax(x,
                         dim=softmax_dim)  # x[5][2] ==> Perform the softmax in the direction "dim=1". ==> batch mode: batch_size(5) x 2 # (update_interval = 5)
        # x[2]    ==> Perform the softmax in the direction "dim=0".
        # prob = tensor([ [0.5080, 0.4920],
        #                 [0.5088, 0.4912],
        #                  ...,
        #                 [0.5028, 0.4972] ], grad_fn=<SoftmaxBackward0>)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


def train(global_model, rank, env_name):  # Called by 3(n_train_processes) agents independently.
    env = gym.make(env_name)
    obs = env.observation_space.shape[0]
    # actions = env.action_space.shape[0]
    actions = env.action_space.n
    local_model = ActorCritic(obs, actions)  # Call the ActorCritic.__init__() --> Creating an local_model Object
    local_model.load_state_dict(global_model.state_dict())  # Copy the global_model network weights & biases to local_model # local_model=global_model
    memory = ReplayBuffer2()

    optimizer = optim.Adam(global_model.parameters(),
                           lr=learning_rate)  # The optimizer updates the parameters of 'global_model'.

      # Create 'CartPole-v1' environment.

    for n_epi in range(max_train_ep):  # max_train_ep = 300
        done = False  # done becomes True when the episode ends.
        s = env.reset()  # Reset Environment - s = [0.02482228  0.00863265 -0.0270073  -0.01102263]

        while not done:  # CartPole-v1 forced to terminates at 500 steps.

            s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

            for t in range(update_interval):  # Collect data during 5(update_interval) steps and proceed with training.
                a = local_model.pi(torch.from_numpy(s).float())
                # torch.from_numpy(s).float() = tensor([-0.0411,  0.0121, -0.0213, -0.0273]) : Casting numpy array To Tensor float
                #prob =  tensor([0.4492, 0.5508], grad_fn=<SoftmaxBackward0>)

                m = Categorical(a)  # Categorical probability distribution model
                a = m.sample().item()  # tensor(0) or tensor(1)
                # a =  tensor(0)
                # a.item() = 0   - .item()-Casting a single tensor to a Python variable

                # a = a.detach()
                # a = a.numpy()
                s_prime, r, done, info = env.step(a)
                # s_prime = [-0.0408579  -0.18269246 -0.0218931   0.25857786]
                # r =  1.0
                # done = False (True==Terminal)
                done_mask = 0.0 if done else 1.0
                memory.put((s, a, r, s_prime, done_mask))

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r/100.0)
                s_prime_lst.append(s_prime)
                done_lst.append(done_mask)

                s = s_prime
                if done:
                    break
             # memory.put((s_lst, a_lst, r_lst, s_prime_lst, done_lst))
            s_lst, a_lst, r_lst, _, _ = memory.sample(update_interval)
            s_prime = s_lst[-1]

            s_final = torch.tensor(s_prime, dtype=torch.float)  # numpy array to tensor - s_final[4]

            R = 0.0 if done else local_model.v(s_final).item()  # .item() is to change tensor to python float type.


            td_target_lst = []
            for reward in r_lst[::-1]:  # r_lst[start,end,step(-1)] ==> 5(update_interval):0:-1
                R = gamma * R + reward
                td_target_lst.append([R])
            td_target_lst.reverse()

            s_batch, a_batch, td_target_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(
                a_lst), torch.tensor(td_target_lst)

            advantage = td_target_batch - local_model.v(s_batch)

            pi = local_model.pi(s_batch, softmax_dim=1)
            # pi = tensor([ [0.5080, 0.4920],
            #               [0.5088, 0.4912],
            #                ...,
            #               [0.5028, 0.4972] ], grad_fn=<SoftmaxBackward0>)
            pi_a = pi.gather(1, a_batch)

            # pi_a = torch.log(pi)
            # sum_pi_a = torch.unsqueeze(torch.sum(pi_a, 1), 1)
            # a_batch  = [       [0],      [1], ...,      [0] ]
            # pi_a     = [ [0.05080], [0.4912], ..., [0.5028] ]
            loss = -torch.log(pi_a) * advantage.detach() + F.smooth_l1_loss(local_model.v(s_batch),
                                                                            td_target_batch.detach())

            #loss = -torch.log(pi_a) * advantage.detach() + F.smooth_l1_loss(local_model.v(s_batch),
            #                                                                td_target_batch.detach())  # This is equivalent to Actor-Critic.
            # " .detach()" means do not update "advantage" and "td_target_batch".

            optimizer.zero_grad()  # In PyTorch, since the differential value is accumulated, the gradient must be initialized.
            loss.mean().backward()  # Backpropagation - gradient calculation

            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param.grad = local_param.grad  # local_param.grad -> A variable that stored the calculated gradient values.

            optimizer.step()  # weights & biases update
            local_model.load_state_dict(
                global_model.state_dict())  # Copy the global_model network weights & biases to local_model # local_model=global_model

    env.close()
    print("Training process {} reached maximum episode.".format(rank))


def print_reward(rwds, stds, eval_interval, label_name, color):  # for plot

    x = np.arange(len(rwds)) + 1
    x = x * eval_interval  # The test is run every eval_interval, and the mean and variance of the reward are stored.

    stds = np.array(stds)  # list to array
    rwds = np.array(rwds)

    y1 = rwds + stds  # for plot variance
    y2 = rwds - stds

    plt.plot(x, rwds, color=color, label=label_name)  # plot average reward
    plt.fill_between(x, y1, y2, color=color, alpha=0.1)  # plot variance

    plt.xlabel('Environment Episodes', fontsize=15)
    plt.ylabel('Average Reward', fontsize=15)

    plt.legend(loc='best', fontsize=15, frameon=False)  # frameon=False: no frame border
    # plt.legend(loc=(0.7,0.1), ncol=2, fontsize=15, frameon=False) #position 0,0 ~ 1,1 (x, y)

    plt.grid(color='w', linestyle='-', linewidth=2, alpha=0.3)  # grid

    plt.tick_params(axis='x', labelsize=10, color='w')
    plt.tick_params(axis='y', labelsize=10, color='w')

    ax = plt.gca()
    ax.set_facecolor('#EAEAF2')  # background color

    # Plot border invisible
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.show()


def test(global_model, run, env_name):
    env = gym.make(env_name)  # Create 'CartPole-v1' environment.
    score = 0.0
    print_interval = 20

    render = False  # for rendering

    reward_means = []  ###for plot
    reward_stds = []  ###for plot
    rwd_list = []  ###for plot

    for n_epi in range(max_test_ep):  # max_test_ep = 400
        done = False  # done becomes True when the episode ends.
        s = env.reset()  # Reset Environment

        rwd_sum = 0.0  # Sum of rewards for each episode

        while not done:
            prob = global_model.pi(torch.from_numpy(s).float())
            a = Categorical(prob).sample().item()

            # a = a.detach()
            # a = a.numpy()
            s_prime, r, done, info = env.step(a)
            s = s_prime

            score += r  # Sum of rewards for max_test_ep episode
            rwd_sum += r  # Sum of rewards for each episode

            if render:  # for rendering
                env.render()

        rwd_list.append(rwd_sum)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{:4d}, avg score : {:7.1f}, std : {:7.1f}".format(n_epi, score / print_interval,
                                                                                   np.std(rwd_list)))
            reward_means.append(score / print_interval)  ###for plot
            reward_stds.append(np.std(rwd_list))  ###for plot
            rwd_list = []

            if abs(score / print_interval) > 450:  # for rendering
                render = False
            score = 0.0
            time.sleep(1)  # 1 second.
    np.save("./rwd/" + alg + "/reward_" + env_name + "_run_" + str(run)+".npy", np.array(reward_means))

    env.close()

    print_reward(reward_means, reward_stds, print_interval, 'A3C',
                 'g')  # label name, color='r','g','b','c','m','y','k','w'




if __name__ == '__main__':
    runs = 10
    envs = ['CartPole-v1']
    # envs = ['Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4']
    for env_name in envs:
        crnt = gym.make(env_name)
        obs = crnt.observation_space.shape[0]
        # actions = crnt.action_space.shape[0]
        actions = crnt.action_space.n
        crnt.close()
        for run in range(runs):
            global_model = ActorCritic(obs, actions)
            global_model.share_memory()  # Move 'global_model' to shared memory to share data between multiple processes.

            processes = []
            # Create 3(n_train_processes) processes for training and 1 process for testing.
            for rank in range(n_train_processes + 1):  # + 1 for test process
                if rank == 0:
                    p = mp.Process(target=test, args=(global_model, run, env_name,))  # Create a test processor.
                else:
                    p = mp.Process(target=train, args=(global_model, rank, env_name,))  # Create training processor.

                p.start()  # Process Start
                processes.append(p)

            for p in processes:
                p.join()  # Wait for all processes to terminate.

