import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from models import ReplayBuffer
import yaml
import hydra
import wandb
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.integration.wandb import wandb_mixin
from utils import get_moving_average

class ActorCritic(nn.Module):
    def __init__(self, lr):
        super(ActorCritic, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        # print("lr:", config["lr"])
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def pi(self, x, softmax_dim = 0):
        # print('x1:', x)
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        # print('x2:', x)
        prob = F.softmax(x, dim=softmax_dim)
        # print("prob:", prob)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)

        return v
    """
    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float),\
            torch.tensor(done_lst)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
    """

    # def train_net(self, device):
    def train_net(self, config, memory, roll_outs, device):
        # s, a, r, s_prime, done = self.make_batch()
        s, a, r, s_prime, done = memory.sample(roll_outs)
        gamma = config["gamma"]
        td_target = r.to(device) + gamma*self.v(s_prime.to(device))*done.to(device)
        delta = td_target - self.v(s.to(device))

        pi = self.pi(s.to(device), softmax_dim=1)
        pi_a = pi.gather(1, a.to(device))
        loss = -torch.log(pi_a)*delta.detach() + F.smooth_l1_loss(self.v(s.to(device)), td_target.detach())
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


@wandb_mixin
def main(config):
    env = gym.make(config["env"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(config["lr"]).to(device)
    print_interval = 20
    score = 0.0
    memory = ReplayBuffer(config)
    render = False
    episode_durations = []
    moving_avrg_period = 100
    moving_average = 0.0
    prev_score = 0.0
    prev_ep = 1
    n_rollout = config["n_rollout"]

    for n_epi in range(config["max_episode"]):
        done = False
        s = env.reset()
        while not done:
            for t in range(n_rollout):
                # print("s: ", s)
                # print("done", done)
                # print("n_epi", n_epi)
                prob = model.pi(torch.from_numpy(s).float().to(device))
                # print("prob", prob)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, _ = env.step(a)
                done_mask = 0.0 if done else 1.0
                memory.put((s, a, r, s_prime, done_mask))
                # model.put_data((s, a, r, s_prime, done))

                s = s_prime
                score += r

                if render:
                    env.render()
                if done:
                    break
            if memory.size() > config["initial_exp"]:
                model.train_net(config, memory, n_rollout, device)
            # model.train_net(device)

        moving_average = get_moving_average(episode_durations, moving_avrg_period)  # break
        if n_epi < moving_avrg_period:
            episode_durations.append(score)
        else:
            episode_durations.pop(0)
            episode_durations.append(score)

        if score > prev_score:  # stores max score and episode when it happens
            prev_ep = n_epi
            prev_score = score

        wandb.log({"score": score, "max_avg_scr": prev_score, "maxscr_ep": prev_ep,
                   "moving_average": moving_average})  # logs episode score to wandb for comparizon
        tune.report(score=moving_average)  # optimize on which gets the best reward faster

        if n_epi % print_interval == 0 and n_epi != 0:  # Update q_target weights copied from q every c episodes
            str_updte = "n_ep : {}, scr : {:.1f}, max_scr : {:.1f}, max_scr_ep : {}".format(
                n_epi, score, prev_score, prev_ep)
            # print("n_episode : {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
            #                                     n_epi, score/print_interval, memory.size(), epsilon*100))
            print(str_updte)  # to show immediately the update

        score = 0.0
    env.close()


if __name__ == '__main__':

    cnfg_path = "./config.yml"
    cnfg = open(cnfg_path, 'r')
    config_dict = yaml.load(cnfg, Loader=yaml.FullLoader)

    # Hyper-parameters
    lr = config_dict['lr']
    # gamma = config_dict['gamma']
    max_episode = config_dict['max_episode']  # maybe same as buffer
    buffer_limit = config_dict['buffer_limit']  # buffer max size
    #batch_size = config_dict['batch_size']  # TBD
    initial_exp = config_dict['initial_exp']  # TBD
    n_rollout = config_dict["n_rollout"]

    # W&B run
    env = config_dict["env"]
    WB_API_KEY = config_dict['WB_API_KEY']
    project = config_dict['project']
    metric = config_dict['metric']
    mode = config_dict['mode']
    num_samples = config_dict['num_samples']


    analysis = tune.run(
        main,
        config={
            "env": env,
            "lr": lr,
            "n_rollout": tune.grid_search(n_rollout),
            "max_episode": max_episode,
            "initial_exp": tune.grid_search(initial_exp),
            "gamma": tune.grid_search(config_dict["gamma"]),
            "buffer_limit": tune.grid_search(buffer_limit),

            "wandb": {
                "project": project,
                "api_key": WB_API_KEY
            }
        },
        metric=metric,
        mode=mode,
        num_samples=num_samples)
