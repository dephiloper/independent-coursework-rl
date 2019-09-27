#!/usr/bin/env python3

from collections import namedtuple
from tensorboardX import SummaryWriter

import gym
import numpy as np
import torch
from torch import nn, optim

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 90
MAX_EPISODE_STEPS = 200

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


class Net(nn.Module):
    def __init__(self, observation_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            # straightforward way would be to include softmax after the last layer
            # --> would increase numerical stability
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    observation = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        observation_v = torch.FloatTensor([observation])
        act_probs_v = sm(net(observation_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=observation, action=action))

        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []

        observation = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))
    train_observations = []
    train_actions = []
    for example in batch:
        if example.reward >= reward_bound:
            train_observations.extend(map(lambda step: step.observation, example.steps))
            train_actions.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_observations)
    train_act_v = torch.LongTensor(train_actions)
    return train_obs_v, train_act_v, reward_bound, reward_mean


def record_vid(net: Net, it: int):
    env = gym.make("CartPole-v0")
    env._max_episode_steps = MAX_EPISODE_STEPS
    env = gym.wrappers.Monitor(env, directory="mon/" + str(it) + "/", force=True)
    observation = env.reset()
    sm = nn.Softmax(dim=1)

    for _ in range(MAX_EPISODE_STEPS):
        # env.render()
        obs_v = torch.FloatTensor([observation])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        # action = np.random.choice(len(act_probs), p=act_probs)
        action = np.argmax(act_probs)
        # env.step(action=action)
        observation, _, is_done, _ = env.step(action)

        if is_done:
            break
    env.close()


def evaluate(net: nn.Module):
    env = gym.make("CartPole-v0")
    env._max_episode_steps = MAX_EPISODE_STEPS

    observation = env.reset()
    sm = nn.Softmax(dim=1)

    for _ in range(MAX_EPISODE_STEPS):
        env.render()
        obs_v = torch.FloatTensor([observation])
        action_probabilities_v = sm(net(obs_v))
        action_probabilities = action_probabilities_v.data.numpy()[0]
        action = np.argmax(action_probabilities)
        env.step(action=action)
        observation, _, is_done, _ = env.step(action)

        if is_done:
            break
    env.close()


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env._max_episode_steps = MAX_EPISODE_STEPS
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter()
    best_mean_reward = 0

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        if best_mean_reward < reward_m:
            # record_vid(net, iter_no)  # for video recording
            best_mean_reward = reward_m

        if reward_m >= MAX_EPISODE_STEPS:
            print("Solved")
            break
    writer.close()
    env.close()

    evaluate(net)
