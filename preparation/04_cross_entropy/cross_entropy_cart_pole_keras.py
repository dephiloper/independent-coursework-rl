#!usr/bin/env python3
from collections import namedtuple

import gym
import numpy as np
from keras import Sequential, losses, optimizers
from keras.layers import Dense

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, model, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = np.array([env.reset()])

    while True:
        act_probs_v = model.predict(obs)
        act_probs = act_probs_v[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []

        obs = np.array([next_obs])


def one_hot(arr):
    oh_arr = []
    for val in arr:
        oh_arr.append([val == 0, val == 1])

    return oh_arr


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))
    train_obs = []
    train_act = []
    for example in batch:
        if example.reward >= reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = np.array(np.squeeze(train_obs))
    train_act_v = np.array(one_hot(train_act), dtype=np.dtype(np.int8))
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    observation_size = env.observation_space.shape[0]
    action_count = env.action_space.n

    model = Sequential()
    model.add(Dense(units=HIDDEN_SIZE, activation='relu', input_dim=4))
    model.add(Dense(units=action_count, activation='softmax', input_dim=HIDDEN_SIZE))

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(lr=0.01), metrics=['accuracy'])

    for iter_no, batch in enumerate(iterate_batches(env, model, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        result = model.train_on_batch(x=obs_v, y=acts_v)
        print("%d: reward_mean=%.1f, reward_bound=%.1f" % (iter_no, reward_m, reward_b))
        if reward_m >= 200:
            break

    obs = env.reset()
    for _ in range(1000):
        env.render()
        obs_v = np.array([obs])
        act_probs_v = model.predict(obs_v)
        act_probs = act_probs_v[0]
        # action = np.random.choice(len(act_probs), p=act_probs)
        action = np.argmax(act_probs)
        env.step(action=action)
        obs, _, is_done, _ = env.step(action)

        if is_done:
            break
    env.close()
