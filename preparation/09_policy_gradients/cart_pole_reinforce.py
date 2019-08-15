import gym
import numpy as np
import ptan as ptan
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
import torch.nn.functional as F

"""
1. Initialize the network with random weights
2. Play N full episodes, saving their (s,a,r,s') transitions
3. For every step t of every episode k, calculate the discounted total reward for subsequent steps
4. Calculate the loss function for all transitions
5. Perform SGD update of weights minimizing the loss
6. Repeate from step 2 until convergence
"""

# hyperparams
GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4  # defines how many complete methods are used for training


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)  # net returns probabilities, important: output of the network is raw scores -> called logits


def calc_q_values(rewards) -> list:
    """
    :param rewards - accepts a list of rewards for the whole episode
    :returns calculates the discounted total reward for every step
    """
    res = []
    sum_r = 0.0

    #  calculate the reward from the end of the local reward list
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r  # discounted total reward for every step
        res.append(sum_r)
    return list(reversed(res))


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-reinforce")
    net = PGN(env.observation_space.shape[0], env.action_space.n)

    # needs to make decisions about actions for every observation. To selection the action to take, we need to obtain
    # the probabilities from the network and then perform random sampling from this probability distribution.
    # policy agent calls random.choice internally
    # apply softmax instructs the agent to convert the network output to probabilities
    # preprocessor is needed because cartpole env returns float64 but pytorch requires float32
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)

    #
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    done_episodes = 0
    batch_episodes = 0
    cur_rewards = []  # contains local rewards for the currently played episode

    # batch_states and batch_actions contain states and actions that we've seen from the last training
    batch_states, batch_actions, batch_q_values = [], [], []

    # training loop
    # every experience from the exp_source contains state, action, local reward and next_state
    # if end has been reached next_state is none
    for step_idx, exp in enumerate(exp_source):
        # for non-terminal experience entries, just save state, action and local_reward to list
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

        # when the episode ends we calculate the discounted total reward from local rewards
        if exp.last_state is None:
            batch_q_values.extend(calc_q_values(cur_rewards))  # and append them to batch_q_values list
            cur_rewards.clear()
            batch_episodes += 1  # increment episode counter

        # performed at the end of the episode, report current progress
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        # when enough episodes have passed since the last training step perform optimization
        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_q_values_v = torch.FloatTensor(batch_q_values)

        """ calculate the loss from the steps """

        # net calculates logits from states
        logits_v = net(states_v)

        # calculate logarithm + softmax with F.log_softmax
        log_prob_v = F.log_softmax(logits_v, dim=1)

        # select log probabilities from the actions taken and scale them with Q-values
        log_prob_actions_v = batch_q_values_v * log_prob_v[range(len(batch_states)), batch_actions_t]

        # average the scaled values and negate them to obtain the loss to minimize
        # negation is important because pg needs to be maximized to improve the policy
        # optimizer in pytorch does minimization in respect to the loss function we need to negate the pg
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward()
        optimizer.step()
        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_q_values.clear()

    writer.close()
