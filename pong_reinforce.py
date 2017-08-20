from __future__ import print_function, division, absolute_import
from __builtin__ import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from itertools import count
from collections import namedtuple
import numpy as np
import gym


class Config(object):

    def __init__(self):
        self.in_dim = 80 * 80
        self.n_actions = 1
        self.hidden_size = 300
        self.discount_factor = 0.99
        self.learning_rate = 1e-4


class PolicyNet(nn.Module):

    def __init__(self, c):
        super(PolicyNet, self).__init__()
        self.config = c
        self.linear1 = nn.Linear(self.config.in_dim, self.config.hidden_size)
        # self.linear2 = nn.Linear(self.config.hidden_size, self.config.n_actions)
        self.linear2 = nn.Linear(self.config.hidden_size, 1)
        self.relu = nn.ReLU()
        # TODO: convert to a softmax version, calculate crossentropy, note the output dim above will different
        # self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return self.sigmoid(x)


# TODO: normalize the input images to see if this make sense
def process_img(img):
    img = img[35:195]
    img = img[::2, ::2, 0]
    img[img == 144] = 0
    img[img == 109] = 0
    img[img != 0] = 1

    return img.astype(np.float).ravel()


def choose_action(policy_net, state):
    probs = policy_net(Variable(torch.FloatTensor(state)))
    action = 0 if np.random.uniform() < probs.data[0] else 1
    # return action, probs.log() if action == 0 else (1 - probs).log()
    return action, probs.log()


if __name__ == '__main__':
    resume = False
    render = False
    np.random.seed(1024)

    env = gym.make('Pong-v0')
    c = Config()
    Step = namedtuple('Step', 'state action reward logprob')

    policy_net = PolicyNet(c)

    if resume:
        policy_net.load_state_dict(torch.load('./models/pytorch.pkl'))
        print("model loaded.")
    optimizer = optim.Adam(policy_net.parameters(), lr=c.learning_rate)

    for i_episode in count():
        episode = list()

        state = env.reset()
        state = process_img(state)
        final_reward = 0

        while True:
            if render:
                env.render()

            action, logprob = choose_action(policy_net, state)
            state, reward, done, _ = env.step(action + 2)
            state = process_img(state)
            episode.append(Step(state, action, reward, logprob))
            final_reward += reward

            if done:
                # Calculate the rewards
                # In the game of Pong, rewards are received mutliple times in one game.
                gamma = c.discount_factor
                running_reward = 0
                rewards = np.zeros(len(episode))
                loss = 0.0
                for i in range(len(episode) - 1, -1, -1):
                    if episode[i].reward != 0:
                        running_reward = 0
                    running_reward = running_reward * gamma + episode[i].reward
                    rewards[i] = running_reward

                # normalize the reward to reduce variance
                rewards -= np.mean(rewards)
                rewards /= np.std(rewards)
                advantage = np.zeros_like(rewards)
                loss = 0.0
                for i in range(len(episode)):
                    if episode[i].reward != 0:
                        # print(policy_net.linear2.weight.max())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss = 0.0
                    else:
                        loss += episode[i].logprob * rewards[i]
                break

        if i_episode and i_episode % 200 == 0:
            torch.save(policy_net.state_dict(), './models/pytorch.pkl')
            print('Model Saved.')
        print('Episode {}: final rewards:{}'.format(i_episode, final_reward))