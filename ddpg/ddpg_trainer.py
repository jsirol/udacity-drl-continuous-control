import torch
import numpy as np
from collections import deque


def train_ddpg(agent, env, n_episodes=400, max_t=1000, save=True):
    """
    Trains a ddpg agent on the environment.
    :param agent: agent
    :param env: environment
    :param n_episodes: number of episodes
    :param max_t: max time steps per episode
    :param save: save model checkpoints and first weights when environment is considered solved
    :return: list of average scores over the agents per episode
    """
    # get the default brain
    brain_name = env.brain_names[0]
    scores_deque = deque(maxlen=100)
    final_scores = []
    not_solved = True
    num_agents = len(env.reset()[brain_name].vector_observations)
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset()[brain_name]
        states = env_info.vector_observations
        agent.reset()
        agent_scores = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            agent_scores += rewards
            if np.any(dones):
                break

        mean_score = np.mean(agent_scores)
        scores_deque.append(mean_score)
        final_scores.append(mean_score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), mean_score), end="", flush=True)
        if len(scores_deque) == 100 and np.mean(scores_deque) > 30 and not_solved:
            not_solved = False
            print("\nEnvironment solved in {} episodes!\n".format(i_episode), flush=True)
            if save:
                torch.save(agent.actor_local.state_dict(), 'models/actor_solved.pth')
                torch.save(agent.critic_local.state_dict(), 'models/critic_solved.pth')
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), flush=True)
            if save:
                torch.save(agent.actor_local.state_dict(), 'models/checkpoint_' + str(i_episode) + '_actor.pth')
                torch.save(agent.critic_local.state_dict(), 'models/checkpoint_' + str(i_episode) + '_critic.pth')
    return final_scores
