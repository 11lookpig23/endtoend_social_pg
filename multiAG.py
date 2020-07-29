
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from network import socialMask
from PGagent import PGagent,social_agent,newPG

class independentAgent():
    def __init__(self,agents):
        self.n_agents = len(agents)
        ## agents : list of agents
        self.agents = agents

    def select_actions(self,state):
        ## state: [obs1,obs2,...]
        actions = []
        for i,ag in zip(range(self.n_agents),self.agents):
            action = ag.select_action(state[i])
            actions.append(action)
        return actions
    
    def push_reward(self,reward):
        for i,ag in zip(range(self.n_agents),self.agents):
            ag.rewards.append(reward[i])
            
    def update_agents(self):
        for ag in self.agents:
            ag.update()
    
    def save(self,file_name):
        for i,ag in zip(range(self.n_agents),self.agents):
            torch.save(ag.policy,file_name+"pg"+str(i)+".pth") 


class socialAgents(independentAgent):
    def __init__(self,agents,agentParam):
        super().__init__(agents)
        self.Law = social_agent(agentParam)

    def select_mask_actions(self,state):
        actions = []
        for i,ag in zip(range(self.n_agents),self.agents):
            masks, prob_mask= self.Law.select_action(state[i])
            self.Law.prob_social.append(prob_mask)
            pron_mask_copy = prob_mask#deepcopy(prob_mask)
            action, prob_indi = ag.select_mask_action(state[i],pron_mask_copy)
            self.Law.pi_step.append(prob_indi)
            actions.append(action)
        return actions
    def push_reward(self,reward):
        for i,ag in zip(range(self.n_agents),self.agents):
            ag.rewards.append(reward[i])
        self.Law.rewards.append(sum(reward))
    def update_agents(self):
        for ag in self.agents:
            ag.update()
        self.Law.update(self.n_agents)
    def save(self,file_name):
        torch.save(self.Law.policy,file_name+"pg_law"+".pth")
        for i,ag in zip(range(self.n_agents),self.agents):
            torch.save(ag.policy,file_name+"pg"+str(i)+".pth")


class AC_Agents():
    def __init__(self,agents):
        self.num_agent = len(agents)
        self.agents = agents

    def select_actions(self,state):
        actions = []
        for agent, s in zip(self.agents, state):
            actions.append(int(agent.choose_action(s).detach().numpy()))
        return actions

    def update(self, state, reward, state_, action):
        for agent, s, r, s_,a in zip(self.agents, state, reward, state_, action):
            agent.update(s,r,s_,a)
'''
class Social_ACAgents():
    def __init__(self,agents,agentParam):
        self.Law = social_agent(agentParam)
        self.agents = agents
        self.n_agents = len(agents)

    def select_masked_actions(self, state):
        actions = []
        for i, ag in zip(range(self.n_agents), self.agents):
            masks, prob_mask = self.Law.select_action(state[i])
            self.Law.prob_social.append(prob_mask)  # prob_social is the list of masks for each agent
            pron_mask_copy = prob_mask  # deepcopy(prob_mask)
            action, prob_indi = ag.select_masked_action(state[i], pron_mask_copy)
            self.Law.pi_step.append(prob_indi)  # pi_step is the list of unmasked policy(prob ditribution) for each agent
            actions.append(action)
        return actions

    def update(self, state, reward, state_, action):
        for agent, s, r, s_,a in zip(self.agents, state, reward, state_, action):
            agent.update(s,r,s_,a)

    def update_law(self):
        self.Law.update(self.n_agents)

    def push_reward(self, reward):
        for i, ag in zip(range(self.n_agents), self.agents):
            ag.rewards.append(reward[i])
        self.Law.rewards.append(sum(reward))

'''

'''
..................
..................
..................
........O.........
.......OOO........
......OOOOO.......
.......OOO........
........O.........
..................
..................
..................
'''