import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from network import Policy,socialMask,Actor,Critic


class PGagent():
    def __init__(self,state_dim,action_dim,agentParam):
        self.state_dim = state_dim#400#env.observation_space.shape[0]
        self.action_dim = action_dim# 8#env.action_space.n
        self.gamma = agentParam["gamma"]
        # init N Monte Carlo transitions in one game
        self.saved_log_probs = []
        self.use_cuda = torch.cuda.is_available()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.rewards = []
        self.device = agentParam["device"]
        # init network parameters
        if agentParam["ifload"]:
            self.policy = torch.load(agentParam["filename"]+"pg"+agentParam["id"]+".pth",map_location = torch.device('cuda'))
        else:
            self.policy = Policy(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
            
        self.optimizer = optim.Adam(self.policy.parameters(), lr=agentParam["LR"])
        self.eps = np.finfo(np.float32).eps.item()

        # init some parameters
        self.time_step = 0


    def select_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state.to(self.device))
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action).to(self.device))
        return action.item()


    def update(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device).type(self.FloatTensor)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]


class social_agent(PGagent):
    def __init__(self,state_dim,action_dim,agentParam):
        super().__init__(state_dim,action_dim,agentParam)
        if agentParam["ifload"]:
            self.policy = torch.load(agentParam["filename"]+"pg_law"+".pth",map_location = torch.device('cuda'))
        else:
            self.policy = socialMask(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        #    #[[pi_1,pi_2,...],[pi_1,pi_2,...],...
        self.pi_step = []
        #    #[[prob_1,prob_2,...],[prob_1,prob_2,...],....
        self.prob_social = []
        # self.reward = [sum(R1),sum(R2),....]
    def select_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state.to(self.device))
        m = Categorical(probs)
        action = m.sample()
        #self.saved_log_probs.append(m.log_prob(action).to(self.device))
        return action.item(),probs        

    def maskFunc(self,probs,masks):
        return F.softmax(torch.mul(probs,masks))
    def update(self,n_agents):
        R = 0
        policy_loss = []
        returns_sum = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns_sum.insert(0, R)
        returns = [ [r]*n_agents for r in returns_sum]
        self.saved_log_probs = []
        for k in range(len(self.pi_step)):
            pi = self.pi_step[k].detach()
            new_probs = self.maskFunc(pi,self.prob_social[k])
            m = Categorical(new_probs)
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action).to(self.device))
        returns = np.array(returns).flatten()
        returns = torch.tensor(returns).to(self.device).type(self.FloatTensor)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.pi_step[:]        
        del self.prob_social[:]


class newPG(PGagent):
    def __init__(self,state_dim,action_dim,agentParam):
        super().__init__(state_dim,action_dim,agentParam)
    def maskFunc(self,probs,masks):
        return F.softmax(torch.mul(probs,masks))
    def select_mask_action(self,state,masks):
        masks = masks.detach()
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state.to(self.device))
        new_probs = self.maskFunc(probs,masks)
        m = Categorical(new_probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action).to(self.device))
        return action.item(),probs


class IAC():
    def __init__(self,action_dim,state_dim):
        self.actor = Actor(action_dim,state_dim)
        self.critic = Critic(state_dim)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.optimizerA = torch.optim.Adam(self.actor.parameters(), lr = 0.001)
        self.optimizerC = torch.optim.Adam(self.critic.parameters(), lr = 0.01)
        self.lr_scheduler = {"optA":torch.optim.lr_scheduler.StepLR(self.optimizerA,step_size=1000,gamma=0.9,last_epoch=-1),
                             "optC":torch.optim.lr_scheduler.StepLR(self.optimizerC,step_size=1000,gamma=0.9,last_epoch=-1)}
        # self.act_prob
        # self.act_log_prob

    def choose_action(self,s):
        s = torch.Tensor(s).squeeze(0)
        self.act_prob = self.actor(s)+0.00001
        m = torch.distributions.Categorical(self.act_prob)
        # self.act_log_prob = m.log_prob(m.sample())
        temp = m.sample()
        return temp

    def cal_tderr(self,s,r,s_):
        s = torch.Tensor(s).unsqueeze(0)
        s_ = torch.Tensor(s_).unsqueeze(0)
        v_ = self.critic(s_).detach()
        v = self.critic(s)
        return r + 0.9*v_ - v

    def learnCritic(self,s,r,s_):
        td_err = self.cal_tderr(s,r,s_)
        loss = torch.mul(td_err,td_err)#torch.square(td_err)
        self.optimizerC.zero_grad()
        loss.backward()
        self.optimizerC.step()
        self.lr_scheduler["optC"].step()

    def learnActor(self,s,r,s_,a):
        td_err = self.cal_tderr(s,r,s_)
        m = torch.log(self.act_prob[a])
        temp = m*td_err.detach()
        loss = -torch.mean(temp)
        self.optimizerA.zero_grad()
        loss.backward()
        self.optimizerA.step()
        self.lr_scheduler["optA"].step()

    def update(self,s,r,s_,a):
        self.learnCritic(s,r,s_)
        self.learnActor(s,r,s_,a)

class Centralised_AC(IAC):
    def __init__(self,action_dim,state_dim):
        super().__init__(action_dim,state_dim)
        self.critic = None

    # def cal_tderr(self,s,r,s_):
    #     s = torch.Tensor(s).unsqueeze(0)
    #     s_ = torch.Tensor(s_).unsqueeze(0)
    #     v = self.critic(s).detach()
    #     v_ = self.critic(s_).detach()
    #     return r + v_ - v

    def learnActor(self,a,td_err):
        m = torch.log(self.act_prob[a])
        temp = m*td_err.detach()
        loss = -torch.mean(temp)
        self.optimizerA.zero_grad()
        loss.backward()
        self.optimizerA.step()
        self.lr_scheduler["optA"].step()

    def update(self,s,r,s_,a,td_err):
        self.learnActor(a,td_err)

class social_IAC(IAC):
    def __init__(self,action_dim,state_dim,agentParam):
        super().__init__(action_dim,state_dim)
        self.saved_log_probs = []
        self.device = agentParam["device"]
        self.reward = []
        self.rewards = []

    def select_masked_action(self,state,masks):
        masks = masks.detach()
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.act_prob = self.actor(state)[0]
        new_probs = self.maskFunc(self.act_prob.detach(),masks)
        print(new_probs)
        m = Categorical(new_probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action).to(self.device))
        return action.item(),self.act_prob

    def maskFunc(self,prob,mask):
        return F.softmax(torch.mul(prob,mask))