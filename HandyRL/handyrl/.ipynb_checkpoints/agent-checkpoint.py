# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# agent classes

import random

import numpy as np

from .util import softmax


class RandomAgent:
    def reset(self, env, show=False):
        pass

    def action(self, env, player, show=False):
        legal_unit_ships = env.legal_units_ships(player)
        legal_unit_shipyards = env.legal_units_shipyards(player)
        
        action_ships = np.zeros((env.num_units(),),dtype=np.int32)
        action_shipyards = np.zeros((env.num_units(),),dtype=np.int32)
        for i in legal_unit_ships:
            action_ships[i] = random.choice(env.legal_actions_ships(player, i))
        for i in legal_unit_shipyards:
            action_shipyards[i] = random.choice(env.legal_actions_shipyards(player, i))
        return {'ships':action_ships,'shipyards': action_shipyards}

    def observe(self, env, player, show=False):
        return [0.0]


class RuleBasedAgent(RandomAgent):
    def __init__(self, key=None):
        self.key = key

    def action(self, env, player, show=False):
        if hasattr(env, 'rule_based_action'):
            return env.rule_based_action(player, key=self.key)
        else:
            return random.choice(env.legal_actions(player))


def print_outputs(env, prob, v):
    if hasattr(env, 'print_outputs'):
        env.print_outputs(prob, v)
    else:
        if v is not None:
            print('v = %f' % v)
        if prob is not None:
            print(prob.shape)
            print('p = %s' % (prob * 1000).astype(int))


class Agent:
    def __init__(self, model, temperature=0.0, observation=True):
        # model might be a neural net, or some planning algorithm such as game tree search
        self.model = model
        self.hidden = None
        self.temperature = temperature
        self.observation = observation

    def reset(self, env, show=False):
        self.hidden = self.model.init_hidden()

    def plan(self, obs):
        outputs = self.model.inference(obs, self.hidden)
        self.hidden = outputs.pop('hidden', None)
        return outputs

    def action(self, env, player, show=False):
        obs = env.observation(player)
        outputs = self.plan(obs)
        p_ships = outputs['policy_ships'].reshape(env.num_units(), -1)
        p_shipyards = outputs['policy_shipyards'].reshape(env.num_units(), -1)
        v = outputs.get('value', None)
        
        mask_ships = env.action_mask_ships(player)
        mask_shipyards = env.action_mask_shipyards(player)
        legal_unit_ships = env.legal_units_ships(player)
        legal_unit_shipyards = env.legal_units_shipyards(player)
        p_ships = softmax(p_ships - mask_ships * 1e32)
        p_shipyards = softmax(p_shipyards - mask_shipyards * 1e32)
        
        action_ships = np.zeros((p_ships.shape[0],),dtype=np.int32)
        action_shipyards = np.zeros((p_shipyards.shape[0],),dtype=np.int32)
        for i in legal_unit_ships:
            action_ships[i] = random.choices(np.arange(p_ships.shape[-1]), weights=p_ships[i])[0]
        for i in legal_unit_shipyards:
            action_shipyards[i] = random.choices(np.arange(p_shipyards.shape[-1]), weights=p_shipyards[i])[0]
        #print(legal_unit)

        if show:
            print_outputs(env, softmax(p_ships), v)

        if self.temperature == 0:
            return {'ships':action_ships,'shipyards': action_shipyards}
            #return [sorted([(a, p_[a]) for a, s in enumerate(p_)], key=lambda x: -x[1])[0][0] for p_ in p]
        else:
            print("gneu")
            return [random.choices(np.arange(len(probs)), weights=probs)[0] for probs in softmax(p / self.temperature)]

    def observe(self, env, player, show=False):
        v = None
        if self.observation:
            obs = env.observation(player)
            outputs = self.plan(obs)
            v = outputs.get('value', None)
            if show:
                print_outputs(env, None, v)
        return v


class EnsembleAgent(Agent):
    def reset(self, env, show=False):
        self.hidden = [model.init_hidden() for model in self.model]

    def plan(self, obs):
        outputs = {}
        for i, model in enumerate(self.model):
            o = model.inference(obs, self.hidden[i])
            for k, v in o.items():
                if k == 'hidden':
                    self.hidden[i] = v
                else:
                    outputs[k] = outputs.get(k, []) + [v]
        for k, vl in outputs.items():
            outputs[k] = np.mean(vl, axis=0)
        return outputs


class SoftAgent(Agent):
    def __init__(self, model):
        super().__init__(model, temperature=1.0)
