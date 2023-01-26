# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# episode generation

import random
import bz2
import pickle
import scipy

import numpy as np

from .util import softmax


class Generator:
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def generate(self, models, args):
        # episode generation
        moments = []
        hidden = {}
        for player in self.env.players():
            hidden[player] = models[player].init_hidden()

        err = self.env.reset()
        if err:
            return None

        while not self.env.terminal():
            moment_keys = ['observation', 
                           'selected_prob_ships', 'action_mask_ships', 'unit_mask_ships', 'action_ships', 
                           'selected_prob_shipyards', 'action_mask_shipyards', 'unit_mask_shipyards', 'action_shipyards', 
                           'value', 'reward', 'return']
            moment = {key: {p: None for p in self.env.players()} for key in moment_keys}

            turn_players = self.env.turns()
            observers = self.env.observers()
            for player in self.env.players():
                if player not in turn_players + observers:
                    continue
                if player not in turn_players and player in args['player'] and not self.args['observation']:
                    continue

                obs = self.env.observation(player)
                model = models[player]
                outputs = model.inference(obs, hidden[player])
                hidden[player] = outputs.get('hidden', None)
                v = outputs.get('value', None)

                moment['observation'][player] = obs
                moment['value'][player] = v

                if player in turn_players:
                    p_ships_ = outputs['policy_ships'].reshape(self.env.num_units(), -1)
                    p_shipyards_ = outputs['policy_shipyards'].reshape(self.env.num_units(), -1)
                    # handle sparse action_mask and unit_mask
                    # and sparse action and selected prob
                    # in order to avoid OMM issues
                    
                    action_mask_ships = self.env.action_mask_ships(player) * 1e32
                    action_mask_shipyards = self.env.action_mask_shipyards(player) * 1e32
                    legal_unit_ships = self.env.legal_units_ships(player)
                    legal_unit_shipyards = self.env.legal_units_shipyards(player)
                    
                    p_ships = softmax(p_ships_ - action_mask_ships)
                    p_shipyards = softmax(p_shipyards_ - action_mask_shipyards)
                    
                    sparse_p_ships = np.zeros_like(p_ships)
                    sparse_p_shipyards = np.zeros_like(p_shipyards)
                    action_ships = np.zeros((p_ships.shape[0],),dtype=np.int32)
                    action_shipyards = np.zeros((p_ships.shape[0],),dtype=np.int32)
                    unit_mask_ships = np.zeros((self.env.num_units(),))
                    unit_mask_shipyards = np.zeros((self.env.num_units(),))
                    for i in legal_unit_ships:
                        sparse_p_ships[i] = p_ships[i]
                        action_ships[i] = random.choices(np.arange(p_ships.shape[-1]), weights=p_ships[i])[0]
                        unit_mask_ships[i] = 1
                    for i in legal_unit_shipyards:
                        sparse_p_shipyards[i] = p_shipyards[i]
                        action_shipyards[i] = random.choices(np.arange(p_shipyards.shape[-1]), weights=p_shipyards[i])[0]
                        unit_mask_shipyards[i] = 1
                    moment['selected_prob_ships'][player] = scipy.sparse.csr_matrix(np.take_along_axis(sparse_p_ships, np.array(action_ships)[:, None], -1)[:, 0])
                    moment['selected_prob_shipyards'][player] = scipy.sparse.csr_matrix(np.take_along_axis(sparse_p_shipyards, np.array(action_ships)[:, None], -1)[:, 0])
                    moment['action_mask_ships'][player] = scipy.sparse.csr_matrix(action_mask_ships)
                    moment['action_mask_shipyards'][player] = scipy.sparse.csr_matrix(action_mask_shipyards)
                    moment['action_ships'][player] = scipy.sparse.csr_matrix(action_ships)
                    moment['action_shipyards'][player] = scipy.sparse.csr_matrix(action_shipyards)
                    moment['unit_mask_ships'][player] = scipy.sparse.csr_matrix(unit_mask_ships)
                    moment['unit_mask_shipyards'][player] = scipy.sparse.csr_matrix(unit_mask_shipyards)
            
            err = self.env.step({
                p: {
                    'ships':moment['action_ships'][p].toarray()[0] for p in moment['action'],
                    'shipyards':moment['action_shipyards'][p].toarray()[0] for p in moment['action']
                }
            })
            if err:
                return None

            reward = self.env.reward()
            for player in self.env.players():
                moment['reward'][player] = reward.get(player, None)

            moment['turn'] = turn_players
            moments.append(moment)

        if len(moments) < 1:
            return None

        for player in self.env.players():
            ret = 0
            for i, m in reversed(list(enumerate(moments))):
                ret = (m['reward'][player] or 0) + self.args['gamma'] * ret
                moments[i]['return'][player] = ret

        episode = {
            'args': args, 'steps': len(moments),
            'outcome': self.env.outcome(),
            'moment': [
                bz2.compress(pickle.dumps(moments[i:i+self.args['compress_steps']]))
                for i in range(0, len(moments), self.args['compress_steps'])
            ]
        }

        return episode

    def execute(self, models, args):
        episode = self.generate(models, args)
        if episode is None:
            print('None episode in generation!')
        return episode
