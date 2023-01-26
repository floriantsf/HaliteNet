
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from ..environment import BaseEnvironment

class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = torch.cat([x[:,:,:,-self.edge_size[1]:], x, x[:,:,:,:self.edge_size[1]]], dim=3)
        h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h


class HaliteNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 16, 128
        self.conv0 = TorusConv2d(9, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        
        self.head_ships_p = nn.Conv2d(filters, 6, kernel_size=1, stride=1)
        self.head_shipyards_p = nn.Conv2d(filters, 2, kernel_size=1, stride=1)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x, action=None):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        ########################## /!\ ###########################
        # Do we concentrate around the ships of current player ? #
        ########################## /!\ ###########################
        #h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_head = h.view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        ships_p = self.head_ships_p(h)
        shipyards_p = self.head_shipyards_p(h)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))
        
        #ships_logits = ships_p.reshape(-1,6,21*21)
        #shipyards_logits = shipyards_p.reshape(-1,2,21*21)
        #action = torch.cat([ships_logits, shipyards_logits], 1)
        
        p_ships = torch.flatten(ships_p.transpose(-3,-2).transpose(-2,-1),1)
        p_shipyards = torch.flatten(shipyards_p.transpose(-3,-2).transpose(-2,-1),1)
   
        return {'policy_ships': p_ships, 'policy_shipyards': p_shipyards, 'value': v}

class Environment(BaseEnvironment):
    NUM_AGENTS = 2
    def __init__(self, args={}):
        super().__init__()
        self.env = make("halite", debug=True)
        self.reset(self.NUM_AGENTS)
    
    def reset(self, args={}):
        del self.env
        self.env = make("halite", debug=True)
        obs = self.env.reset(self.NUM_AGENTS)
        self.done = False
        self.update((obs, {}), True)
        
    def update(self, info, reset):
        obs, last_actions = info
        if reset:
            self.obs_list = []
        self.obs_list.append(obs)
        self.last_actions = last_actions
    
    def get_observation(self, raw_obs, conf):
        raw_obs['remainingOverageTime'] = 60
        board = Board(raw_observation=raw_obs, \
                      raw_configuration=conf)

        obs = np.zeros((9,\
                        board.configuration.size,\
                        board.configuration.size), dtype=np.float32)

        current_player = board.current_player
        opponents = board.opponents

        # player halite
        obs[7] = current_player.halite/5000
        obs[8] = opponents[0].halite/5000

        # Halite map
        for c in board.cells:
            obs[(0,)+tuple(c)] = board.cells[c].halite/1000

        # Ships map
        for s in current_player.ships:
            obs[(1,)+tuple(s.position)] = 1
        for i,o in enumerate(opponents):
            for s in o.ships:
                obs[(i+2,)+tuple(s.position)] = 1

        # Ships halite map
        for s in current_player.ships:
            obs[(3,)+tuple(s.position)] = s.halite/1000
        for i,o in enumerate(opponents):
            for s in o.ships:
                obs[(i+4,)+tuple(s.position)] = s.halite/1000

        # Shipyard map
        for s in current_player.shipyards:
            obs[(5,)+tuple(s.position)] = 1
        for i,o in enumerate(opponents):
            for s in o.shipyards:
                obs[(i+6,)+tuple(s.position)] = 1

        return obs, board
    
    def step(self, actions):
        # state transition
        game_state = self.obs_list[-1][0].observation
        previous_board = Board(raw_observation=game_state,\
                               raw_configuration=self.env.configuration)
        
        for sy in previous_board.current_player.shipyards:
            a_id = sy.position.x*21+sy.position.y
            if actions.get(0, None)['shipyards'][a_id]:
                sy.next_action = ShipyardAction.SPAWN

        for s in previous_board.current_player.ships:
            a_id = s.position.x*21+s.position.y
            
            action = actions.get(0, None)['ships'][a_id]
            if action and action != 5:
                action = ShipAction.moves()[action-1]
                s.next_action = action
            elif action == 5:
                s.next_action = ShipAction.CONVERT
        
        for i,o in enumerate(previous_board.opponents):
            for sy in o.shipyards:
                a_id = sy.position.x*21+sy.position.y
                if actions.get(i+1, None)['shipyards'][a_id]:
                    sy.next_action = ShipyardAction.SPAWN
                
        for i,o in enumerate(previous_board.opponents):
            for s in o.ships:
                a_id = s.position.x*21+s.position.y

                action = actions.get(i+1, None)['ships'][a_id]
                if action and action != 5:
                    action = ShipAction.moves()[action-1]
                    s.next_action = action
                elif action == 5:
                    s.next_action = ShipAction.CONVERT
                
        obs = self.env.step([previous_board.current_player.next_actions,
                             previous_board.opponents[0].next_actions])
        self.done = not(obs[0].status=='ACTIVE' and obs[1].status=='ACTIVE')
        self.update((obs, actions), False)
                
    def observation(self, player=None):
        if player is None:
            player = 0
        game_state = deepcopy(self.obs_list[-1][0].observation)
        game_state.player = player
        peproc_obs, board = self.get_observation(game_state, self.env.configuration)
                    
        return peproc_obs
    
    def net(self):
        return HaliteNet()
    
    #
    # Should be defined in multi-action settings
    #
    def turns(self):
        # players to move
        #return [p for p in self.players() if self.obs_list[-1][p]['status'] == 'ACTIVE']
        return [0, 1]
    
    def num_units(self):
        return self.env.configuration.size*self.env.configuration.size
    
    def action_mask_ships(self, player):
        return np.zeros((self.env.configuration.size*self.env.configuration.size,6))
    
    def action_mask_shipyards(self, player):
        return np.zeros((self.env.configuration.size*self.env.configuration.size,2))
    
    def legal_units_ships(self, player):
        lu = []
        game_state = self.obs_list[-1][0].observation
        board = Board(raw_observation=game_state,\
                      raw_configuration=self.env.configuration)
        
        if player:
            current_player = board.opponents[0]
        else:
            current_player = board.current_player
            
        for s in current_player.ships:
            lu.append(s._observation[0])
            
        return np.array(lu)
    
    def legal_units_shipyards(self, player):
        lu = []
        game_state = self.obs_list[-1][0].observation
        board = Board(raw_observation=game_state,\
                      raw_configuration=self.env.configuration)
        
        if player:
            current_player = board.opponents[0]
        else:
            current_player = board.current_player
            
        for sy in current_player.shipyards:
            lu.append(sy._observation)
            
        return np.array(lu)
            
    def legal_actions(self, player, pos=None):
        # return legal action list
        return list(range(8))
    
    def terminal(self):
        # check whether terminal state or not
        return self.done
    
    def outcome(self):
        game_state = self.obs_list[-1]
        r1 = game_state[0].reward
        r2 = game_state[1].reward
        if r1 > r2:
            return {0:1,1:-1}
        if r1 < r2:
            return {0:-1,1:1}
        
        return {0:0.5,1:0.5}
    
    def players(self):
        return list(range(self.NUM_AGENTS))
