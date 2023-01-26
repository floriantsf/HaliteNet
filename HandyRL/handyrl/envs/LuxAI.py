from kaggle_environments import make
from lux.config import EnvConfig
from lux.kit import process_obs, to_json, from_json, process_action, obs_to_game_state
import json
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from luxai2022.env import LuxAI2022
from ..environment import BaseEnvironment
import matplotlib.pyplot as plt
import pickle as pkl

class IceLakes():
    def __init__(self, ice, rubble):
        self.factory_map = np.zeros((48,48))
        self.exact_factory_map = np.zeros((48,48))
        clusters = np.zeros((48,48))
        counter = 0
        cls = []
        self.ice = ice
        for i in range(48):
            for j in range(48):
                if not ice[i][j] or clusters[i][j]:
                    continue
                counter += 1
                _cls = []
                min_rubble = rubble[i][j]
                to_visit_stack = [[i,j]]
                while to_visit_stack:
                    el = to_visit_stack.pop()
                    
                    if True:#rubble[el[0]][el[1]] < 50:
                        _cls.append({"pos":el,"rubble":rubble[el[0]][el[1]]})
                    
                    clusters[el[0],el[1]] = counter
                    if min_rubble > rubble[el[0]][el[1]]:
                        min_rubble = rubble[el[0]][el[1]]

                    if el[0]<47 and ice[el[0]+1][el[1]] and not clusters[el[0]+1][el[1]]:
                        to_visit_stack.append((el[0]+1,el[1]))
                    if el[1]<47 and ice[el[0]][el[1]+1] and not clusters[el[0]][el[1]+1]:
                        to_visit_stack.append((el[0],el[1]+1))
                    if el[1]>0 and ice[el[0]][el[1]-1] and not clusters[el[0]][el[1]-1]:
                        to_visit_stack.append((el[0],el[1]-1))
                    if el[0]>0 and ice[el[0]-1][el[1]] and not clusters[el[0]-1][el[1]]:
                        to_visit_stack.append((el[0]-1,el[1]))
                if True:#min_rubble < 50:
                    _cls.sort(reverse=True,key=lambda c : c["rubble"])
                    cls.append({
                        "tiles":_cls,
                        "min_rubble": min_rubble
                    })
        cls.sort(reverse=False,key=lambda c : c["min_rubble"])
        """for c in cls:
            print(f"new class {c['min_rubble']}")
            for e in c["tiles"]:
                print(e["rubble"])"""
            
        self.cls = cls
        self.next_cls = 0
        self.__cls = deepcopy(cls)
    
    def spwan_ice_location(self, spawns, player_id):
        #print(f"{player_id} trying class with {self.cls[self.next_cls]['min_rubble']} min rubble")
        # /!\ All the ascii schemes are upward down : x and y axis have been exchanged
        while self.cls[self.next_cls]["tiles"]:
            el = self.cls[self.next_cls]["tiles"].pop()
            #print(f"tile {el['rubble']}")
            for i in range(len(spawns)):
                # spawn next to the water supply target tile
                # 012
                # !!! 0
                # !x! 1
                # !!! 2
                # .o.
                if el["pos"][1] > 2 and el["pos"][0] > 0 and el["pos"][0] < 47:
                    factory_pos = [el["pos"][0], el["pos"][1]-2]
                    # do not spawn a factory on another factory
                    if not self.factory_map[el["pos"][1], el["pos"][0]] and \
                        not (self.factory_map[factory_pos[1]-1:factory_pos[1]+2,factory_pos[0]-1:factory_pos[0]+2]>0).any():
                        if spawns[i][0] == factory_pos[1] and spawns[i][1] == factory_pos[0]:
                            #print("succes")
                            self.next_cls = (self.next_cls+1)%(len(self.cls))

                            self.factory_map[factory_pos[1]-1:factory_pos[1]+2,factory_pos[0]-1:factory_pos[0]+2] = 1
                            self.exact_factory_map[factory_pos[1],factory_pos[0]] = 1
                            return [factory_pos[1], factory_pos[0]]
                # 0123
                # .!!! 0
                # o!x! 1
                # .!!! 2
                if el["pos"][1]>0 and el["pos"][1]<47 and el["pos"][0]<47-2:
                    factory_pos = [el["pos"][0]+2, el["pos"][1]]
                    if not self.factory_map[el["pos"][1], el["pos"][0]] and \
                        not (self.factory_map[factory_pos[1]-1:factory_pos[1]+2,factory_pos[0]-1:factory_pos[0]+2]>0).any():
                        if spawns[i][0] == factory_pos[1] and spawns[i][1] == factory_pos[0]:
                            #print("succes")
                            self.next_cls = (self.next_cls+1)%(len(self.cls))

                            self.factory_map[factory_pos[1]-1:factory_pos[1]+2,factory_pos[0]-1:factory_pos[0]+2] = 1
                            self.exact_factory_map[factory_pos[1],factory_pos[0]] = 1
                            return [factory_pos[1], factory_pos[0]]
                
                # 012
                # .o.
                # !!!
                # !x!
                # !!!
                if el["pos"][1]<47-2 and el["pos"][0] > 0 and el["pos"][0] < 47:
                    factory_pos = [el["pos"][0], el["pos"][1]+2]
                    if not self.factory_map[el["pos"][1], el["pos"][0]] and \
                        not (self.factory_map[factory_pos[1]-1:factory_pos[1]+2,factory_pos[0]-1:factory_pos[0]+2]>0).any():
                        if spawns[i][0] == factory_pos[1] and spawns[i][1] == factory_pos[0]:
                            #print("succes")
                            self.next_cls = (self.next_cls+1)%(len(self.cls))

                            self.factory_map[factory_pos[1]-1:factory_pos[1]+2,factory_pos[0]-1:factory_pos[0]+2] = 1
                            self.exact_factory_map[factory_pos[1],factory_pos[0]] = 1
                            return [factory_pos[1], factory_pos[0]]

                # 0123
                # !!!. 0
                # !x!o 1
                # !!!. 2
                if el["pos"][1]>0 and el["pos"][1] < 47 and el["pos"][0]>2:
                    factory_pos = [el["pos"][0]-2, el["pos"][1]]
                    if not self.factory_map[el["pos"][1], el["pos"][0]] and \
                        not (self.factory_map[factory_pos[1]-1:factory_pos[1]+2,factory_pos[0]-1:factory_pos[0]+2]>0).any():
                        if spawns[i][0] == factory_pos[1] and spawns[i][1] == factory_pos[0]:
                            #print("succes")
                            self.next_cls = (self.next_cls+1)%(len(self.cls))

                            self.factory_map[factory_pos[1]-1:factory_pos[1]+2,factory_pos[0]-1:factory_pos[0]+2] = 1
                            self.exact_factory_map[factory_pos[1],factory_pos[0]] = 1
                            return [factory_pos[1], factory_pos[0]]
       
        self.next_cls = (self.next_cls+1)%(len(self.cls))
        result = self.spwan_ice_location(spawns, player_id)
        return result

class Agent():
    def __init__(self, player_id):
        self.factories_owned = 0
        self.init_metal_left = 0
        self.init_water_left = 0
        self.prec_water = 0
        self.test_prec_water = 0
        self.prec_fact_water = 0
        self.prec_power = 51
        self.ice_lakes = None
        self.game_state = None
        self.total_water_supplied = 0
        self.total_water_collected = 0

        self.times_dug_upon_ice = 0
        self.total_times_dug_upon_ice = 0
        
        self.current_ice_supplied = 0
        
def early_steps(agent, obs, player, step):
    """
    Logic here to make actions in the early game. Select faction, bid for an extra factory, and place factories
    """
    # various maps to help aid in decision making over factory placement
    rubble = obs["board"]["rubble"]
    # if ice[y][x] > 0, then there is an ice tile at (x, y)
    ice = obs["board"]["ice"]
    # if ore[y][x] > 0, then there is an ore tile at (x, y)
    ore = obs["board"]["ore"]

    if step == 0:
        agent[player].ice_lakes = IceLakes(ice, rubble)
        # decide on a faction, and make a bid for the extra factory. 
        # Each unit of bid removes one unit of water and metal from your initial pool
        faction = "MotherMars"
        if player == "player_1":
            faction = "AlphaStrike"
        return dict(faction=faction, bid=10)
    elif step <= 1:
        # decide on where to spawn the next factory. Returning an empty dict() will skip your factory placement

        # how much water and metal you have in your starting pool to give to new factories
        water_left = obs["teams"][player]["water"]
        metal_left = obs["teams"][player]["metal"]
        # how many factories you have left to place
        factories_to_place = 2#obs["teams"][self.player]["factories_to_place"]
        if step == 1:
            # first step of factory placement, we save our initial pool of water and factory amount
            agent[player].factories_owned = factories_to_place
            agent[player].init_metal_left = metal_left
            agent[player].init_water_left = water_left
        # obs["teams"][self.opp_player] has the same information but for the other team
        # potential spawnable locations in your half of the map
        potential_spawns = obs["board"]["spawns"][player]
        # as a naive approach we randomly select a spawn location and spawn a factory there
        #spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
        spawn_loc = agent[player].ice_lakes.spwan_ice_location(potential_spawns, player)
        #plt.imsave(f"factory_{self.player}_{step}.png", np.array(self.ice_lakes.factory_map))
        return dict(spawn=spawn_loc, 
                       metal=agent[player].init_metal_left // agent[player].factories_owned, 
                       water=agent[player].init_water_left // agent[player].factories_owned)
    else:
        return dict()

def generate_unit(obs, player):
    actions = dict()
    factories = obs["factories"][player]
    # iterate over all active factories
    for unit_id, factory in factories.items():
        actions[unit_id] = 0
    return actions

def move_unit(obs, player):
    actions = dict()
    units = obs["units"][player]
    # iterate over all active factories
    for unit_id, ship in units.items():
        actions[unit_id] = [np.array([0, 1, 0, 0, 0])]
    return actions

class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding='same')
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        #h = torch.cat([x[:,:,:,-self.edge_size[1]:], x, x[:,:,:,:self.edge_size[1]]], dim=3)
        #h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h


class GeeseNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 6, 32

        self.conv0 = TorusConv2d(7, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Conv2d(filters, 7, kernel_size=1, stride=1)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)
        self.head_r = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x, _=None):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,4:5]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = torch.flatten(self.head_p(h).transpose(-3,-2).transpose(-2,-1),1)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))
        r = self.head_r(torch.cat([h_head, h_avg], 1))
        
        return {'policy': p, 'return': r}

class Environment(BaseEnvironment):
    NUM_AGENTS = 2
    ACTION = ['NORTH', 'SOUTH', 'WEST', 'EAST', 'DIG', 'TRANSFER', 'RECHARGE']
    
    def __init__(self, args={}):
        super().__init__()
        self.env = LuxAI2022(verbose=0, validate_action_space=False)
        self.reset()

    def reset(self, args={}):
        del self.env
        self.env = LuxAI2022(verbose=0, validate_action_space=False)
        obs = self.env.reset()
        self.done = False
        self.cstep = 0
        self.agent = {player_id: Agent(player_id) for player_id in obs}
        factories_per_team = np.inf
        while self.cstep <= factories_per_team + 1:
            if self.cstep == 0:
                for p in self.agent:
                    factories_per_team = obs[p]["board"]["factories_per_team"]
            
            obs, _, _, _ = self.env.step({p:process_action(early_steps(self.agent, obs[p], p, self.cstep)) for p in self.agent})
            self.cstep += 1

        obs, _, _, _ = self.env.step({p:process_action(generate_unit(obs[p], p)) for p in self.agent})
        self.cstep += 1
        
        self.update((obs, {}), True)
        for unit_id, factory in obs["player_0"]["factories"]["player_0"].items():
            self.agent["player_0"].prec_fact_water = factory["cargo"]["water"]
        for unit_id, factory in obs["player_1"]["factories"]["player_1"].items():
            self.agent["player_1"].prec_fact_water = factory["cargo"]["water"]
        
    def update(self, info, reset):
        obs, last_actions = info
        if reset:
            self.obs_list = []
            self.obs_list_opp = []
        self.obs_list.append(obs["player_0"])
        self.obs_list_opp.append(obs["player_1"])
        self.last_actions = last_actions

    def action2str(self, a, player=None):
        return self.ACTION[a]

    def str2action(self, s, player=None):
        return self.ACTION.index(s)

    def direction(self, pos_from, pos_to):
        if pos_from is None or pos_to is None:
            return None
        x, y = pos_from // 11, pos_from % 11
        for i, d in enumerate(self.DIRECTION):
            nx, ny = (x + d[0]) % 7, (y + d[1]) % 11
            if nx * 11 + ny == pos_to:
                return i
        return None
    
    def check_free_move(self, new_pos, action, pos):
        # action = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
        if action == 1:
            if pos[1]>0:
                if new_pos[pos[1]-1, pos[0]]:
                    return False
                new_pos[pos[1]-1, pos[0]] = 1
                return True
            else:
                return False
        elif action == 2:
            if pos[0]<47:
                if new_pos[pos[1], pos[0]+1]:
                    return False
                new_pos[pos[1], pos[0]+1] = 1
                return True
            else:
                return False
        elif action == 3:
            if pos[1]<47:
                if new_pos[pos[1]+1, pos[0]]:
                    return False
                new_pos[pos[1]+1, pos[0]] = 1
                return True
            else:
                return False
        elif action == 4:
            if pos[0]>0:
                if new_pos[pos[1], pos[0]-1]:
                    return False
                new_pos[pos[1], pos[0]-1] = 1
                return True
            else:
                return False
        else:
            print("error")
    
    def step(self, actions):
        # state transition
        game_state = self.obs_list[-1]
        a1 = dict()
        a2 = dict()
        self.agent["player_0"].total_water_collected -= self.agent["player_0"].test_prec_water
        self.agent["player_0"].test_prec_water = 0
        
        new_pos1 = np.zeros((48,48))
        for unit_id, unit in game_state["units"]["player_0"].items():
            new_pos1[unit["pos"][1], unit["pos"][0]] = 1
        
        new_pos2 = np.zeros((48,48))
        for unit_id, unit in game_state["units"]["player_1"].items():
            new_pos2[unit["pos"][1], unit["pos"][0]] = 1
        
        for factory_id, factory in game_state["factories"]["player_0"].items():
            if not new_pos1[factory["pos"][1], factory["pos"][0]]:
                a1[factory_id] = 0
        for factory_id, factory in game_state["factories"]["player_1"].items():
            if not new_pos2[factory["pos"][1], factory["pos"][0]]:
                a2[factory_id] = 0
            
        for unit_id, unit in game_state["units"]["player_0"].items():
            self.agent["player_0"].total_water_collected += unit['cargo']['ice']
            self.agent["player_0"].test_prec_water += unit['cargo']['ice']
            pos = unit["pos"][0]+unit["pos"][1]*48
            a = actions.get(0, None)[pos]
            if a <= 3:
                if self.check_free_move(new_pos1, a+1, unit["pos"]):
                    new_pos1[unit["pos"][1], unit["pos"][0]] = 0
                    a1[unit_id] = [np.array([0, a+1, 0, 0, 0])]
            elif a == 4:
                if game_state["board"]["ice"][unit["pos"][1], unit["pos"][0]]:
                    self.agent["player_0"].times_dug_upon_ice += 1
                    self.agent["player_0"].total_times_dug_upon_ice += 1
                a1[unit_id] = [np.array([3, 0, 0, 0, 0])]
            elif a == 5:
                a1[unit_id] = [np.array([1, 0, 0, unit['cargo']['ice'], 0])]
                if unit['cargo']['ice'] > 0:
                    for _, factory in game_state["factories"]["player_0"].items():
                        if unit["pos"][0]<=factory["pos"][0]+1 and \
                            unit["pos"][0]>=factory["pos"][0]-1 and \
                            unit["pos"][1]<=factory["pos"][1]+1 and \
                            unit["pos"][1]>=factory["pos"][1]-1:
                            #print(f"transfer {unit['cargo']['ice']}")
                            #self.agent["player_0"].test_prec_water = 0
                            #self.agent["player_0"].prec_water = 0
                            self.agent["player_0"].current_ice_supplied += unit['cargo']['ice']
                            self.agent["player_0"].total_water_supplied += unit['cargo']['ice']
            else:
                """if unit['power'] < 50 and unit['cargo']['ice'] > 0:
                    print("transfer")"""
                a1[unit_id] = [np.array([2, 0, 4, 50, 0])]
        
        self.agent["player_1"].total_water_collected -= self.agent["player_1"].test_prec_water
        self.agent["player_1"].test_prec_water = 0
        for unit_id, unit in game_state["units"]["player_1"].items():
            self.agent["player_1"].total_water_collected += unit['cargo']['ice']
            self.agent["player_1"].test_prec_water += unit['cargo']['ice']
            pos = unit["pos"][0]+unit["pos"][1]*48
            a = actions.get(1, None)[pos]
            if a <= 3:
                if self.check_free_move(new_pos2, a+1, unit["pos"]):
                    new_pos2[unit["pos"][1], unit["pos"][0]] = 0
                    a2[unit_id] = [np.array([0, a+1, 0, 0, 0])]
            elif a == 4:
                a2[unit_id] = [np.array([3, 0, 0, 0, 0])]
                if game_state["board"]["ice"][unit["pos"][1], unit["pos"][0]]:
                    self.agent["player_1"].times_dug_upon_ice += 1
                    self.agent["player_1"].total_times_dug_upon_ice += 1
            elif a == 5:
                if unit['cargo']['ice'] > 0:
                    for _, factory in game_state["factories"]["player_1"].items():
                        if unit["pos"][0]<=factory["pos"][0]+1 and \
                            unit["pos"][0]>=factory["pos"][0]-1 and \
                            unit["pos"][1]<=factory["pos"][1]+1 and \
                            unit["pos"][1]>=factory["pos"][1]-1:
                            #print(f"transfer {unit['cargo']['ice']}")
                            #self.agent["player_1"].test_prec_water = 0
                            #self.agent["player_1"].prec_water = 0
                            self.agent["player_1"].current_ice_supplied += unit['cargo']['ice']
                            self.agent["player_1"].total_water_supplied += unit['cargo']['ice']
                
                a2[unit_id] = [np.array([1, 0, 0, unit['cargo']['ice'], 0])]
            else:
                """if unit['power'] < 50 and unit['cargo']['ice'] > 0:
                    print("transfer")"""
                a2[unit_id] = [np.array([2, 0, 4, 50, 0])]
        
        obs, _, dones, _ = self.env.step({"player_0":process_action(a1), "player_1":process_action(a2)})
        self.done = dones['player_0'] or dones['player_1']
        self.cstep += 1

        self.update((obs, actions), False)
        
    def diff_info(self, _):
        return self.obs_list[-1], self.last_actions

    def turns(self):
        # players to move
        #return [p for p in self.players() if self.obs_list[-1][p]['status'] == 'ACTIVE']
        return [0, 1]

    def terminal(self):
        # check whether terminal state or not
        if self.obs_list[-1]["real_env_steps"] <= 150 and not self.done:
            return False
        return True
    
    def reward(self):
        game_state = self.obs_list[-1]
        
        r1 = self.agent["player_0"].current_ice_supplied/200
        r2 = self.agent["player_1"].current_ice_supplied/200
        #r1 = -self.agent["player_0"].prec_water
        #r2 = -self.agent["player_1"].prec_water
        
        #self.agent["player_0"].prec_water = 0
        #self.agent["player_1"].prec_water = 0
        self.agent["player_0"].current_ice_supplied = 0
        self.agent["player_1"].current_ice_supplied = 0
        
        """for unit_id, factory in game_state["factories"]["player_0"].items():
            #r1 = (unit.cargo.ice-self.agent["player_0"].prec_water)/2+\
            #     (unit.power-self.agent["player_0"].prec_power)/50
            #r1 = (factory["cargo"]["water"]+1-self.agent["player_0"].prec_fact_water)/500
            self.agent["player_0"].total_water_supplied += (factory["cargo"]["water"]+1-self.agent["player_0"].prec_fact_water)/500
            self.agent["player_0"].prec_fact_water = factory["cargo"]["water"]
            self.agent["player_0"].prec_power = factory["power"]
        
        # 1/500=0.002 2/5000 = 0.0004
        for unit_id, factory in game_state["factories"]["player_1"].items():
            #r2 = (unit.cargo.ice-self.agent["player_1"].prec_water)/2+\
            #     (unit.power-self.agent["player_1"].prec_power)/50
            #r2 = (factory["cargo"]["water"]+1-self.agent["player_1"].prec_fact_water)/500
            self.agent["player_1"].total_water_supplied += (factory["cargo"]["water"]+1-self.agent["player_1"].prec_fact_water)/500
            self.agent["player_1"].prec_fact_water = factory["cargo"]["water"]
            self.agent["player_1"].prec_power = factory["power"]"""
        
        """for unit_id, unit in game_state["units"]["player_0"].items():
            r1 += unit['cargo']['ice']
            self.agent["player_0"].prec_water += unit['cargo']['ice']
            
        for unit_id, unit in game_state["units"]["player_1"].items():
            r2 += unit['cargo']['ice']
            self.agent["player_1"].prec_water += unit['cargo']['ice']
        
        r1 /= 4
        r2 /= 4"""
        """if r1 != 0 or r2 != 0:
            print(r1, r2)
        r1 += self.agent["player_0"].times_dug_upon_ice/8
        r2 += self.agent["player_1"].times_dug_upon_ice/8
        self.agent["player_0"].times_dug_upon_ice = 0
        self.agent["player_1"].times_dug_upon_ice = 0"""
        
        return {0: r1, 1: r2}
    
    def outcome(self):
        # return terminal outcomes
        game_state = self.obs_list[-1]
        
        #r1 = 0
        #r2 = 0
        r1 = self.agent["player_0"].total_water_supplied
        r2 = self.agent["player_1"].total_water_supplied
        #r1 = self.agent["player_0"].total_water_collected
        #r2 = self.agent["player_1"].total_water_collected
        """for unit_id, factory in game_state["factories"]["player_0"].items():
            r1 += factory["cargo"]["ice"]"""
        
        """for unit_id, factory in game_state["factories"]["player_1"].items():
            r2 += factory["cargo"]["ice"]"""
        if r1 > r2:
            print("end", r1, r2, len(game_state["units"]["player_0"]), len(game_state["units"]["player_1"]))
            return {0: 1, 1: -1}
        elif r2 > r1:
            print("end", r1, r2, len(game_state["units"]["player_0"]), len(game_state["units"]["player_1"]))
            return {0: -1, 1: 1}
        else:
            """if self.agent["player_0"].total_times_dug_upon_ice>self.agent["player_1"].total_times_dug_upon_ice:
                return {0: 1, 1: -1}
            elif self.agent["player_0"].total_times_dug_upon_ice>self.agent["player_1"].total_times_dug_upon_ice:
                return {0: -1, 1: 1}
            else:"""
            return {0: 0, 1: 0}
    
    #
    # Should be defined in multi-action settings
    #
    def num_units(self):
        return 48*48
    
    def action_mask(self, player):
        return np.zeros((48*48,len(self.ACTION)))
    
    def legal_units(self, player):
        lu = []
        game_state = self.obs_list[-1]
        for _, unit in game_state["units"][f"player_{player}"].items():
            posx = unit["pos"][0]
            posy = unit["pos"][1]*48
            lu.append(posx+posy)
            
        return np.array(lu)
            
    def legal_actions(self, player, pos=None):
        # return legal action list
        """game_state = self.obs_list[-1]
        for _, unit in game_state["units"][f"player_{player}"].items():
            for _, factory in game_state["factories"][f"player_{player}"].items():
                if unit["pos"][0]==factory["pos"][0] and unit["pos"][1]==factory["pos"][1]:
                    return list(range(len(self.ACTION)))
                
        return list(range(len(self.ACTION)-1))"""
        # unit["pos"] is expressed as (x,y), where (0,0) is top left corner
        """posx = pos % 48
        posy = pos // 48
        game_state = self.obs_list[-1]
        for _, unit in game_state["units"][f"player_{player}"].items():
            if unit["pos"][0] == posx and unit["pos"][1] == posy:
                return list(range(len(self.ACTION)))
        
        return []"""
        return list(range(len(self.ACTION)))

    def players(self):
        return list(range(self.NUM_AGENTS))

    def net(self):
        return GeeseNet()
    
    def observation(self, player=None):
        if player is None:
            player = 0
        game_state = self.obs_list[-1]

        peproc_obs = np.zeros((7,48,48), dtype=np.float32)
        peproc_obs[0] = np.array(game_state["board"]["rubble"])/100
        peproc_obs[1] = np.array(game_state["board"]["ice"])
        for unit_id, factory in game_state["factories"][f"player_{player}"].items():
            peproc_obs[2][factory["pos"][1]-1:factory["pos"][1]+2, factory["pos"][0]-1:factory["pos"][0]+2] = 1+factory["cargo"]["water"]/1000

        for unit_id, factory in game_state["factories"][f"player_{(player+1)%2}"].items():
            peproc_obs[3][factory["pos"][1]-1:factory["pos"][1]+2, factory["pos"][0]-1:factory["pos"][0]+2] = 1+factory["cargo"]["water"]/1000
        
        for unit_id, unit in game_state["units"][f"player_{player}"].items():
            peproc_obs[4][unit["pos"][1], unit["pos"][0]] = 1+unit["power"]/100
            peproc_obs[5][unit["pos"][1], unit["pos"][0]] = unit["cargo"]["ice"]/100
        
        for unit_id, unit in game_state["units"][f"player_{(player+1)%2}"].items():
            peproc_obs[6][unit["pos"][1], unit["pos"][0]] = 1
                    
        return peproc_obs
