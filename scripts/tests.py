import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env

import packing_test
from ray.rllib.models import ModelCatalog
from dcn_env.envs.packing_env import PackingEnv, ParametricActionWrapper, ParametricActionsModel#, NonParametricActionsModel
import os

import gym
from gym.envs.registration import register

import json
import numpy as np

from rollout import rollout, PackingAgent
from plots import acceptance_ratio, util, failure, acceptance_load, ratio

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf

import time
import argparse

ray.shutdown()
ray.init(temp_dir='/tmp/uceezs0_ray_tmp_0',ignore_reinit_error=True)

with open('{}/params.json'.format('../models/{}'.format('uniform')),'r') as f:
    config = json.load(f)
'uniform'
#kwargs
config['env_config']['rnd_seed'] = 10
config['env_config']['num_init_requests'] = 2
config['env_config']['request_type'] = 'SingleResourceRequest'
config['env_config']['network_dir'] = '../topologies/{}'.format('alpha')
config['model']['custom_model_config']['graph_dir'] = '../topologies/{}'.format('alpha')
config['model']['custom_model_config']['top_k'] = 1

config['env_config']['lb_route_weighting'] = False
env = ParametricActionWrapper(env_config=config['env_config'])

env.wrapped.manager.network.show_graph()
dgl_graph = env.wrapped.manager.network.to_dgl_with_edges({'node':['node','mem'],'link':['ports']})
# print(dgl_graph.ndata['_name'][10])
env.step(10)