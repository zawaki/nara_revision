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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', nargs='?', default='delta')
    parser.add_argument('--test_baselines', nargs='?', default='yes')
    parser.add_argument('--agent', nargs='?', default='uniform')
    parser.add_argument('--dataset', nargs='?', default='uniform')
    parser.add_argument('--episode_length', nargs='?', default='512',type=int)
    parser.add_argument('--iterations', nargs='?', default='5',type=int)
    parser.add_argument('--save_dir', nargs='?', default='../../nara_data/large_graph/baselines/')
    args = parser.parse_args()

    #kwargs
    if args.dataset == 'uniform':
        request_type = 'SingleResourceRequest'
    elif args.dataset == 'azure':
        request_type = 'AzureRequestTest'
    elif args.dataset == 'alibaba':
        request_type = 'AlibabaRequestTest'

    model_dir = '../models/{}'.format(args.agent)

    ray.shutdown()
    ray.init(temp_dir='/tmp/uceezs0_ray_tmp_0',ignore_reinit_error=True)

    checkpoint_path = [
        '8.0_64.0_64.0/PPO/PPO_pa_network_0_lr=0.005,sgd_minibatch_size=256,train_batch_size=2048_2021-09-24_14-13-29ij6kbta8']

    check_dir_0 = '/home/uceezs0/Code/nara_data/uniform/agent_train/{}'.format(checkpoint_path[0])
    check_dir_1 = 'checkpoint_100/checkpoint-100'
    with open('{}/params.json'.format(check_dir_0),'r') as f:
        config = json.load(f)

    print(config.keys())

    #kwargs
    config['env_config']['rnd_seed'] = 10
    config['env_config']['num_init_requests'] = args.episode_length
    config['env_config']['request_type'] = request_type
    config['env_config']['network_dir'] = '../topologies/{}'.format(args.topology)
    config['model']['custom_model_config']['graph_dir'] = '../topologies/{}'.format(args.topology)
    config['model']['custom_model_config']['top_k'] = 1

    register_env("pa_network", lambda config: ParametricActionWrapper(config))
    ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)

    agent = ppo.PPOTrainer(config=config)
    agent.restore('{}'.format('{}/{}'.format(check_dir_0,check_dir_1)))

    if args.test_baselines:
        tetris = PackingAgent('tetris')
        nulb = PackingAgent('nulb')
        nalb = PackingAgent('nalb')
        random = PackingAgent('random')

    n = 1
    low_tier_channels = [8.0,16.0,32.0]
    oversubscription_ratio_multipliers = [(n*8,n*8),(n*4,n*2),(n*4,n*1),(n*2,n*0.5)]

    all_combos = []

    for t_ch in low_tier_channels:
        for ov_r in oversubscription_ratio_multipliers:
            all_combos.append([t_ch,ov_r[0]*t_ch,ov_r[1]*t_ch])
    ##############################

    for i in range(len(all_combos)):

        combo = all_combos[i]
        print(combo)
        sr_channel = combo[0]
        ra_channel = combo[1]
        ac_channel = combo[2]

        agent = ppo.PPOTrainer(config=config)
        agent.restore('/home/uceezs0/Code/nara_data/uniform/agent_train/{}/checkpoint_100/checkpoint-100'.format(checkpoint_path[i]))

        print(checkpoint_path[i])
        print(combo)

        with open('../topologies/{}/components/SRLink.txt'.format(args.topology),'r+') as f:
            tmp_config = json.load(f)
            tmp_config['ports'] = sr_channel
            f.close()

        with open('../topologies/{}/components/SRLink.txt'.format(args.topology),'w') as f:
            f.write(json.dumps(tmp_config))
            f.close()

        with open('../topologies/{}/components/RALink.txt'.format(args.topology),'r+') as f:
            tmp_config = json.load(f)
            tmp_config['ports'] = ra_channel
            f.close()
        with open('../topologies/{}/components/RALink.txt'.format(args.topology),'w') as f:
            f.write(json.dumps(tmp_config))
            f.close()

        with open('../topologies/{}/components/ACLink.txt'.format(args.topology),'r+') as f:
            tmp_config = json.load(f)
            tmp_config['ports'] = ac_channel
            f.close()

        with open('../topologies/{}/components/ACLink.txt'.format(args.topology),'w') as f:
            f.write(json.dumps(tmp_config))
            f.close()

        config['env_config']['lb_route_weighting'] = False
        env = ParametricActionWrapper(env_config=config['env_config'])
        rollout(agent,env,'{}/agent/{}_{}_{}'.format(args.save_dir,sr_channel,ra_channel,ac_channel),rl=True,iterations=args.iterations)