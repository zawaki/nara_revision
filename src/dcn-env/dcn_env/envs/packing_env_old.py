 # """
# Trivial gym environment that will be used to sanity check understanding of gym/rllib.

# Reward is defined by avg. utilization per server node w.r.t. their single compute resource ('node').

# Since greedy w.r.t. high utilization corresponds directly to a best-fit heuristic, expect a q-learning
# algorithm (implemented with the generic rllib framework for it) to learn a best fit over time.

# For now, will allow the framework to also choose switch nodes, to see if it can reliably learn to ignore
# all nodes which provide none of the requested resource.

# Network dynamics/allocations are totally ignored in this sanity-check example.
# """

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork

import gym
from gym import error, spaces, utils
from gym.spaces import Discrete, Box, Dict
from gym.utils import seeding

import os
os.environ['USE_OFFICIAL_TFDLPACK'] = "true"
import dgl
from sklearn import preprocessing
from math import inf
import random as rnd
import numpy as np

from NetworkSimulator import *

from gcn import *

NUM_FEATURES = 2
OBS_SPACE_DIM = 1
EMBEDDING_DIM = 1

class PackingEnv(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, env_config):

        super(PackingEnv, self).__init__()

        self.seed = env_config['rnd_seed']
        rnd.seed(self.seed)
        # np.random.seed(self.seed)

        self.num_init_requests = env_config['num_init_requests']
        self.network_dir = env_config['network_dir']
        self.features = env_config['features']
        self.request_type = env_config['request_type']

        self.seed_on_reset = env_config['seed_on_reset']

        self.reset()

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=-10,high=10,shape=(OBS_SPACE_DIM,))
        
        self.path_links = set()

    def reset(self):

        if self.seed_on_reset:
            rnd.seed(self.seed)

        #metric variables
        self.allocated_requests = 0

        '''init manager'''
        self.manager = NetworkManager(self.network_dir)

        '''get DGL graph from manager'''
        self._get_graph()
        self.initial_resource_quantity = np.sum(self.dgl_graph.ndata['features'],axis=0)[0]
        
        self.path_links = set()
        self.num_actions = len(self.dgl_graph.ndata['_ID'])
        # print('REQUEST: {}'.format(self.manager.allocated_requests[self.current_request].requirements))
        
        '''get initial N requests, and set first one (top of buffer)
        as the current one to allocated_requests'''
        for _ in range(self.num_init_requests):
            self.manager.get_request(self.request_type)
        self._update_current_request()
        
        return self._get_observation()
    
    def connectivity_check(self, component_id):
        
        #needs to be same as step, but with validity checking at the beginning

        #get list of nodes currently allocated to current_request, and name of selected action node
        currently_allocated = list(self.manager.allocated_requests[self.current_request].allocations.keys())
        req_network = self.manager.allocated_requests[self.current_request].requirements['ports']

        fail = False
        for comp in currently_allocated:
            path = nx.algorithms.shortest_paths.weighted.dijkstra_path(self.manager.network.graph,component_id,comp)
#             print('PATH: {}'.format(path))
            for i in range(1,len(path)):
                link = self.manager.network.graph.get_edge_data(path[i-1],path[i])['label']

                self.path_links.add(link)
                ports = self.manager.network.components[link].available['ports']

                if ports < req_network:
                    fail = True
                    break
            if fail:
                break

        return fail

    def allocate_network(self):

        for link in self.path_links:
            
            amount = self.manager.allocated_requests[self.current_request].requirements['ports']

            self.manager.allocate(self.current_request,link,'ports',amount)
            self.path_links = set()
    
    def step(self, action):
        # #NOTE: must later augment this implementation to check path-availability
        # '''NOTE: consider better way of dealing with resources that exist in nodes but are not explicitly requested (i.e. 'ports')
        # Currently is just allocating to the request even though it's not being asked for explicitly.
        # '''
        # #allocate max of available or required from the chosen node to the request
        # #NOTE: consider adding condition to penalize (and drop allocation) for empty nodes being selected
        component_id = self.dgl_graph.ndata['_name'][action]
        component_id = str(np.char.decode(component_id.numpy()))

        failed = self.connectivity_check(component_id)
        if failed:
            if self.manager.buffered_requests == {}:
                done = True
                reward = self._get_reward(failed=failed)
                observation = self._get_observation()
                info = {'allocated_requests':self.allocated_requests}
                
                return observation,reward,done,info
            
            else:
#                 print('conn fail return')
                self.manager.de_allocate(self.current_request)
                self.path_links = set()
        
                observation = self._get_observation()
#                 print('getting new request')
                #update request and assume that network holds sufficient resources to allocate it
                self._update_current_request()

                too_large = False
                done = False

                #check if network actually has enough resources to allocate current request
                if self.initial_resource_quantity < self.manager.allocated_requests[self.current_request].requirements['node']:
                    too_large = True

                    #while current request is too large and there are other buffered request
                    while too_large and self.manager.buffered_requests != {}:
                        self._update_current_request()
                        too_large = (self.initial_resource_quantity < self.manager.allocated_requests[self.current_request].requirements['node'])
                
                #finish episode if all buffered requests are too big, else continue
                if not too_large:
                    self._get_graph()
                    done = False
                else:
                    done = True
                    
                reward = self._get_reward(allocated=self.successful_allocation,failed=failed)
                info = {'allocated_requests':self.allocated_requests}
                
                return observation,reward,done,info
        
        else:
            
            for resource in self.features['node']:

                available_quantity = self.manager.network.components[component_id].available[resource]
                required_quantity = self.manager.allocated_requests[self.current_request].requirements[resource]
                allocated_quantity = self.manager.allocated_requests[self.current_request].allocated[resource]
                to_allocate = min(available_quantity,required_quantity-allocated_quantity)

                self.manager.allocate(self.current_request,component_id,resource,to_allocate)

        # get observation
        observation = self._get_observation()

        #give reward
        requested_node = [self.manager.allocated_requests[self.current_request].requirements[resource] for resource in self.features['node']]
        allocated_node = [self.manager.allocated_requests[self.current_request].allocated[resource] for resource in self.features['node']]

        self.successful_allocation = (requested_node == allocated_node)
        
        if self.successful_allocation:

            self.allocated_requests += 1
            #reward for successful allocation. if buffer also not empty, finish. if not, update current_request
            
            #TODO: MAKE THIS MULTI-DIM FOR ARBITRARY RESOURCES
            self.initial_resource_quantity -= np.array(self.manager.allocated_requests[self.current_request].requirements['node'])
            # print('remaining: {}'.format(self.initial_resource_quantity))
            #if no more requests, end episode
            self.allocate_network()
            if self.manager.buffered_requests == {}:
                done = True

            #otherwise, get next request from buffer. only accept one that doesn't require more resources than are in the network.
            else:
#                 print('getting new request')
                #update request and assume that network holds sufficient resources to allocate it
                self._update_current_request()

                too_large = False
                done = False

                #check if network actually has enough resources to allocate current request
                if self.initial_resource_quantity < self.manager.allocated_requests[self.current_request].requirements['node']:
#                     print(self.manager.allocated_requests[self.current_request].requirements['node'])
                    too_large = True

                    #while current request is too large and there are other buffered request
                    while too_large and self.manager.buffered_requests != {}:
                        self._update_current_request()
                        too_large = (self.initial_resource_quantity < self.manager.allocated_requests[self.current_request].requirements['node'])
                
                #finish episode if all buffered requests are too big, else continue
                if not too_large:
                    self._get_graph()
                    done = False
                else:
                    done = True

        else:

            '''if allocation not finished, given relevant reward and continue'''
#             reward = self._get_reward(allocated=False)
            self._get_graph()
            done = False

        reward = self._get_reward(allocated=self.successful_allocation,failed=failed)
        info = {'allocated_requests':self.allocated_requests}

        return observation,reward,done,info

    def _get_observation(self):

        packing_eff = self.manager.packing_efficiency('Resource')
        return np.array([packing_eff])

    def _update_current_request(self):
#         print('UPDATING REQUEST')
        self.new_req = True
        self.current_request = np.random.choice(list(self.manager.buffered_requests.keys()))
        self.manager.move_request_to_allocated(self.current_request)

    def _get_reward(self,allocated=False,failed=False):

        if allocated:
            return 10
        elif failed:
            return -10
        else:
            return -1

    def _get_graph(self):
        '''NOTE: currently implemented to normalise w.r.t the initial requirements. 
        This essentially assumes that re-embeddings are not needed. If going to re-embed every choice, 
        will need to also normalise action space w.r.t. remaining required quantity, not the initial amount.'''

        #initialise the dgl_graph based on raw features of the network
        self.dgl_graph = self.manager.network.to_dgl_with_edges(self.features)

class ParametricActionWrapper(gym.Env):

    def __init__(self, env_config):

        self.wrapped = PackingEnv(env_config)

        num_actions = self.wrapped.action_space.n
        num_edges = self.wrapped.dgl_graph.number_of_edges()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Dict({
            "avail_actions": spaces.Box(-10, 10, shape=(num_actions, len(self.wrapped.features['node']))),
            "action_mask": spaces.Box(0, 1, shape=(num_actions,)),
            "network": self.wrapped.observation_space,
            "edges":spaces.Box(-100, 100, shape=(num_edges, len(self.wrapped.features['link']))),
            "chosen_nodes":spaces.Box(0, 1, shape=(num_actions,))
        })
        self.chosen = np.zeros(num_actions)

    def update_avail_actions(self):
        # print('WRAPPER UPDATE ACTIONS')
        '''
        Define action attribute as being the features normalised wrt request quantities
        and scaled wrt mean and std-dev.
        Do this for the 'features' specified resources so that the request and the
        feature representations have the same components (i.e. no mismatching --> shape errors)
        '''
        #NODE RESOURCES
        req_vec = [self.wrapped.manager.allocated_requests[self.wrapped.current_request].requirements[resource] for resource in self.wrapped.features['node']]
        allocd_vec = [self.wrapped.manager.allocated_requests[self.wrapped.current_request].allocated[resource] for resource in self.wrapped.features['node']]
        '''
        Mask actions which have no resource of any type left
        (i.e. if the sum of all resources at a node is zero, mask it out)
        '''
        remaining_vec = np.array(req_vec) - np.array(allocd_vec)
        lamda_func = lambda x: 1/x if x != 0. else 0.
        vec_func = np.vectorize(lamda_func)
        req_vec = vec_func(remaining_vec)
        mask_idx = tf.where(tf.squeeze(tf.equal(tf.reduce_sum(self.wrapped.dgl_graph.ndata['features'],-1),0.)))

        action_mask = np.ones(self.action_space.n)
        action_mask[tf.squeeze(mask_idx).numpy()] = 0
        self.action_mask = action_mask
        
        self.wrapped.dgl_graph.ndata['orig_features'] = self.wrapped.dgl_graph.ndata['features']
        self.wrapped.dgl_graph.ndata['features'] =  tf.math.multiply(
                                                        self.wrapped.dgl_graph.ndata['features'],
                                                        req_vec
                                                    )
        
        self.wrapped.dgl_graph.ndata['h'] = self.wrapped.dgl_graph.ndata['features']#preprocessing.scale(
        self.action_assignments = tf.keras.activations.sigmoid(tf.math.subtract(self.wrapped.dgl_graph.ndata['features'],1))

        #EDGE RESOURCES
        req_vec = [self.wrapped.manager.allocated_requests[self.wrapped.current_request].requirements[resource] for resource in self.wrapped.features['link']]
        '''
        Mask actions which have no resource of any type left
        (i.e. if the sum of all resources at a node is zero, mask it out)
        '''
        remaining_vec = np.array(req_vec)
        lamda_func = lambda x: 1/x if x != 0. else 0.
        vec_func = np.vectorize(lamda_func)
        req_vec = vec_func(remaining_vec)

        self.wrapped.dgl_graph.edata['features'] =  tf.math.multiply(
                                                        self.wrapped.dgl_graph.edata['features'],
                                                        req_vec
                                                        )
        
        self.wrapped.dgl_graph.edata['h'] = self.wrapped.dgl_graph.edata['features']#preprocessing.scale(
        self.edge_features = tf.keras.activations.sigmoid(tf.math.subtract(self.wrapped.dgl_graph.edata['features'],1))

    def reset(self):

        obs = self.wrapped.reset()
        num_actions = self.wrapped.action_space.n
        self.chosen = np.zeros(num_actions)
        self.update_avail_actions()
        self.wrapped.new_req = False
        
#         print('reset actions: {}'.format(self.action_assignments))
#         print('reset masks: {}'.format(self.action_mask))
#         print('reset network: {}'.format(obs))
#         print('reset chosen: {}'.format(self.chosen))
        
        return {
            "avail_actions": self.action_assignments,
            "action_mask": self.action_mask,
            "network": obs,
            "edges":self.edge_features,
            "chosen_nodes":self.chosen
        }

    def step(self, action):
        
        orig_obs, rew, done, info = self.wrapped.step(action)
        
        if self.wrapped.new_req:
            self.chosen = np.zeros(self.wrapped.action_space.n)
            self.wrapped.new_req = False
        else:
            self.chosen[action] = 1

        self.update_avail_actions()

        obs = {
            "avail_actions": self.action_assignments,
            "action_mask": self.action_mask,
            "network": orig_obs,
            "edges":self.edge_features,
            "chosen_nodes":self.chosen
        } 
#         print('reward: {}'.format(rew))
        return obs, rew, done, info

class ParametricActionsModel(TFModelV2):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(OBS_SPACE_DIM, ),
                 action_embed_size=EMBEDDING_DIM,
                 **kw):
        super(ParametricActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)
        '''NOTE:
        Currenty using a FullyConnectedNetwork since that's what is used in the
        rllib parametric_actions_model example. Doesn't seem to be necessary, 
        except that using this provides a default value function that seems to
        work fine. 
        Could probably just use another keras layer for this, but would then
        have to write a value function which seems uneccesary.
        '''
        self.save_embeddings = model_config['custom_model_config']['embedding_save_dir']
        self.config = model_config
        self.use_gnn = model_config['custom_model_config']['use_gnn']

        if self.use_gnn:
            logits_input_dim = model_config['custom_model_config']['obs_emb_dim'] + \
                                2*model_config['custom_model_config']['agg_dim']
        else:
            logits_input_dim = model_config['custom_model_config']['obs_emb_dim'] + \
                               len(self.config['custom_model_config']['features']['node'])

        self.action_logits_model = tf.keras.Sequential()
        self.action_logits_model.add(tf.keras.layers.Dense(1,input_shape=(None,logits_input_dim),name='logits_dense'))
        self.register_variables(self.action_logits_model.variables)

        #Define and register (via a forced pass through) the GNN model
        if self.use_gnn:
            self.gnn = SAGE(agg_type=model_config['custom_model_config']['agg_type'],
                        agg_dim=model_config['custom_model_config']['agg_dim'],
                        agg_activation='relu',
                        num_mp_stages=model_config['custom_model_config']['num_mp_stages'])

            manager = NetworkManager(model_config['custom_model_config']['graph_dir'])
            self.dgl_graph = manager.network.to_dgl()

        #Define and register the observation embedding model
        final_obs_shape = (true_obs_shape[0],)
        self.action_embed_model = FullyConnectedNetwork(
            Box(-1, 1, shape=final_obs_shape), 
                action_space, model_config['custom_model_config']['obs_emb_dim'],
                model_config, name + "_action_embed")

        self.register_variables(self.action_embed_model.variables())

        self.registered = False

    def forward(self, input_dict, state, seq_lens):

        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        obs = input_dict["obs"]["network"]
        edges = input_dict['obs']['edges']
        chosen_nodes = input_dict['obs']['chosen_nodes']

        # print('init actions: {}'.format(avail_actions))
#         print('step actions: {}'.format(avail_actions))
#         print('step masks: {}'.format(action_mask))
#         print('step network: {}'.format(obs))

        #append chosen to node_features (avail_actions) before GNN passthrough
        # print(tf.expand_dims(chosen_nodes,-1))
        new_feats = tf.concat([avail_actions,tf.expand_dims(chosen_nodes,-1)],-1)
        # print('feature shape: {}'.format(avail_actions.shape))
        # print('new feature shape: {}'.format(new_feats.shape))
        # print('new features: {}'.format(new_feats))

        if self.use_gnn:
            #in first pass-through, initialise the GNN with dummy tensors so they have the correct size.
            #seemed to be neccessary as weren't registering variables otherwise, but should look into it.
            if self.registered == False:
                #+1 is added to account for the binary usage indicator
                tmp_feats = tf.ones([self.dgl_graph.number_of_nodes(),len(self.config['custom_model_config']['features']['node'])+1])
                self.dgl_graph.ndata['features'] = tmp_feats
                self.dgl_graph.ndata['h'] = self.dgl_graph.ndata['features']
                
                tmp_edge = tf.ones([self.dgl_graph.number_of_edges(),len(self.config['custom_model_config']['features']['link'])])
                self.dgl_graph.edata['features'] = tmp_edge
                self.dgl_graph.edata['h'] = self.dgl_graph.edata['features']
                
                self.gnn(self.dgl_graph,mode='no_sampling')

                self.register_variables(self.gnn.variables)
                self.registered = True
            
            #batch all graphs (1 for each set of actions) and do MP on dgl.batched graph (quicker than loop)
            num_batches = avail_actions.shape[0]
            num_actions = avail_actions.shape[1]

            all_graphs = [self.dgl_graph] * num_batches
            all_graphs = dgl.batch(all_graphs)
            
            # avail_actions = tf.reshape(avail_actions,[num_batches*avail_actions.shape[1],avail_actions.shape[2]])
            #tmp change here for usage indicator addition
            avail_actions = tf.reshape(new_feats,[num_batches*new_feats.shape[1],new_feats.shape[2]])
            # print(avail_actions)
            edges = tf.reshape(edges,[num_batches*edges.shape[1],edges.shape[2]])
            
            all_graphs.ndata['features'] = avail_actions
            all_graphs.ndata['h'] = avail_actions
            
            all_graphs.edata['features'] = edges
            all_graphs.edata['h'] = edges
            
            embedded_actions = self.gnn(all_graphs,mode='no_sampling')
            embedded_actions = tf.reshape(embedded_actions,[num_batches,num_actions,embedded_actions.shape[-1]])

        else:
            embedded_actions = avail_actions
#         print('step embeddings: {}'.format(embedded_actions))
        '''POOL PREVIOUSLY CHOSEN ACTIONS AND APPEND TO THE OBSERVATION SPACE
        -if no actions chosen, just use an array of zeros
        -else, mean-pool applied to the current embedding of the nodes that have previously been chosen
        for the current request.
        -the FC-net is applied to the post-append obs vector
        '''

        if tf.math.reduce_sum(chosen_nodes,-1)[0].numpy() == 0:
            pooled_actions = tf.zeros((embedded_actions.shape[0],embedded_actions.shape[-1]))
        else:
            num_chosen = tf.reduce_sum(chosen_nodes,-1)
            batch = chosen_nodes.shape[0]
            num_actions = chosen_nodes.shape[1]
            emb_dim = self.config['custom_model_config']['agg_dim']

            chosen_nodes = tf.reshape(chosen_nodes,[batch*num_actions,1])
            chosen_nodes = tf.repeat(chosen_nodes,emb_dim,axis=-1)
            chosen_nodes = tf.expand_dims(chosen_nodes,0)
            chosen_nodes = tf.cast(tf.reshape(chosen_nodes,[batch,num_actions,emb_dim]),tf.float32)
            
            reduced_actions = tf.math.multiply(embedded_actions,chosen_nodes)
            pooled_actions = tf.reduce_sum(reduced_actions,-2)
            pooled_actions /= tf.cast(num_chosen.numpy()[0],tf.float32)

        action_embed, _ = self.action_embed_model({
            "obs": input_dict["obs"]["network"]
        })

        action_embed = tf.concat([tf.cast(action_embed,tf.float32),tf.cast(pooled_actions,tf.float32)],axis=-1)
        intent_vector = tf.expand_dims(action_embed, 1)
        action_embed = tf.repeat(intent_vector,embedded_actions.shape[-2],axis=1)
#         print('obs embed: {}'.format(action_embed))
        action_obs_embed = tf.concat([action_embed,tf.cast(embedded_actions,tf.float32)],-1)
        # print(action_obs_embed)
        action_logits = tf.squeeze(self.action_logits_model(action_obs_embed),axis=-1)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)

        if self.save_embeddings is not None:
            self.save(tf.squeeze(embedded_actions))
            # print('logits: {}'.format((tf.keras.activations.softmax(action_logits + tf.cast(inf_mask,tf.float32)))))

        return action_logits + tf.cast(inf_mask,tf.float32), state

    def value_function(self):
        return self.action_embed_model.value_function()
    
    def save(self,embeddings):
        np.savetxt(self.save_embeddings,embeddings.numpy())