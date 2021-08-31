import tensorflow as tf
import os
os.environ['USE_OFFICIAL_TFDLPACK'] = "true"
import dgl

#message-passing layers
class MeanPool2(tf.keras.layers.Layer):

    #max pool as defined in the GraphSAGE paper
    def __init__(self,output_dim,dropout=0.5,activation=None):

        super(MeanPool2,self).__init__()
        
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation

        self.mean_layer = tf.keras.layers.Dense(self.output_dim,activation=self.activation)    
        self.drop = tf.keras.layers.Dropout(self.dropout)
        self.act = tf.keras.layers.Activation(self.activation)
    
    def call(self,graph,feats,stage=None,mp_stages=None,mode='sampling'):

        if mode == 'sampling':
                
            nf = graph

            for i in range(stage,mp_stages):
                nf.block_compute(i,
                                message_func=self._sample_mp_func,
                                reduce_func=self._sample_reduce_func
                                )

            for i in range(stage,mp_stages):
                nf.layers[i+1].data['h'] = nf.layers[i+1].data['new_h']
                    
            return tf.math.l2_normalize(self.act(nf.layers[-1].data['h']),axis=-1)

        if mode == 'no_sampling':
            '''NOTE: for whole graph implementation, the NN pass-throughs
            are done outside of the MP call (update_all). This is because 
            otherwise the input-output values are tracked by GradientTape
            during MP, which will easily overflow the GPU memory for large graphs.
                
            This is why dgl.function.copy_src/max are used rather than the 
            self.'''
            graph.update_all(
                            message_func=self._sample_mp_func,
                            reduce_func=self._sample_reduce_func
                            )

            graph.ndata['h'] = graph.ndata['new_h']
            return tf.math.l2_normalize(self.act(graph.ndata['h']),axis=-1)

    def _sample_mp_func(self,edges):
        self.edge_feat_size = edges.data['h'].shape[-1]
#         print('edge feat size: {}'.format(self.edge_feat_size))
        return {'m' : tf.concat([tf.cast(edges.src['h'],tf.float32),tf.cast(edges.data['h'],tf.float32)],-1)}
    
    def _sample_reduce_func(self,nodes):

        zeros = tf.zeros((nodes.data['h'].shape[0],self.edge_feat_size))

        state_self = tf.concat([tf.cast(nodes.data['h'],tf.float32),tf.cast(zeros,tf.float32)],-1)
        state_self = tf.expand_dims(state_self,1)

        all_msgs = self.mean_layer(tf.concat([state_self,nodes.mailbox['m']],1))
        all_msgs = tf.reduce_mean(all_msgs, axis=-2)
        # print('messages: {}'.format(all_msgs.shape))

        return {'new_h' : all_msgs}

class MeanPoolEdgeData(tf.keras.layers.Layer):

    #max pool as defined in the GraphSAGE paper
    def __init__(self,output_dim,dropout=0.5,activation=None):

        super(MeanPoolEdgeData,self).__init__()
        
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation

        self.mean_layer = tf.keras.layers.Dense(self.output_dim,activation=self.activation)    
        self.drop = tf.keras.layers.Dropout(self.dropout)
        self.act = tf.keras.layers.Activation(self.activation)
    
    def call(self,graph,feats,stage=None,mp_stages=None,mode='sampling'):

        if mode == 'sampling':
                
            nf = graph

            for i in range(stage,mp_stages):
                nf.block_compute(i,
                                message_func=self._sample_mp_func,
                                reduce_func=self._sample_reduce_func
                                )

            for i in range(stage,mp_stages):
                nf.layers[i+1].data['h'] = nf.layers[i+1].data['new_h']
                    
            return tf.math.l2_normalize(self.act(nf.layers[-1].data['h']),axis=-1)

        if mode == 'no_sampling':
            '''NOTE: for whole graph implementation, the NN pass-throughs
            are done outside of the MP call (update_all). This is because 
            otherwise the input-output values are tracked by GradientTape
            during MP, which will easily overflow the GPU memory for large graphs.
                
            This is why dgl.function.copy_src/max are used rather than the 
            self.'''
            graph.update_all(
                            message_func=self._sample_mp_func,
                            reduce_func=self._sample_reduce_func
                            )

            graph.ndata['h'] = graph.ndata['new_h']
            return tf.math.l2_normalize(self.act(graph.ndata['h']),axis=-1)

    def _sample_mp_func(self,edges):
        self.edge_feat_size = edges.data['h'].shape[-1]
#         print('edge feat size: {}'.format(self.edge_feat_size))
        return {'m' : tf.concat([tf.cast(edges.src['h'],tf.float32),tf.cast(edges.data['h'],tf.float32)],-1)}
    
    def _sample_reduce_func(self,nodes):

        zeros = tf.zeros((nodes.data['h'].shape[0],self.edge_feat_size))

        state_self = tf.concat([tf.cast(nodes.data['h'],tf.float32),tf.cast(zeros,tf.float32)],-1)
        state_self = tf.expand_dims(state_self,1)

        all_msgs = tf.concat([state_self,nodes.mailbox['m']],1)
        all_msgs = tf.reduce_mean(all_msgs, axis=-2)
        return {'new_h' : self.mean_layer(all_msgs)}

class MeanPool(tf.keras.layers.Layer):

    #max pool as defined in the GraphSAGE paper
    def __init__(self,output_dim,dropout=0.5,activation=None):

        super(MeanPool,self).__init__()
        
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation

        self.mean_layer = tf.keras.layers.Dense(self.output_dim,activation=self.activation)    
        self.drop = tf.keras.layers.Dropout(self.dropout)
        self.act = tf.keras.layers.Activation(self.activation)
    
    def call(self,graph,feats,stage=None,mp_stages=None,mode='sampling'):

        if mode == 'sampling':
                
            nf = graph

            for i in range(stage,mp_stages):
                nf.block_compute(i,
                                message_func=self._sample_mp_func,
                                reduce_func=self._sample_reduce_func
                                )

            for i in range(stage,mp_stages):
                nf.layers[i+1].data['h'] = nf.layers[i+1].data['new_h']
                    
            return tf.math.l2_normalize(self.act(nf.layers[-1].data['h']),axis=-1)

        if mode == 'no_sampling':
            '''NOTE: for whole graph implementation, the NN pass-throughs
            are done outside of the MP call (update_all). This is because 
            otherwise the input-output values are tracked by GradientTape
            during MP, which will easily overflow the GPU memory for large graphs.
                
            This is why dgl.function.copy_src/max are used rather than the 
            self.'''
            graph.update_all(
                            message_func=self._sample_mp_func,
                            reduce_func=self._sample_reduce_func
                            )
            graph.ndata['h'] = graph.ndata['new_h']
            return tf.math.l2_normalize(self.act(graph.ndata['h']),axis=-1)

    def _sample_mp_func(self,edges):

        return {'m' : edges.src['h']}
    
    def _sample_reduce_func(self,nodes):
        msgs = tf.reduce_mean(nodes.mailbox['m'], axis=-2)
        return {'new_h' : self.mean_layer(msgs)}

#message-passing layers
class MaxPool(tf.keras.layers.Layer):

    #max pool as defined in the GraphSAGE paper
    def __init__(self,output_dim,input_shape=None,hidden_dim=16,dropout=0.5,activation=None):

        super(MaxPool,self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation

        self.agg_layer = tf.keras.layers.Dense(self.hidden_dim,activation=self.activation)
        self.neighbour_layer = tf.keras.layers.Dense(self.output_dim,activation=self.activation)
        self.node_layer = tf.keras.layers.Dense(self.output_dim,activation=self.activation)
        self.drop = tf.keras.layers.Dropout(self.dropout)
        self.act = tf.keras.layers.Activation(self.activation)
    
    def call(self,graph,feats,stage=None,mp_stages=None,mode='sampling'):

        if mode == 'sampling':
                
            nf = graph

            for i in range(stage,mp_stages):
                nf.block_compute(i,
                                message_func=self._sample_mp_func,
                                reduce_func=self._sample_reduce_func
                                )
            for i in range(stage,mp_stages):
                nf.layers[i+1].data['h'] = nf.layers[i+1].data['new_h']
                    
            return tf.math.l2_normalize(self.act(nf.layers[-1].data['h']),axis=-1)

        if mode == 'no_sampling':
            '''NOTE: for whole graph implementation, the NN pass-throughs
            are done outside of the MP call (update_all). This is because 
            otherwise the input-output values are tracked by GradientTape
            during MP, which will easily overflow the GPU memory for large graphs.
                
            This is why dgl.function.copy_src/max are used rather than the 
            self.'''
            #first pass through of features before sending message
            h = self.node_layer(graph.ndata['h'])
            graph.ndata['h'] = self.agg_layer(graph.ndata['h'])

            graph.update_all(
                            message_func=self._sample_mp_func,
                            reduce_func=self._sample_reduce_func
                            )

            #reduce size of agg messages and init features
            neighs = self.neighbour_layer(graph.ndata['agg'])
            new_h = tf.concat([h,neighs],-1)
            graph.ndata['h'] = new_h

            return tf.math.l2_normalize(self.act(new_h),axis=-1)

    def _sample_mp_func(self,edges):
        return {'m' : edges.src['h']}
    
    def _sample_reduce_func(self,nodes):
        return {'agg' : tf.reduce_max(nodes.mailbox['m'], axis=-2)}


class Attention(tf.keras.layers.Layer):

    def __init__(self,output_dim,coef_dim=128,att_dim=256,activation=None):

        super(Attention,self).__init__()

        self.output_dim = output_dim
        self.coef_dim = coef_dim
        self.att_dim = att_dim
        self.activation = activation
    
    def build(self,input_shape):

        self.coef_layer = tf.keras.layers.Dense(self.coef_dim,activation=self.activation)
        self.att_layer = tf.keras.layers.Dense(self.att_dim,activation=self.activation)
        self.reduce_layer = tf.keras.layers.Dense(1,activation=self.activation)
    
    def call(self,graph,feats,stage=None,mp_stages=None,mode='sampling'):

        if mode == 'sampling':
                
            nf = graph
            for i in range(stage,mp_stages):
                nf.block_compute(i,
                                message_func=self._sample_mp_func,
                                reduce_func=self._sample_reduce_func
                                )
            for i in range(stage,mp_stages):
                nf.layers[i+1].data['h'] = nf.layers[i+1].data['new_h']
                    
            return tf.math.l2_normalize(self.act(nf.layers[i+1].data['h']),axis=-1)

    def _sample_mp_func(self,edges):
        return {'m' : self.coef_layer(edges.src['h']),'neigh': edges.src['h']}
    
    def _sample_reduce_func(self,nodes):
        m = nodes.mailbox['m']
        neigh = nodes.mailbox['neigh']
        att_coefs = []
        for message in m:
            att_coefs.append(self.reduce_layer(tf.concat([self.coef_layer(nodes.data['h'],message)],-1)))
        return {'new_h' : tf.concat([self.node_layer(nodes.data['h']),self.neighbour_layer(msgs)],-1)}
    
#read-out layers
class SingleLayerReadout(tf.keras.layers.Layer):
    
    def __init__(self,output_dim,activation='softmax'):
        
        super(SingleLayerReadout,self).__init__()
        
        self.units = output_dim
        self.activation = activation
        
    def build(self,input_shape):
        
        self.readout = tf.keras.layers.Dense(self.units,activation=self.activation)
    
    def call(self,inputs,mode='node'):
        
        if mode == 'node':
            return self.readout(inputs)
        
        elif mode == 'graph':
            '''TODO: implement a full-graph readout here
            for use with graph-batched datasets.'''
            pass