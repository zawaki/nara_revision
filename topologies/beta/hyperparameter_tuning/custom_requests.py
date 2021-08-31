from NetworkSimulator import Request
from AllocationFuncs.distribution_funcs import SingleResource

import random as rnd
import numpy as np

import sqlite3 as lite

class AzureRequest(Request):

    def __init__(self,config_file,id):
        super(AzureRequest,self).__init__(config_file,id)

        conn = lite.connect('/scratch/datasets/azure_cluster/packing_trace_zone_a_v1.sqlite')
        self.cur = conn.cursor()
        self.node_capacity_quantity()
    
    def node_capacity_quantity(self):

        self.cur.execute("SELECT core, memory, nic FROM vmType ORDER BY RANDOM() LIMIT 1",())
        quantity = self.cur.fetchone()

        #mean: 9.137908638233384
        self.requirements['node'] = np.float32(int(quantity[0]*50))
        self.requirements['node'] = np.float32(self.requirements['node'] + 1) if self.requirements['node'] == 0 else np.float32(self.requirements['node'])

        #mean: 8.925525005412426
        self.requirements['mem'] = np.float32(int(quantity[1]*50))
        self.requirements['mem'] = np.float32(self.requirements['mem'] + 1) if self.requirements['mem'] == 0 else np.float32(self.requirements['mem'])

        #mean: 0.16684526412643433
        self.requirements['ports'] = np.float32(quantity[2])
        self.requirements['ports'] = np.float32(self.requirements['ports'] + 0.01) if self.requirements['ports'] == 0 else np.float32(self.requirements['ports'])

        self.requirements['holding_time'] *= 100
        ht_norm = self.requirements['holding_time']
        self.requirements['holding_time'] = np.float32(rnd.randint(1,self.requirements['holding_time']))
        self.requirements['holding_time_feat'] = np.float32(self.requirements['holding_time']/ht_norm)

class AlibabaRequest(Request):

    def __init__(self,config_file,id):
        super(AlibabaRequest,self).__init__(config_file,id)

        conn = lite.connect('/scratch/datasets/alibaba/trace_data.sqlite')
        self.cur = conn.cursor()
        self.node_capacity_quantity()
    
    def node_capacity_quantity(self):

        self.cur.execute("SELECT cpu_request, mem_size FROM container_meta ORDER BY RANDOM() LIMIT 1",())
        quantity = self.cur.fetchone()

        #mean: 7.599278348567897
        self.requirements['node'] = np.float32(int((quantity[0]/800)*12))

        #mean: 4.260288390695716
        self.requirements['mem'] = np.float32(int((quantity[1]/100)*240))
        self.requirements['mem'] = np.float32(self.requirements['mem']+1) if self.requirements['mem'] == 0 else np.float32(self.requirements['mem'])

        self.requirements['ports'] = np.float32(rnd.uniform(0.1,self.requirements['ports']))

        ht_norm = self.requirements['holding_time']
        self.requirements['holding_time'] = np.float32(rnd.randint(1,self.requirements['holding_time']))
        self.requirements['holding_time_feat'] = np.float32(self.requirements['holding_time']/ht_norm)

class SingleResourceRequest(Request):

    def __init__(self,config_file,id):

        super(SingleResourceRequest,self).__init__(config_file,id)
        self.node_capacity_quantity()

    def node_capacity_quantity(self):
        self.requirements['node'] = np.float32(rnd.randint(1,50))
        self.requirements['mem'] = np.float32(rnd.randint(1,50))
        self.requirements['ports'] = np.float32(rnd.uniform(0.1,self.requirements['ports']))
        ht_norm = self.requirements['holding_time']
        self.requirements['holding_time'] = np.float32(rnd.randint(1,self.requirements['holding_time']))
        self.requirements['holding_time_feat'] = np.float32(self.requirements['holding_time']/ht_norm)
