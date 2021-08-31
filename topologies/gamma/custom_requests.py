from NetworkSimulator import Request
from AllocationFuncs.distribution_funcs import SingleResource

import random as rnd
import numpy as np

import sqlite3 as lite

config = {'azure':CHANGE TO OWN FILEPATH,
            'alibaba':CHANGE TO OWN FILEPATH}

class AzureRequestTest(Request):

    def __init__(self,config_file,id):
        super(AzureRequestTest,self).__init__(config_file,id)

        conn = lite.connect('/scratch/datasets/azure_cluster/packing_trace_zone_a_v1.sqlite')
        self.cur = conn.cursor()
        self.node_capacity_quantity()
    
    def node_capacity_quantity(self):

        while True:

            self.cur.execute("SELECT core, memory, nic FROM vmType WHERE train=0 ORDER BY RANDOM() LIMIT 1",())
            quantity = self.cur.fetchone()

            self.requirements['node'] = quantity[0]*10
            self.requirements['node'] = int(self.requirements['node']*50)
            self.requirements['node'] = np.float32(self.requirements['node'] + 1) if self.requirements['node'] == 0 else np.float32(self.requirements['node'])

            self.requirements['mem'] = quantity[1]*10
            self.requirements['mem'] = int(self.requirements['mem']*50)
            self.requirements['mem'] = np.float32(self.requirements['mem'] + 1) if self.requirements['mem'] == 0 else np.float32(self.requirements['mem'])

            # self.requirements['ports'] = np.float32(quantity[2]*10)
            # self.requirements['ports'] = np.float32(self.requirements['ports'] + 0.01) if self.requirements['ports'] == 0 else np.float32(self.requirements['ports'])

            if (self.requirements['node'] <= 100) and (self.requirements['mem'] <= 100):# and (self.requirements['ports'] <= 1.0):
                break

        self.requirements['ports'] = np.float32(rnd.uniform(0.1,self.requirements['ports']))

        ht_norm = self.requirements['holding_time']
        self.requirements['holding_time'] = np.float32(rnd.randint(1,self.requirements['holding_time']))
        self.requirements['holding_time_feat'] = np.float32(self.requirements['holding_time']/ht_norm)


class AzureRequest(Request):

    def __init__(self,config_file,id):
        super(AzureRequest,self).__init__(config_file,id)

        conn = lite.connect('/scratch/datasets/azure_cluster/packing_trace_zone_a_v1.sqlite')
        self.cur = conn.cursor()
        self.node_capacity_quantity()
    
    def node_capacity_quantity(self):

        while True:

            self.cur.execute("SELECT core, memory, nic FROM vmType WHERE train=1 ORDER BY RANDOM() LIMIT 1",())
            quantity = self.cur.fetchone()

            self.requirements['node'] = quantity[0]*10
            self.requirements['node'] = int(self.requirements['node']*50)
            self.requirements['node'] = np.float32(self.requirements['node'] + 1) if self.requirements['node'] == 0 else np.float32(self.requirements['node'])

            self.requirements['mem'] = quantity[1]*10
            self.requirements['mem'] = int(self.requirements['mem']*50)
            self.requirements['mem'] = np.float32(self.requirements['mem'] + 1) if self.requirements['mem'] == 0 else np.float32(self.requirements['mem'])

            # self.requirements['ports'] = np.float32(quantity[2]*10)
            # self.requirements['ports'] = np.float32(self.requirements['ports'] + 0.01) if self.requirements['ports'] == 0 else np.float32(self.requirements['ports'])

            if (self.requirements['node'] <= 100) and (self.requirements['mem'] <= 100):# and (self.requirements['ports'] <= 1.0):
                break

        self.requirements['ports'] = np.float32(rnd.uniform(0.1,self.requirements['ports']))

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

        while True:

            self.cur.execute("SELECT cpu_request, mem_size FROM container_meta WHERE train=1 ORDER BY RANDOM() LIMIT 1",())
            quantity = self.cur.fetchone()

            self.requirements['node'] = quantity[0]/800 #assumes octa-core server architecture
            self.requirements['node'] = int(self.requirements['node']*50)

            self.requirements['mem'] = quantity[1]
            self.requirements['mem'] = int(self.requirements['mem']*5)

            if (self.requirements['node'] <= 100) and (self.requirements['mem'] <= 100):
                break

        self.requirements['ports'] = np.float32(rnd.uniform(0.1,self.requirements['ports']))

        ht_norm = self.requirements['holding_time']
        self.requirements['holding_time'] = np.float32(rnd.randint(1,self.requirements['holding_time']))
        self.requirements['holding_time_feat'] = np.float32(self.requirements['holding_time']/ht_norm)

class AlibabaRequestTest(Request):

    def __init__(self,config_file,id):
        super(AlibabaRequestTest,self).__init__(config_file,id)

        conn = lite.connect('/scratch/datasets/alibaba/trace_data.sqlite')
        self.cur = conn.cursor()
        self.node_capacity_quantity()
    
    def node_capacity_quantity(self):

        while True:

            self.cur.execute("SELECT cpu_request, mem_size FROM container_meta WHERE train=0 ORDER BY RANDOM() LIMIT 1",())
            quantity = self.cur.fetchone()

            self.requirements['node'] = quantity[0]/800 #assumes octa-core server architecture
            self.requirements['node'] = int(self.requirements['node']*50)

            self.requirements['mem'] = quantity[1]
            self.requirements['mem'] = int(self.requirements['mem']*5)

            if (self.requirements['node'] <= 100) and (self.requirements['mem'] <= 100):
                break

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