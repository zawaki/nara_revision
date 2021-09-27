import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def _acceptance_average(path,n=1):
    
    path = '{}/acceptance'.format(path)
    avg = 0
    
    for i in range(n):
        accpt = np.genfromtxt('{}_{}.csv'.format(path,i))[1:]
        avg += np.mean(accpt[-10:])
        
    avg /= n
    
    return '{:.2f}'.format(round(avg,2))
        
def _node_util_average(path,n=1):
    
    path = '{}/util'.format(path)
    cpu_avg = 0
    mem_avg = 0
    
    for i in range(n):
        file = open('{}_{}.csv'.format(path,i))
        util = np.genfromtxt(file,delimiter=',',skip_header=1).transpose()
        cpu = np.mean(util[0][-10:])
        mem = np.mean(util[1][-10:])
        cpu_avg += cpu
        mem_avg += mem
        
    cpu_avg /= n
    mem_avg /= n

    return '{:.2f}'.format(round(cpu_avg,2)),'{:.2f}'.format(round(mem_avg,2))

def _link_util_average(path,n=1):
    
    sr_avg = 0
    ra_avg = 0
    ac_avg = 0
    
    for i in range(n):
        file = open('{}/sr_bw_features_{}.csv'.format(path,i))
        sr_util = np.genfromtxt(file,delimiter=',',skip_header=1)[-10:]
        sr_avg += np.mean(sr_util)
        
        file = open('{}/ra_bw_features_{}.csv'.format(path,i))
        ra_util = np.genfromtxt(file,delimiter=',',skip_header=1)[-10:]
        ra_avg += np.mean(ra_util)
        
        file = open('{}/ac_bw_features_{}.csv'.format(path,i))
        ac_util = np.genfromtxt(file,delimiter=',',skip_header=1)[-10:]
        ac_avg += np.mean(ac_util)

    sr_avg /= n
    ra_avg /= n
    ac_avg /= n
    
    return '{:.2f}'.format(round(sr_avg,2)),'{:.2f}'.format(round(ra_avg,2)),'{:.2f}'.format(round(ac_avg,2))

def _oversub(topology_str):
    
    a,b,c = [int(float(x)) for x in topology_str.split('_')]

    return ((2*b)/(16*a))*(c/b)

def _num_nodes(path,n=1):
        
    path = '{}/success'.format(path)
    avg = 0
    
    for i in range(n):
        nodes = np.loadtxt('{}_{}.csv'.format(path,i),skiprows=1,delimiter=',').transpose()[-1]
        avg += np.mean(nodes)

    avg /= n
    
    return '{:.2f}'.format(round(avg,2))

def _cpu_failures(path,n=1):
        
    path = '{}/success'.format(path)
    avg = 0
    
    for i in range(n):
        cpu = np.loadtxt('{}_{}.csv'.format(path,i),skiprows=1,delimiter=',').transpose()[0]
        avg += np.mean(nodes)

    avg /= n
    
    return '{:.2f}'.format(round(avg,2))

def gen_plot_data(packers=['agent','tetris','nalb','nulb','random'],num_rollouts=5,path_root = '/home/uceezs0/Code/nara_data/uniform/baselines'):
    
    packer_stats = {}
    
    for packer in packers:
        packer_stats[packer] = {'accpt':[],'cpu':[],'mem':[],'sr':[],'ra':[],'ac':[],'nodes':[]}
        
    table_data = {}
    method = []
    tops = []
    accpt = []
    cpu = []
    mem = []
    sr = []
    ra = []
    ac = []
    oversub = []
#     topologies = os.listdir('{}/{}'.format(path_root,'nulb'))
    topologies = ['8.0_32.0_8.0', 
                  '8.0_32.0_16.0', 
                  '8.0_64.0_64.0', 
                  '16.0_64.0_16.0', 
                  '16.0_64.0_32.0', 
                  '16.0_128.0_128.0', 
                  '32.0_128.0_32.0', 
                  '32.0_128.0_64.0', 
                  '32.0_256.0_256.0']

    # topologies.remove('.ipynb_checkpoints')
    #for each topology...
    for topology in topologies:
        #for each packer...
        for packer in packers:
            accpt_avg = _acceptance_average('{}/{}/{}'.format(path_root,packer,topology),n=num_rollouts)
            cpu_avg,mem_avg = _node_util_average('{}/{}/{}'.format(path_root,packer,topology),n=num_rollouts)
            sr_avg,ra_avg,ac_avg = _link_util_average('{}/{}/{}'.format(path_root,packer,topology),n=num_rollouts)

            method.append(packer)
            tops.append(topology)
            accpt.append(accpt_avg)
            cpu.append(cpu_avg)
            mem.append(mem_avg)
            sr.append(sr_avg)
            ra.append(ra_avg)
            ac.append(ac_avg)
            n = _num_nodes('{}/{}/{}'.format(path_root,packer,topology),n=num_rollouts)

            packer_stats[packer]['accpt'].append(accpt_avg)
            packer_stats[packer]['cpu'].append(cpu_avg)
            packer_stats[packer]['mem'].append(mem_avg)
            packer_stats[packer]['sr'].append(sr_avg)
            packer_stats[packer]['ra'].append(ra_avg)
            packer_stats[packer]['ac'].append(ac_avg)
            packer_stats[packer]['nodes'].append(float(n)/float(topology.split('_')[0]))

        oversub.append(_oversub(topology))

    table_data = {
                    'Method':method,
                    'Topology':tops,
                    'Accpt.':accpt,
                    'CPU':cpu,
                    'Mem':mem,
                    'SR-Link':sr,
                    'RA-Link':ra,
                    'AC-Link':ac
    }
        
    return table_data,packer_stats,oversub,topologies

def get_latex_table(table_data):
    
    df = pd.DataFrame(table_data)
    table_tex = df.to_latex(index=False).split('\n')
    tex_final = []
    accpt_topology = []
    cpu_topology = []
    mem_topology = []
    for i in range(len(table_tex)):
        line = table_tex[i]
        if 'tabular' in line:
            line = line.replace('tabular','longtable')
        tex_final.append(line)
        if line[:7] in ['  agent',' tetris','   nalb','   nulb',' random']:
            values = [float(x) for x in line[34:54].split(' &  ')]
            accpt_topology.append(values[0])
            cpu_topology.append(values[1])
            mem_topology.append(values[2])

        if line[:7] == ' random':
            tex_final.append('\midrule')

    for line in tex_final:
        print(line)
        
def get_metric_distributions(oversub,packer_stats):
    
    all_oversubs = np.array(list(set(oversub)))

    for ovsub in all_oversubs:
        indices = np.argwhere(np.array(oversub) == ovsub)
        for metric in ['accpt','cpu','mem','sr','ra','ac']:
            plt.figure()
            plt.xlabel(metric)
            plt.title('bottom-top oversubscription: {}'.format(ovsub))
            for packer in list(packer_stats.keys()):

                values = np.array(packer_stats[packer][metric])[indices]
                values = [float(x) for x in values]

                sns.distplot(values, 
                             hist = False, 
                             kde = True, 
                             rug=True,
                             bins=np.arange(0,1.05,0.05),
                             kde_kws = {'linewidth': 3},
                             label=packer)
                plt.legend()

def get_metric_line_plots(topologies,packer_stats,oversub,oversub_selector=None,channels_selector=None,agent_normalised=False):
    
    reshaped_topology_names = []
    indices = []
    for i in range(len(topologies)):
        tp = topologies[i]
        value = [float(x) for x in tp.split('_')]
        if oversub_selector is not None and oversub[i] != oversub_selector:
            continue
        elif channels_selector is not None and value[0] != channels_selector:
            continue
        else:
            name = '{}\n({})'.format(value[0],oversub[i])
            reshaped_topology_names.append(name)
            indices.append(i)

    for metric in ['accpt','cpu','mem','sr','ra','ac','nodes']:
        plt.figure(figsize=(10,2))
        plt.xlabel('topology')
        plt.ylabel(metric)
        if metric in ['sr','ra','ac','nodes']:
            plt.ylim(0,3.0)
        else:
            plt.ylim(0,1.1)
        plt.title('topology vs {}'.format(metric))
        for packer in list(packer_stats.keys()):
            if not agent_normalised:
                values = np.array(packer_stats[packer][metric])
                values = np.array([float(x) for x in values])[indices]
            else:
                values = np.array(packer_stats[packer][metric])
                values = np.array([float(x) for x in values])
                agent_values = np.array(packer_stats['agent'][metric])
                agent_values = np.array([float(x) for x in agent_values])
                values = (values/agent_values)[indices]

            plt.plot(reshaped_topology_names,values,'o-',label=packer)
        
        plt.legend(ncol=5)