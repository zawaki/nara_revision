import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid.inset_locator import inset_axes
import matplotlib.gridspec as gridspec


AVERAGE_SAMPLES = 25

def _acceptance_average(path,n=1):
    
    path = '{}/acceptance'.format(path)
    avg = 0
    
    for i in range(n):
        accpt = np.genfromtxt('{}_{}.csv'.format(path,i))[1:]
        avg += np.mean(accpt[-AVERAGE_SAMPLES:])
        
    avg /= n
    
    return '{:.2f}'.format(round(avg,2))

def _success_nodes(path,n=1):
    
    path = '{}/success'.format(path)
    all_sizes = []
    
    for i in range(n):
        cpu = np.loadtxt('{}_{}.csv'.format(path,i),skiprows=1,delimiter=',')
        if cpu != []:
            cpu = cpu.transpose()[0]
        mem = np.loadtxt('{}_{}.csv'.format(path,i),skiprows=1,delimiter=',')
        if mem != []:
            mem = mem.transpose()[1]
#         all_sizes += list(np.maximum(cpu,mem)/16) 
        all_sizes += list((cpu+mem))
#         all_sizes += list(cpu)
#         all_sizes += list(mem)
    
    return all_sizes

def _ratio_average(path,n=1):
    
    path = '{}/success'.format(path)
    all_sizes = []
    
    for i in range(n):
        ratio = np.loadtxt('{}_{}.csv'.format(path,i),skiprows=1,delimiter=',')
        if ratio != []:
            ratio = ratio.transpose()[-2]

        all_sizes += list(ratio)
    
    return '{:.2f}'.format(round(np.mean(all_sizes),2))

def _ratio_full(path,n=1):
    
    path = '{}/success'.format(path)
    all_sizes = []
    
    for i in range(n):
        ratio = np.loadtxt('{}_{}.csv'.format(path,i),skiprows=1,delimiter=',')
        if ratio != []:
            ratio = ratio.transpose()[-2]

        all_sizes += list(ratio)
    
    return all_sizes


def _failure_nodes(path,n=1):
    
    path = '{}/failure'.format(path)
    all_sizes = []
    
    for i in range(n):
        cpu = np.loadtxt('{}_{}.csv'.format(path,i),skiprows=1,delimiter=',')
        if cpu != []:
            cpu = cpu.transpose()[0]
        mem = np.loadtxt('{}_{}.csv'.format(path,i),skiprows=1,delimiter=',')
        if mem != []:
            mem = mem.transpose()[1]
#         all_sizes += list(np.maximum(cpu,mem)/16) 
        all_sizes += list((cpu+mem))
#         all_sizes += list(cpu)
#         all_sizes += list(mem)
    
    return all_sizes
        
def _node_util_average(path,n=1):
    
    path = '{}/util'.format(path)
    cpu_avg = 0
    mem_avg = 0
    
    for i in range(n):
        file = open('{}_{}.csv'.format(path,i))
        util = np.genfromtxt(file,delimiter=',',skip_header=1).transpose()
        cpu = np.mean(util[0][-AVERAGE_SAMPLES:])
        mem = np.mean(util[1][-AVERAGE_SAMPLES:])
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
        sr_util = np.genfromtxt(file,delimiter=',',skip_header=1)[-AVERAGE_SAMPLES:]
        sr_avg += np.mean(sr_util)
        
        file = open('{}/ra_bw_features_{}.csv'.format(path,i))
        ra_util = np.genfromtxt(file,delimiter=',',skip_header=1)[-AVERAGE_SAMPLES:]
        ra_avg += np.mean(ra_util)
        
        file = open('{}/ac_bw_features_{}.csv'.format(path,i))
        ac_util = np.genfromtxt(file,delimiter=',',skip_header=1)[-AVERAGE_SAMPLES:]
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
        packer_stats[packer] = {'accpt':[],'cpu':[],'mem':[],'sr':[],'ra':[],'ac':[],'ratio':[]}
        
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
#     topologies = os.listdir('{}/{}'.format(path_root,'agent'))
    topologies = [
                  '8.0_16.0_4.0',
                  '8.0_32.0_8.0', 
                  '8.0_32.0_16.0', 
                  '8.0_64.0_64.0', 
                  '16.0_32.0_8.0',
                  '16.0_64.0_16.0', 
                  '16.0_64.0_32.0', 
                  '16.0_128.0_128.0', 
                  '32.0_64.0_16.0',
                  '32.0_128.0_32.0', 
                  '32.0_128.0_64.0', 
                  '32.0_256.0_256.0'
    ]

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
#             n = _num_nodes('{}/{}/{}'.format(path_root,packer,topology),n=num_rollouts)
            n = _ratio_average('{}/{}/{}'.format(path_root,packer,topology),n=num_rollouts)

            packer_stats[packer]['accpt'].append(accpt_avg)
            packer_stats[packer]['cpu'].append(cpu_avg)
            packer_stats[packer]['mem'].append(mem_avg)
            packer_stats[packer]['sr'].append(sr_avg)
            packer_stats[packer]['ra'].append(ra_avg)
            packer_stats[packer]['ac'].append(ac_avg)
#             packer_stats[packer]['nodes'].append(float(n)/float(topology.split('_')[0]))
            packer_stats[packer]['ratio'].append(float(n))

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
        
def get_comparison_table(packer_stats,topologies,oversub):
    
    channels = ['8.0','16.0','32.0']
    channel_dict = {}
    for channel in channels:
        channel_dict[channel] = 0
    
    table = {}
    for os in oversub:
        table[os] = channel_dict.copy()
    for i in range(len(oversub)):
        agent_accpt = packer_stats['agent']['accpt'][i]
        baseline_accpt = max([packer_stats[baseline]['accpt'][i] for baseline in ['tetris','nalb','nulb','random']])
        improvement_accpt = (float(agent_accpt)/float(baseline_accpt) - 1)*100
        
        agent_cpu = packer_stats['agent']['cpu'][i]
        baseline_cpu = max([packer_stats[baseline]['cpu'][i] for baseline in ['tetris','nalb','nulb','random']])
        improvement_cpu = (float(agent_cpu)/float(baseline_cpu) - 1)*100
        
        agent_mem = packer_stats['agent']['mem'][i]
        baseline_mem = max([packer_stats[baseline]['mem'][i] for baseline in ['tetris','nalb','nulb','random']])
        improvement_mem = (float(agent_mem)/float(baseline_mem) - 1)*100
        
        channel = topologies[i].split('_')[0]
        table[oversub[i]][channel] = '{},{},{}'.format(round(improvement_accpt,1),round(improvement_cpu,1),round(improvement_mem,1))
        
    df = pd.DataFrame(table)
    df = df.reindex(index = ['8.0','16.0','32.0'])

    table_tex = df.to_latex(index=True).split('\n')

    tex_final = []
    for i in range(len(table_tex)):
        line = table_tex[i]
        line = line.replace(',',', ')
        if 'tabular' in line:
            line = line.replace('tabular','longtable')
            line = line.replace('lllll','llllllllllllll')
            tex_final.append(line)
            if i == 0:
#                 tex_final.append('\multicolumn{14}{c}{Agent improvement over second best baseline} \\\\')
                tex_final.append('\\midrule')
                tex_final.append('\multicolumn{14}{c}{\\textbf{Oversubscription}} \\\\')
        elif i == 2:
            line = '{} & ' + line
            line = line.replace('0.0625','\multicolumn{3}{c}{1:16}')
            line = line.replace('0.1250','\multicolumn{3}{c}{1:8}')
            line = line.replace('0.2500','\multicolumn{3}{c}{1:4}')
            line = line.replace('1.0000','\multicolumn{3}{c}{1:1}')
            tex_final.append(line)
            tex_final.append('{} & {} & \multicolumn{3}{c}{accpt, cpu, mem} & \multicolumn{3}{c}{accpt, cpu, mem} & \multicolumn{3}{c}{accpt, cpu, mem} & \multicolumn{3}{c}{accpt, cpu, mem} \\\\')
        elif i == 4:
            split_line = line.split(' & ')
            for i in range(1,len(split_line)):
                values = split_line[i]
                if values[-2:] == '\\\\':
                    split_line[i] = ' & \multicolumn{{3}}{{c}}{{{}}} \\\\'.format(values[:-4])
                else:
                    split_line[i] = ' & \multicolumn{{3}}{{c}}{{{}}}'.format(values)
            line = ''.join(split_line)
            line = ' \\textbf{Tier-1} & ' + line
            tex_final.append(line)
        elif i == 5:
            split_line = line.split(' & ')
            for i in range(1,len(split_line)):
                values = split_line[i]
                if values[-2:] == '\\\\':
                    split_line[i] = ' & \multicolumn{{3}}{{c}}{{{}}} \\\\'.format(values[:-4])
                else:
                    split_line[i] = ' & \multicolumn{{3}}{{c}}{{{}}} '.format(values)
            line = ''.join(split_line)
            line = ' \\textbf{channels} & ' + line
            tex_final.append(line)
        elif i == 6:
            split_line = line.split(' & ')
            for i in range(1,len(split_line)):
                values = split_line[i]
                if values[-2:] == '\\\\':
                    split_line[i] = ' & \multicolumn{{3}}{{c}}{{{}}} \\\\'.format(values[:-4])
                else:
                    split_line[i] = ' & \multicolumn{{3}}{{c}}{{{}}}'.format(values)
            line = ''.join(split_line)
            line = ' \\textbf{per-link} & ' + line
            tex_final.append(line)
        else:
            tex_final.append(line)

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
                             hist = True, 
                             kde = False, 
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
            if oversub[i] == 0.0625:
                ovs = '1:16'
            elif oversub[i] == 0.125:
                ovs = '1:8'
            elif oversub[i] == 0.25:
                ovs = '1:4'
            else:
                ovs = '1:1'
            name = '{}\n{}'.format(int(value[0]),ovs)
            reshaped_topology_names.append(name)
            indices.append(i)

#     for metric in ['accpt','cpu','mem','sr','ra','ac','nodes']:
    for metric in ['accpt','cpu','mem','sr','ra','ac','ratio']:
        plt.figure(figsize=(5,3))
#         plt.xlabel('low-tier channels-per-link\n(oversubscription)')
        if metric == 'accpt':
            plt.ylim(0,1.1)
            plt.ylabel('accpt.',fontsize=15)
            plt.xlabel('f\nf',fontsize=18,color='w')
        elif metric == 'cpu':
            plt.ylim(0,1.1)
            plt.ylabel('cpu util.',fontsize=15)
            plt.xlabel('low-tier channels-per-link\n(oversubscription)',fontsize=18)
        elif metric == 'mem':
            plt.ylim(0,1.1)
            plt.ylabel('memory util.',fontsize=15)
            plt.xlabel('f\nf',fontsize=18,color='w')
        elif metric == 'sr':
            plt.ylabel('tier-1 util.',fontsize=15)
            plt.ylim(0,1)
        elif metric == 'ra':
            plt.ylim(0,1.1)
            plt.ylabel('tier-2 util.',fontsize=15)
            plt.xlabel('low-tier channels-per-link\n(oversubscription)',fontsize=18)
        elif metric == 'ac':
            plt.ylim(0,1.1)
            plt.ylabel('tier-3 util.',fontsize=15)
        elif metric == 'ratio':
            plt.ylim(0,3.1)
            plt.ylabel('spread',fontsize=15)
            plt.xlabel('low-tier channels-per-link\n(oversubscription)',fontsize=18)
            
#         plt.title('topology vs {}'.format(metric))
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
            plt.xticks(fontsize=12)
#         if metric == 'mem':
#             plt.legend(ncol=5,loc='lower center')
        plt.savefig('./figs/{}_line.png'.format(metric),bbox_inches='tight')


        
# def get_success_distribution(path,topologies,oversub,oversub_selector=None,channels_selector=None):
    
#     reshaped_topology_names = []
#     indices = []
#     for i in range(len(topologies)):
#         tp = topologies[i]
#         value = [float(x) for x in tp.split('_')]
#         if oversub_selector is not None and oversub[i] != oversub_selector:
#             continue
#         elif channels_selector is not None and value[0] != channels_selector:
#             continue
#         else:
#             indices.append(i)
            
# #     plt.figure()
#     for packer in ['agent','tetris','nalb','nulb','random']:
#         successes = []
#         failures = []
#         for index in indices:
#             new_path = '{}/{}/{}'.format(path,packer,topologies[index])
#             successes += _success_nodes(new_path,n=5)
#             failures += _failure_nodes(new_path,n=5)
        
#         fig = plt.figure(figsize=(18,5))
#         ax = fig.add_subplot(121)
#         plt.subplots_adjust(wspace=None, hspace=None)
#         pos = ax.get_position()
# #         ax.set_position([pos.x0-3, pos.y0-3, pos.width, pos.height])

#         plt.ylim(0,500)

#         ###
#         plt.title(packer,fontsize=40)
#         sns.distplot(successes, 
#             hist = True, 
#             kde = False, 
#             rug=False,
# #             bins=np.arange(0,1.05,0.05),
#             kde_kws = {'linewidth': 3},
#             hist_kws={'histtype':'stepfilled','linewidth':4,'alpha':1.0,'color':'g'},#'edgecolor':'k','linewidth':4,'alpha':0.1},
#             label='successes')
        
# #         plt.title('failures')
#         sns.distplot(failures, 
#             hist = True, 
#             kde = False, 
#             rug=False,
# #             bins=np.arange(0,1.05,0.05),
#             kde_kws = {'linewidth': 3},
#             hist_kws={'histtype':'stepfilled','linewidth':4,'alpha':1.0,'color':'r'},#'edgecolor':'k','linewidth':4,'alpha':0.1},
#             label='failures')

# #         plt.legend(ncol=2,fontsize=17)
#         plt.xticks([0,250],fontsize=22)
#         plt.yticks([0,500],fontsize=22)
        
#         ###    
#         in_ax = inset_axes(ax, 
#                             width="50%", # width = 30% of parent_bbox
#                             height=1.0, # height : 1 inch
#                             loc=1)
#         sns.distplot(successes, 
#             hist = True, 
#             kde = False, 
#             rug=False,
# #             bins=np.arange(0,1.05,0.05),
#             kde_kws = {'linewidth': 3},
#             hist_kws={'histtype':'stepfilled','linewidth':4,'alpha':1.0,'color':'g'},#'edgecolor':'k','linewidth':4,'alpha':0.1},
#             label='successes')
        
# #         plt.title('failures')
#         sns.distplot(failures, 
#             hist = True, 
#             kde = False, 
#             rug=False,
# #             bins=np.arange(0,1.05,0.05),
#             kde_kws = {'linewidth': 3},
#             hist_kws={'histtype':'stepfilled','linewidth':4,'alpha':1.0,'color':'r'},#'edgecolor':'k','linewidth':4,'alpha':0.1},
#             label='failures')
#         plt.xticks([200,240],fontsize=15)
#         plt.yticks([0,300],fontsize=15)
#         plt.title('upper 25% of request sizes',y=0.75,fontsize=15)
#         plt.xlim(192,256)
#         plt.ylim(0,300)
#         plt.tight_layout()
#         ###
#         plt.tight_layout()
#         plt.savefig('./figs/success_failure_{}.png'.format(packer),set_bbox_inches='tight')

def get_ratio_distributions(oversub,topologies,packer_stats):
    
    all_oversubs = np.array(list(set(oversub)))
#     all_oversubs = [all_oversubs[0],all_oversubs[-1]]
    values = {'agent':{0.0625:[],0.125:[],0.25:[],1.0:[]},'tetris':{0.0625:[],0.125:[],0.25:[],1.0:[]},'nalb':{0.0625:[],0.125:[],0.25:[],1.0:[]},'nulb':{0.0625:[],0.125:[],0.25:[],1.0:[]},'random':{0.0625:[],0.125:[],0.25:[],1.0:[]}}
    for i in range(len(all_oversubs)):
        indices = np.argwhere(oversub==all_oversubs[i])
        tops = np.array(topologies)[indices]
        
        for packer in list(packer_stats.keys()):
            for j in range(len(tops)):

                topology = tops[j][0]
                values[packer][all_oversubs[i]] += _ratio_full('/home/uceezs0/Code/nara_data/uniform/baselines/{}/{}'.format(packer,topology),n=5)
    plt.figure()
    for packer in list(packer_stats.keys()):
        oversub_0,bins = np.histogram(values[packer][0.0625],bins=[0,1,2,4])
        oversub_1,bins = np.histogram(values[packer][0.125],bins=[0,1,2,4])
        oversub_2,bins = np.histogram(values[packer][0.25],bins=[0,1,2,4])
        oversub_3,bins = np.histogram(values[packer][1.0],bins=[0,1,2,4])
        print(packer)
#         print(oversub_0)
#         print(oversub_1)
#         print(oversub_2)
#         print(oversub_3)
        print(np.std([oversub_0[0],oversub_1[0],oversub_2[0],oversub_3[0]])/np.mean([oversub_0[0],oversub_1[0],oversub_2[0],oversub_3[0]]))
        print(np.std([oversub_0[1],oversub_1[1],oversub_2[1],oversub_3[1]])/np.mean([oversub_0[1],oversub_1[1],oversub_2[1],oversub_3[1]]))
        print(np.std([oversub_0[2],oversub_1[2],oversub_2[2],oversub_3[2]])/np.mean([oversub_0[2],oversub_1[2],oversub_2[2],oversub_3[2]]))
        print('\n')
#         sns.distplot(high_oversub/low_oversub, bins=bins,
#                     hist = True,
#                     kde = False, 
#                     rug=False,
#                     kde_kws = {'linewidth': 3})
                
                    
#         plt.xlim(0,4)
#         plt.title('oversubscription: {}'.format(oversub[i]),fontsize=25)
#         plt.xticks([0.0,1.0,2.0,3.0,4.0],fontsize=25)
#         plt.yticks([0.0,1.0,2.0,3.0,4.0],fontsize=25)
#         if oversub[i] == 0.0625:
#             plt.xlabel("'distribution' value",fontsize=25)
#             plt.ylabel('probability density',fontsize=25)
#         else:
#             plt.xlabel("'distribution' value",fontsize=25,color='w')
#             plt.ylabel('probability density',fontsize=25,color='w')
#         plt.tight_layout()
#         plt.savefig('./figs/distribution_{}.png'.format(topology))
    
# def get_ratio_distributions(oversub,topologies,packer_stats):
    
#     all_oversubs = np.array(list(set(oversub)))
    
#     for i in range(len(topologies)):
#         topology = topologies[i]
#         plt.figure()
# #         plt.title('Topology: {}'.format(topology))
#         for packer in list(packer_stats.keys()):
            
#             values = _ratio_full('/home/uceezs0/Code/nara_data/uniform/baselines/{}/{}'.format(packer,topology),n=5)
            
#             sns.distplot(values, 
#             hist = False,
#             kde = True, 
#             rug=False,
# #             bins=np.arange(0,1.05,0.05),
#             kde_kws = {'linewidth': 3})
#         plt.xlim(0,4)
#         plt.title('oversubscription: {}'.format(oversub[i]),fontsize=25)
#         plt.xticks([0.0,1.0,2.0,3.0,4.0],fontsize=25)
#         plt.yticks([0.0,1.0,2.0,3.0,4.0],fontsize=25)
#         if oversub[i] == 0.0625:
#             plt.xlabel("'distribution' value",fontsize=25)
#             plt.ylabel('probability density',fontsize=25)
#         else:
#             plt.xlabel("'distribution' value",fontsize=25,color='w')
#             plt.ylabel('probability density',fontsize=25,color='w')
# #         plt.legend()
#         plt.tight_layout()
#         plt.savefig('./figs/distribution_{}.png'.format(topology))
        
def get_ratio_probabilities(oversub,topologies,packer_stats):
    
    packs = {'agent':0,'tetris':0,'nalb':0,'nulb':0,'random':0}
    probs = {0.0625:packs.copy(),0.125:packs.copy(),0.25:packs.copy(),1.0:packs.copy()}
    oversub = np.array(list(oversub))
    all_oversubs = np.array(list(set(oversub)))
    for ovs in all_oversubs:
        plt.figure()
        plt.title('oversubscription: {}'.format(ovs),fontsize=25)
        
        plt.xlim(0,4)
        plt.xticks([0.0,1.0,2.0,3.0,4.0],fontsize=25)
        plt.yticks([0.0,1.0,2.0,3.0,4.0],fontsize=25)
        if ovs == 0.0625:
            plt.xlabel("'spread' value",fontsize=25)
            plt.ylabel('probability density',fontsize=25)
        else:
            plt.xlabel("'spread' value",fontsize=25,color='w')
            plt.ylabel('probability density',fontsize=25,color='w')
            
            
        print('\n')
        print(ovs)
        indices = np.argwhere(oversub==float(ovs))
        tops = np.array(topologies)[indices]
        for packer in packer_stats.keys():
            print(packer)
            values = []
            for topology in tops:
                topology = topology[0]
                values +=_ratio_full('/home/uceezs0/Code/nara_data/uniform/baselines/{}/{}'.format(packer,topology),n=5)
            bins,limits = np.histogram(values,bins=[0.0,1.0,2.0,4.0])
            bins = bins/len(values)
            print(bins)
            
            sns.distplot(values, 
            hist = False,
            kde = True, 
            rug=False,
            bins=np.arange(0,1.05,0.05),
            kde_kws = {'linewidth': 3})
        plt.tight_layout()
        plt.savefig('./figs/distribution_{}.png'.format(ovs))
            
            
def get_success_distribution(path,topologies,oversub,oversub_selector=None,channels_selector=None):
    
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
            indices.append(i)
            
    plt.figure(figsize=(12,3))
    for i in range(len(['agent','tetris','nalb','nulb','random'])):
        packer = ['agent','tetris','nalb','nulb','random'][i]
        successes = []
        failures = []
        for index in indices:
            new_path = '{}/{}/{}'.format(path,packer,topologies[index])
            successes += _success_nodes(new_path,n=5)
            failures += _failure_nodes(new_path,n=5)
        
        ticks = bins = np.arange(-32,264,32)
        bins = ticks-25+(i*5)
        successes,limits = np.histogram(successes,bins=ticks)
        failures,limits = np.histogram(failures,bins=ticks)
        total = successes - failures
        print(total)
        plt.bar(bins[1:],total,label=packer,width=5)
        plt.xticks(ticks,fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(ticks[1],ticks[-1])

    plt.xlabel('Request size (CPU + Memory)',fontsize=18)
    plt.ylabel('|accepted| - |rejected|',fontsize=18)
    plt.tight_layout()
    plt.savefig('./figs/success_failure.png',set_bbox_inches='tight')