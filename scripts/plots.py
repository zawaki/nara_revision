import matplotlib.pyplot as plt
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import time
import csv
import os
import pandas as pd
from NetworkSimulator import Network
import networkx as nx
import os
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

def csv2numpy(directory_name,file_root):

    files = [x for x in os.listdir(directory_name) if x.split('_')[0] == file_root]
    return np.array([np.genfromtxt('{}/{}'.format(directory_name,file),skip_header=1,delimiter=',') for file in files])


def servers_size_bar(directory_name):

    fig = plt.figure()
    plt.rcParams.update({'font.size': 15})
    all_dists = {'agent':[],
                'tetris':[],
                'random':[],
                'nalb':[],
                'nulb':[]}

    for d in all_dists.keys():

        success = csv2numpy('{}/{}'.format(directory_name,d),'success')
        for suc in success:
            all_dists[d] += [suc[i][5] for i in range(suc.shape[0])]

    bins = [0,6,11,16,20]
    for d in all_dists.keys():
        all_dists[d] = np.histogram(all_dists[d],bins=bins)
    
    all_values = {'0-5':[],
                    '6-10':[],
                    '11-15':[],
                    '16-20':[]}
    for d in all_dists.keys():
        for i in range(len(all_values.keys())):
            c = list(all_values.keys())[i]
            all_values[c].append(all_dists[d][0][i]/np.sum(all_dists[d][0]))

    plt.bar(all_dists.keys(),all_values['0-5'],label='0-5')
    plt.bar(all_dists.keys(),all_values['6-10'],bottom=all_values['0-5'],label='6-10')
    plt.bar(all_dists.keys(),all_values['11-15'],bottom=np.array(all_values['0-5'])+np.array(all_values['6-10']),label='11-15')
    plt.bar(all_dists.keys(),all_values['16-20'],bottom=np.array(all_values['0-5'])+np.array(all_values['6-10'])+np.array(all_values['11-15']),label='16-20')

    plt.ylabel('Number of servers used in allocation')
    plt.ylim(0,1.1)

    plt.legend(ncol=4,loc='upper center',bbox_to_anchor=(0.5, 1.15),prop={'size':13})

    print('NUMBER OF SERVERS')
    print(all_values)
    print('\n')

def get_util(directory_name):

    print(directory_name)
    all_dists = {'agent':[],
                'tetris':[],
                'random':[],
                'nalb':[],
                'nulb':[]}
    for c in ['cpu','mem','sr','ra','ac']:
        for dist in all_dists.keys():
            cpu = csv2numpy('{}/{}_features'.format(directory_name,dist),c)
            size = cpu.shape[1]*cpu.shape[2]
            cpu = np.reshape(cpu,(cpu.shape[0],size))
            all_cpu = []
            limit = int((4/5)*size)
            for i in range(len(cpu)):
                all_cpu.append(list(cpu[i][limit:]))
            all_dists[dist].append(1 - np.mean(all_cpu))
    
    return all_dists
    

def util_bars(directory_name):

    all_dists = get_util(directory_name)

    categories = ['cpu','mem','server-\nrack b/w','rack-\nfabric b/w','fabric-\nspine b/w']
    x = np.arange(len(categories))
    width = 0.35

    all_values = {}
    for i in range(len(categories)):
        cat = categories[i]
        all_values[cat] = []
        for d in all_dists.keys():
            all_values[cat].append(all_dists[d][i])

    bar_width = width/2
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    agnt = ax.bar(x-(1.0*width), all_dists['agent'], bar_width, label='agent')
    ttrs = ax.bar(x-(0.5*width), all_dists['tetris'], bar_width, label='tetris')
    rndm = ax.bar(x, all_dists['random'], bar_width, label='random')
    na = ax.bar(x+(0.5*width), all_dists['nalb'], bar_width, label='nalb')
    nu = ax.bar(x+(1.0*width), all_dists['nulb'], bar_width, label='nulb')

    ax.set_ylabel('Utilisation (normalised)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    plt.legend(ncol=3,loc='upper center',bbox_to_anchor=(0.5, 1.25),prop={'size':13},framealpha=1.0)
    plt.tight_layout()
    print('UTILISATION')
    print(all_dists)
    print('\n')

def distribution_bars(directory_name):
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()

    all_dists = {'agent':[],
                'tetris':[],
                'random':[],
                'nalb':[],
                'nulb':[]}

    for d in all_dists.keys():

        success = csv2numpy('{}/{}'.format(directory_name,d),'success')
        for suc in success:
            all_dists[d] += [suc[i][4] for i in range(suc.shape[0])]
    
    bins = [0.0,1.0,2.0,4.0]
    for d in all_dists.keys():
        all_dists[d] = np.histogram(all_dists[d],bins=bins)
    
    all_values = {'intra-server':[],
                    'intra-rack':[],
                    'inter-rack':[]}
    for d in all_dists.keys():
        for i in range(len(all_values.keys())):
            c = list(all_values.keys())[i]
            all_values[c].append(all_dists[d][0][i]/np.sum(all_dists[d][0]))

    plt.bar(all_dists.keys(),all_values['intra-server'],label='intra-server')
    plt.bar(all_dists.keys(),all_values['intra-rack'],bottom=all_values['intra-server'],label='intra-rack')
    plt.bar(all_dists.keys(),all_values['inter-rack'],bottom=np.array(all_values['intra-server'])+np.array(all_values['intra-rack']),label='inter-rack')

    plt.ylabel('Number of allocations (normalised)')
    plt.ylim(0,1.1)

    plt.legend(ncol=3,loc='upper center',bbox_to_anchor=(0.5, 1.15),prop={'size':13})

    print('DISTRIBUTION')
    print(all_values)
    print('\n')

def failure_histogram(directory_name,name=None,data=None):

    failures = csv2numpy(directory_name,'failure')

    labels = ['Compute','Memory','Bandwidth']#,'Holding-Time']
    fails = []

    for i in range(len(labels)):
        resource = np.array([])
        for j in range(len(failures)):
            tmp = failures[j]
            tmp = [tmp[k][i] for k in range(tmp.shape[0])]
            maximum = np.max(tmp)
            tmp = np.array(tmp)
            resource = np.concatenate((resource,tmp),axis=0)
        
        plt.figure()
        plt.xlabel('Request Quantity')
        plt.ylabel('Count')
        plt.ylim(0,100)
        
        bins = []
        
        if name is not None:
            plt.title('{} failures - {}'.format(labels[i],name))
        if labels[i] == 'Bandwidth':
            bins = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        else:
            if data == 'uniform':
                bins = [0,10,20,30,40,50]
            else:
                bins = [0,10,20,30,40,50,60,70,80,90,100]
            
        sns.distplot(resource, hist = True, kde = False, rug=False, bins=bins,
                    kde_kws = {'linewidth': 3},
                    label=labels[i])

def ratio_distribution(directory_name,name=None):

    success = csv2numpy(directory_name,'success')

    # plt.figure()
    r = []

    for suc in success:
        r_tmp = []
        for i in range(len(suc)):
            r_tmp.append(suc[i][4])

        r_tmp = np.array(r_tmp)
        r += list(r_tmp)


    sns.distplot(r, hist = False, kde = True, rug=True,
            kde_kws = {'linewidth': 3},
            label=name)

    plt.xlabel('Distribution of Request')
    plt.ylabel('Probability Density')
    
def ratio_histogram(directory_name,name=None):

    success = csv2numpy(directory_name,'success')

    # plt.figure()
    r = []

    for suc in success:
        r_tmp = []
        for i in range(len(suc)):
            r_tmp.append(suc[i][4])

        r_tmp = np.array(r_tmp)
        r += list(r_tmp)

    plt.figure()
    sns.distplot(r, hist = True, kde = False, rug=False, bins=[0.0,1.0,1.5,2.0,2.5,3.0,4.0],
            kde_kws = {'linewidth': 3},
            label=None)
    plt.ylim(0,400)
    plt.title('{}'.format(name))
    plt.xlabel('Distribution of Request (bin)')
    plt.ylabel('Count')
    


def ratio_distribution_slice(directory_name,low,high,name=None):

    success = csv2numpy(directory_name,'success')

    plt.figure()
    r = []

    for suc in success:
        r_tmp = []
        for i in range(low,high):
            r_tmp.append(suc[i][4])

        r_tmp = np.array(r_tmp)/np.max(r_tmp)
        r += list(r_tmp)


    sns.distplot(r, hist = False, kde = True, rug=True,
            kde_kws = {'linewidth': 3},
            label=name)

    plt.xlabel('Distribution of Request')
    plt.ylabel('Probability Density')
    
def failure_3d(directory_name,name=None,save=None):

    success = csv2numpy(directory_name,'failure')

    cpu = []
    mem = []
    net = []
    t = []

    for suc in success:
        cpu_tmp = []
        mem_tmp = []
        net_tmp = []
        t_tmp = []

        for i in range(len(suc)):

            cpu_tmp.append(suc[i][0])
            mem_tmp.append(suc[i][1])
            net_tmp.append(suc[i][2])
            t_tmp.append(suc[i][3])

        cpu += cpu_tmp
        mem += mem_tmp
        net += net_tmp
        t_tmp = np.array(t_tmp)
        # r_tmp /= 3
        t += list(t_tmp)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    c = ax.scatter(cpu, mem, net, c=t, cmap=cm.jet)
    ax.set_xlabel('CPU')
    ax.set_ylabel('Memory')
    ax.set_zlabel('Bandwidth')

    if name is not None:
        ax.set_title('{} \n Number of accepted requests = {}'.format(name,len(t)))
    else:
        ax.set_title('Number of failed requests = {}'.format(len(t_tmp)))

    cbar = fig.colorbar(c,orientation='horizontal')
    cbar.set_label('Holding Time')


    plt.tight_layout()

def resources_ratio_3d(directory_name,name=None,save=None):

    success = csv2numpy(directory_name,'success')

    cpu = []
    mem = []
    net = []
    r = []

    for suc in success:
        cpu_tmp = []
        mem_tmp = []
        net_tmp = []
        r_tmp = []

        for i in range(len(suc)):

            cpu_tmp.append(suc[i][0])
            mem_tmp.append(suc[i][1])
            net_tmp.append(suc[i][2])
            r_tmp.append(suc[i][4])

        cpu += cpu_tmp
        mem += mem_tmp
        net += net_tmp
        r_tmp = np.array(r_tmp)
        # r_tmp /= 3
        r += list(r_tmp)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(5,3.7))
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)
    plt.rcParams['axes.labelpad'] = 10
    c = ax.scatter(cpu, mem, net, c=r, cmap=cm.jet,vmin=0,vmax=4)
    ax.set_xlabel('CPU')
    ax.set_ylabel('Memory')
    ax.set_zlabel('Bandwidth')

    if name is not None:
        ax.set_title('{} \n Number of accepted requests = {}'.format(name,len(r)))
    else:
        ax.set_title('Number of accepted requests = {}'.format(len(r_tmp)))

    cbar = fig.colorbar(c,orientation='horizontal')
    cbar.set_label('Distribution of Requests')


    plt.tight_layout()


def resources_nodes_3d(directory_name,name=None):

    success = csv2numpy(directory_name,'success')

    cpu = []
    mem = []
    net = []
    nodes = []

    for suc in success:
        cpu_tmp = []
        mem_tmp = []
        net_tmp = []
        nodes_tmp = []

        for i in range(len(suc)):

            cpu_tmp.append(suc[i][0])
            mem_tmp.append(suc[i][1])
            net_tmp.append(suc[i][2])
            nodes_tmp.append(suc[i][5])

        cpu += cpu_tmp
        mem += mem_tmp
        net += net_tmp
        nodes += nodes_tmp

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    c = ax.scatter(cpu, mem, net, c=nodes, cmap=cm.gnuplot,vmin=0,vmax=25)
    ax.set_xlabel('CPU')
    ax.set_ylabel('Memory')
    ax.set_zlabel('Bandwidth')

    if name is not None:
        ax.set_title('{} \n Number of accepted requests = {}'.format(name,len(nodes)))
    else:
        ax.set_title('Number of accepted requests = {}'.format(len(nodes)))

    cbar = fig.colorbar(c,orientation='horizontal')
    cbar.set_label('Number of Allocated Servers')

    plt.tight_layout()

def resources_nodes_ratio_3d(directory_name,name=None):

    success = csv2numpy(directory_name,'success')

    cpu = []
    mem = []
    nodes = []
    r = []

    for suc in success:
        cpu_tmp = []
        mem_tmp = []
        nodes_tmp = []
        r_tmp = []

        for i in range(len(suc)):

            cpu_tmp.append(suc[i][0])
            mem_tmp.append(suc[i][1])
            nodes_tmp.append(suc[i][5])
            r_tmp.append(suc[i][4])

        cpu += cpu_tmp
        mem += mem_tmp
        nodes += nodes_tmp
        r_tmp = np.array(r_tmp)
        # r_tmp /= 3
        r += list(r_tmp)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    c = ax.scatter(cpu, mem, nodes, c=r, cmap=cm.jet,vmin=0,vmax=4)
    ax.set_xlabel('CPU')
    ax.set_ylabel('Memory')
    ax.set_zlabel('Servers Allocated')
    ax.set_zlim(0,25)

    if name is not None:
        ax.set_title('{} \n Number of accepted requests = {}'.format(name,len(r)))
    else:
        ax.set_title('Number of accepted requests = {}'.format(len(r)))

    cbar = fig.colorbar(c,orientation='horizontal')
    cbar.set_label('Distribution of Requests')


    plt.tight_layout()

def resources_nodes_2d(directory_name,name=None):

    success = csv2numpy(directory_name,'success')

    cpu = []
    mem = []
    nodes = []

    for suc in success:
        cpu_tmp = []
        mem_tmp = []
        nodes_tmp = []

        for i in range(len(suc)):

            cpu_tmp.append(suc[i][0])
            mem_tmp.append(suc[i][1])
            nodes_tmp.append(suc[i][5])

        cpu += cpu_tmp
        mem += mem_tmp
        nodes += nodes_tmp

    fig, ax = plt.subplots()

    c = ax.scatter(cpu, mem, c=nodes, cmap=cm.gnuplot,vmin=0,vmax=25)
    ax.set_xlabel('CPU')
    ax.set_ylabel('Memory')

    if name is not None:
        ax.set_title('{} \n Number of accepted requests = {}'.format(name,len(nodes)))
    else:
        ax.set_title('Number of accepted requests = {}'.format(len(nodes)))

    cbar = fig.colorbar(c,orientation='horizontal')
    cbar.set_label('Number of Allocated Servers')


    plt.tight_layout()

def resources_ratio_2d(directory_name,name=None):

    success = csv2numpy(directory_name,'success')

    cpu = []
    mem = []
    r = []

    for suc in success:
        cpu_tmp = []
        mem_tmp = []
        r_tmp = []

        for i in range(len(suc)):

            cpu_tmp.append(suc[i][0])
            mem_tmp.append(suc[i][1])
            r_tmp.append(suc[i][4])

        cpu += cpu_tmp
        mem += mem_tmp
        r_tmp = np.array(r_tmp)
        # r_tmp /= 3.0
        r += list(r_tmp)

    fig, ax = plt.subplots()

    c = ax.scatter(cpu, mem, c=r, cmap=cm.jet,vmin=0,vmax=4)
    ax.set_xlabel('CPU')
    ax.set_ylabel('Memory')


    if name is not None:
        ax.set_title('{} \n Number of accepted requests = {}'.format(name,len(r)))
    else:
        ax.set_title('Number of accepted requests = {}'.format(len(r)))

    cbar = fig.colorbar(c,orientation='horizontal')
    cbar.set_label('Distribution of Requests')

    plt.tight_layout()

def nodes_ratio_2d(directory_name,name=None):

    success = csv2numpy(directory_name,'success')

    nodes = []
    r = []

    for suc in success:
        nodes_tmp = []
        r_tmp = []

        for i in range(len(suc)):

            nodes_tmp.append(suc[i][5])
            r_tmp.append(suc[i][4])

        nodes += nodes_tmp
        r_tmp = np.array(r_tmp)
        # r_tmp /= 3
        r += list(r_tmp)

    c = Counter(zip(nodes,r))
    # create a list of the sizes, here multiplied by 10 for scale
    s = [10*c[(x,y)] for x,y in zip(nodes,r)]

    fig, ax = plt.subplots(figsize=(5,3.2))

    c = ax.scatter(nodes, r, c='blue', s=s,alpha=0.5)
    ax.set_xlabel('Number of Allocated Servers')
    ax.set_ylabel('Distribution')
    ax.set_xlim(0,25)
    ax.set_ylim(0,4)

    if name is not None:
        ax.set_title('{}'.format(name,))
    plt.tight_layout()



def load_comparison(directory_name):

    subdirs = os.listdir(directory_name)
    plt.figure()
    values = {}

    for subdir in subdirs:
        if subdir == '.ipynb_checkpoints':
            continue
        label = subdir.split('_')[0]
        if label not in values.keys():
            values[label] = ([],[])
        i = subdir.split('_')[-1]
        acceptance = csv2numpy('{}/{}'.format(directory_name,subdir),'acceptance')
        req_nums_list = np.arange(len(acceptance[0]))
    
        mean = np.mean(acceptance,axis=0)
        mean = np.mean(mean[int((4*len(mean))/5):])

        values[label][0].append(int(i))
        values[label][1].append(mean)

    for k in values.keys():
        plt.plot(values[k][0],values[k][1],'x-',label=k)
    
    plt.legend()
    plt.xlabel('% of average load seen during training')
    plt.ylabel('Avg. acceptance ratio')

def load_comparison_cpu(directory_name):

    subdirs = os.listdir(directory_name)
    plt.figure()
    values = {}

    for subdir in subdirs:
        if subdir == '.ipynb_checkpoints':
            continue
        label = subdir.split('_')[0]
        if label not in values.keys():
            values[label] = ([],[])
        i = subdir.split('_')[-1]
        util = csv2numpy('{}/{}'.format(directory_name,subdir),'util')

        mean = np.mean(util,axis=0)
        req_nums_list = np.arange(mean.shape[0])
        mean = [mean[i][0] for i in range(len(req_nums_list))]
        mean = np.mean(mean[int((5*len(mean))/10):])

        values[label][0].append(int(i))
        values[label][1].append(mean)

    for k in values.keys():
        plt.plot(values[k][0],values[k][1],'x-',label=k)

    # plt.plot([10,25,50,75],[0.1,0.25,0.5,0.75],'k--',label='Perfect utilisation')
    
    plt.legend()
    plt.xlabel('% of average load seen during training')
    plt.ylabel('Avg. CPU utilisation')

def load_comparison_mem(directory_name):

    subdirs = os.listdir(directory_name)
    plt.figure()
    values = {}

    for subdir in subdirs:
        if subdir == '.ipynb_checkpoints':
            continue
        label = subdir.split('_')[0]
        if label not in values.keys():
            values[label] = ([],[])
        i = subdir.split('_')[-1]
        util = csv2numpy('{}/{}'.format(directory_name,subdir),'util')

        mean = np.mean(util,axis=0)
        req_nums_list = np.arange(mean.shape[0])
        mean = [mean[i][1] for i in range(len(req_nums_list))]
        mean = np.mean(mean[int((5*len(mean))/10):])

        values[label][0].append(int(i))
        values[label][1].append(mean)

    for k in values.keys():
        plt.plot(values[k][0],values[k][1],'x-',label=k)
    # plt.plot([10,25,50,75],[0.1,0.25,0.5,0.75],'k--',label='Perfect utilisation')
    
    plt.legend()
    plt.xlabel('% of average load seen during training')
    plt.ylabel('Avg. Memory utilisation')


def acceptance_ratio(directory_name):

    acceptance = csv2numpy(directory_name,'acceptance')
    req_nums_list = np.arange(len(acceptance[0]))

    mean = np.mean(acceptance,axis=0)
    var = np.std(acceptance,axis= 0)

    ci_acceptance = []
    for acpt in acceptance:
        ci_acceptance += list(acpt[int((4*len(mean))/5):])
    ci_std = np.std(ci_acceptance)
    ci_mean = np.mean(mean[int((4*len(mean))/5):])
#     print(ci_mean,np.sqrt(mean.shape[0]))
    print((1.96*(ci_std/np.sqrt(mean.shape[0])))/ci_mean)

    plt.figure()
    plt.xlabel('Request')
    plt.ylabel('Acceptance Ratio')
    plt.ylim(0,1.1)
    # plt.title('Mean Acceptance Ratio per Request in Episode \n (± Std-Dev Error bars)')
    plt.errorbar(req_nums_list,mean,c='black',yerr=var,ecolor='r',elinewidth=2,capsize=3)
    plt.hlines(np.mean(mean[int((4*len(mean))/5):]),0,len(mean),colors='blue',linestyles='dashed',label='Convergence Mean: {}'.format(round(np.mean(mean[int((4*len(mean))/5):]),2)))
    plt.legend()

def util(directory_name):

    util = csv2numpy(directory_name,'util')
    mean = np.mean(util,axis=0)
    var = np.var(util,axis=0)
    req_nums_list = np.arange(mean.shape[0])

    compute_mean = [mean[i][0] for i in range(len(req_nums_list))]
    memory_mean = [mean[i][1] for i in range(len(req_nums_list))]
    bandwidth_mean = [mean[i][2] for i in range(len(req_nums_list))]
    
    for j in range(2):
        ci_util = []
        for utl in util:
            ci_util += [utl[i][j] for i in range(int((5*len(compute_mean))/10),len(compute_mean))]
        ci_std = np.std(ci_util)
        ci_mean = np.mean(ci_util)
        print((1.96*(ci_std/np.sqrt(len(ci_util))))/ci_mean)
    
    
    
    plt.figure()
    # plt.title('Mean and Variance of Resource Utilisation w.r.t. Sequential Requests')
    
    plt.xlabel('Request')
    plt.ylabel('Mean')
    plt.ylim(0,1.1)
    
    plt.plot(compute_mean,color='blue',label='CPU utilisation: {}'.format(round(np.mean(compute_mean[int((5*len(compute_mean))/10):]),2)))
    # plt.hlines(np.mean(mean[int((5*len(compute_mean))/10):]),0,len(compute_mean),colors='blue',linestyles='dashed',label='CPU utilisation mean: {}'.format(round(np.mean(compute_mean[int((5*len(compute_mean))/10):]),2)))
    
    plt.plot(memory_mean,color='red',label='Memory utilisation: {}'.format(round(np.mean(memory_mean[int((5*len(memory_mean))/10):]),2)))
    # plt.hlines(np.mean(memory_mean[int((5*len(memory_mean))/10):]),0,len(memory_mean),colors='red',linestyles='dashed',label='Memory utilisation mean: {}'.format(round(np.mean(memory_mean[int((5*len(memory_mean))/10):]),2)))

    plt.plot(bandwidth_mean,color='green',label='Bandwidth utilisation: {}'.format(round(np.mean(bandwidth_mean[int((5*len(bandwidth_mean))/10):]),2)))

    # plt.plot(bandwidth_mean,color='green')#,label='memory')
    # plt.hlines(np.mean(bandwidth_mean[int((5*len(bandwidth_mean))/10):]),0,len(bandwidth_mean),colors='green',linestyles='dashed',label='Bandwidth utilisation mean: {}'.format(round(np.mean(bandwidth_mean[int((5*len(bandwidth_mean))/10):]),2)))
    
    plt.legend()

# def util(directory_name):

#     util = csv2numpy(directory_name,'util')
#     mean = np.mean(util,axis=0)
#     var = np.var(util,axis=0)
#     req_nums_list = np.arange(mean.shape[0])

#     compute_mean = [mean[i][0] for i in range(len(req_nums_list))]
#     memory_mean = [mean[i][1] for i in range(len(req_nums_list))]
#     bandwidth_mean = [mean[i][2] for i in range(len(req_nums_list))]

#     compute_var = [var[i][0] for i in range(len(req_nums_list))]
#     memory_var = [var[i][1] for i in range(len(req_nums_list))]
#     bandwidth_var = [var[i][2] for i in range(len(req_nums_list))]
    
#     fig, (mean_plt, var_plt) = plt.subplots(2, 1)
#     fig.suptitle('Mean and Variance of Resource Utilisation w.r.t. Sequential Requests')
    
#     mean_plt.set_xlabel('Request')
#     mean_plt.set_ylabel('Mean')
#     mean_plt.set_ylim(0,1.1)
    
#     var_plt.set_xlabel('Request')
#     var_plt.set_ylabel('Variance')
    
#     mean_plt.plot(compute_mean,label='compute')
#     var_plt.plot(compute_var,label='memory')
#     mean_plt.plot(memory_mean,label='bandwidth')
#     var_plt.plot(memory_var,label='compute')
#     mean_plt.plot(bandwidth_mean,label='memory')
#     var_plt.plot(bandwidth_var,label='bandwidth')
    
#     plt.legend(ncol=3)

def failure(directory_name):

    failures = csv2numpy(directory_name,'failure')

    labels = ['Compute','Memory','Bandwidth','Holding-Time']
    fails = []

    for i in range(len(labels)):
        resource = np.array([])
        for j in range(len(failures)):
            tmp = failures[j]
            tmp = [tmp[k][i] for k in range(tmp.shape[0])]
            maximum = np.max(tmp)
            tmp = np.array(tmp)
            resource = np.concatenate((resource,tmp),axis=0)
        
        plt.figure()
        plt.title('Failure Probability Density Function vs \n {} Request Quantity'.format(labels[i]))
        plt.xlabel('Request Quantity')
        plt.ylabel('Probability Density')

        sns.distplot(resource, hist = False, kde = True, rug=True,
                    kde_kws = {'linewidth': 3},
                    label=labels[i])

def acceptance_load(directory_names,loads):

    acceptance = []
    for directory in directory_names:
        accpt = csv2numpy(directory,'acceptance')
        tmp = []
        for i in range(len(accpt)):
            accp = accpt[i]
            accp = np.mean(accp[int(len(accp)/2):])
            tmp.append(accp)
        acceptance.append(tmp)

    acceptance = np.array(acceptance)
    mean = np.mean(acceptance,axis=-1)
    var = np.std(acceptance,axis=-1)

    plt.figure()
    plt.xlabel('Average Resource Load (Normalised)')
    plt.ylabel('Acceptance Ratio')
    plt.ylim(0,1.1)
    plt.title('Converged Acceptance Ratio vs Average Resource Load \n (± Std-Dev Error bars)')
    plt.errorbar(loads,mean,c='black',yerr=var,ecolor='r',elinewidth=2,capsize=3)
    plt.legend()

def ratio(directory_name):

    success = csv2numpy(directory_name,'success')
    num_success = 0

    plt.figure()
    # plt.title('Allocated Edge:Allocated Nodes for Accepted Requests')
    plt.xlabel('Server Resource Request')
    plt.ylabel('Bandwidth Resource Request')

    for suc in success:
        res_tmp = []
        net_tmp = []
        r_tmp = []
        num_success += len(suc)

        for i in range(len(suc)):

            res_tmp.append((suc[i][0]+suc[i][1])/2)
            net_tmp.append(suc[i][2])
            r_tmp.append(suc[i][4])

        plt.scatter(res_tmp,net_tmp,c=r_tmp,cmap=cm.seismic,vmin=0, vmax=3)

    cbar = plt.colorbar()
    cbar.set_label('Distribution',rotation=90,x=1)

def fragmentation(directory_name):

    util = csv2numpy(directory_name,'util')
    mean = np.mean(util,axis=0)
    var = np.var(util,axis=0)
    req_nums_list = np.arange(mean.shape[0])

    compute_mean = [mean[i][3] for i in range(len(req_nums_list))]
    memory_mean = [mean[i][4] for i in range(len(req_nums_list))]

    compute_var = [var[i][3] for i in range(len(req_nums_list))]
    memory_var = [var[i][4] for i in range(len(req_nums_list))]
    
    fig, (mean_plt, var_plt) = plt.subplots(2, 1)
    # fig.suptitle('Mean and Variance of Resource Fragmentation w.r.t. Sequential Requests')
    
    mean_plt.set_xlabel('Request')
    mean_plt.set_ylabel('Mean')
    mean_plt.set_ylim(0,1.1)

    
    var_plt.set_xlabel('Request')
    var_plt.set_ylabel('Variance')
    
    mean_plt.plot(compute_mean,label='compute')
    var_plt.plot(compute_var,label='compute')

    mean_plt.plot(memory_mean,label='memory')
    var_plt.plot(memory_var,label='memory')
    
    plt.legend(ncol=2)

def show_policy_state(directory_name,i,topology_dir,size=(5,5)):

    #load networkx graph, then convert to and back from dgl to get features as attributes
    net = Network(os.path.abspath(topology_dir))
    dgl_g = net.to_dgl_with_edges({'node':['node','mem'],'link':['ports']})

    actions = pd.read_csv('{}/actions_{}.csv'.format(directory_name,i))
    actions = [x for x in actions['action'][:]]

    feats = pd.read_csv('{}/features_{}.csv'.format(directory_name,i))
    compute = [[float(x) for x in feats['compute'][j][1:-1].split(', ')] for j in range(len(feats['compute']))]
    memory = [[float(x) for x in feats['memory'][j][1:-1].split(', ')] for j in range(len(feats['memory']))]
    bandwidth = [[float(x) for x in feats['bandwidth'][j][1:-1].split(', ')] for j in range(len(feats['bandwidth']))]
    
    for k in range(len(actions)):
    
        comp = np.reshape(compute[k],(len(compute[k]),1))
        mem = np.reshape(memory[k],(len(memory[k]),1))
        bw = np.reshape(bandwidth[k],(len(bandwidth[k]),1))
        feat = np.concatenate((comp,mem),axis=-1)

        dgl_g.ndata['features'] = feat
        dgl_g.edata['features'] = bw

        netx_g = dgl_g.to_networkx(node_attrs=['features','_name'],edge_attrs=['features','_name'])
        
        node_colz = {}
        for node in netx_g.nodes():
            name = dgl_g.ndata['_name'][node].numpy().decode("utf-8")
            node_colz[name] = {}
            if node == actions[k]:
                node_colz[name]['colour'] = [0,0,1]
            elif name.split('_')[0] != 'Resource':
                node_colz[name]['colour'] = [0,0,0]
            else:
                ft = np.sum(netx_g.nodes[node]['features'])/20
                node_colz[name]['colour'] = [1-ft,ft,0]

        edge_colz = {}
        for edge in netx_g.edges():
            
            src = dgl_g.ndata['_name'][edge[0]].numpy().decode("utf-8")
            dst = dgl_g.ndata['_name'][edge[1]].numpy().decode("utf-8")

            ft = netx_g[edge[0]][edge[1]][0]['features'].numpy()[0]

            edge_colz[(src,dst)] = {}
            edge_colz[(src,dst)]['colour'] = [1-ft,1-ft,1-ft]
            edge_colz[(dst,src)] = {}
            edge_colz[(dst,src)]['colour'] = [1-ft,1-ft,1-ft]


        g = net.graph
        nx.set_node_attributes(g, node_colz)
        nx.set_edge_attributes(g, edge_colz)
        pos = nx.get_node_attributes(g,'pos')

        node_colz = [g.nodes[node]['colour'] for node in g.nodes()]
        edge_colz = [g.edges[edge]['colour'] for edge in g.edges()]

        plt.figure(figsize=size)
        nx.draw(g,pos,node_color=node_colz,edge_color=edge_colz,with_labels=False)