# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:27:16 2021

@author: nahran
"""

#input: exog_start, exog_end, endog_time, seq
#exog_start: starting year of observation
#exog_end: end year of observation
#endog_year: year of dependent variable
#seq: sequence number of the cross section

#output: dataframe of the cross section
#final: unbalanced pooled cross-sectional logit model with interactions
import itertools
import networkx as nx
from networkx.algorithms import bipartite #B=(U,V,E)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def cross_section(exog_start, exog_end, endog_year, seq):
    df_whole = pd.read_csv(r'C:\Users\nahran\X1\df_final.csv', header=0, encoding='latin1', index_col=False)
    Indy_raw = df_whole[(df_whole.round_time >= exog_start) & (df_whole.round_time <= exog_end)]
    Indy_raw.drop(columns=['investor_count','round_time'])
    Indy_bipart = nx.Graph()
    Indy_bipart.add_nodes_from(Indy_raw.round_company, bipartite=0)
    Indy_bipart.add_nodes_from(Indy_raw.investor, bipartite=1)
    Indy_bipart.add_edges_from(Indy_raw[['round_company', 'investor']].values.tolist())
    #Dep Set(t)
    Dep_raw = df_whole[df_whole.round_time == endog_year]
    Dep_raw.drop(columns=['investor_count','round_time'])
    Dep_bipart = nx.Graph()
    Dep_bipart.add_nodes_from(Dep_raw.round_company, bipartite=0)
    Dep_bipart.add_nodes_from(Dep_raw.investor, bipartite=1)
    Dep_bipart.add_edges_from(Dep_raw[['round_company', 'investor']].values.tolist())
    
    Indy_top = {n for n, d in Indy_bipart.nodes(data=True) if d["bipartite"]==0}
    Indy_bottom = set(Indy_bipart) - Indy_top
    Dep_top = {n for n, d in Dep_bipart.nodes(data=True) if d["bipartite"]==0}
    Dep_bottom = set(Dep_bipart) - Dep_top
    Indy_projection=nx.bipartite.generic_weighted_projected_graph(Indy_bipart, Indy_bottom)
    Indy_sorted_edges=sorted(list((Indy_projection.edges(data=True))))
    Indy_edge_weight=pd.DataFrame(Indy_sorted_edges, columns=["investor_a","investor_b", "weight"]) #AB and corresponding weight
    Indy_edge_weight.weight=[d['weight'] for d in Indy_edge_weight.weight]
    Indy_edge_weight["weight"].apply(type);
    Indy_edge_weight.describe(include='all')
    Indy_pairs = pd.DataFrame(list(itertools.combinations(Indy_bottom, 2)), columns=['investor_a', 'investor_b'])
    econ1 = pd.merge(Indy_pairs, Indy_edge_weight, on=['investor_a','investor_b'],how='left')
    econ1=econ1.fillna(0)
    Indy_pair_list=Indy_pairs[['investor_a', 'investor_b']].values.tolist()
    com_neigh_count = []
    for p in Indy_pair_list:
        neigh = nx.common_neighbors(Indy_projection,  p[0], p[1])
        neighbor = len(list(neigh))
        count = [p[0], p[1], neighbor]
        com_neigh_count.append(count)
    Indy_common = pd.DataFrame(com_neigh_count)
    Indy_common.rename(columns={0:'investor_a',1:'investor_b',2:'common_neighbors'})
    Indy_commons = Indy_common.rename(columns={0:'investor_a',1:'investor_b',2:'common_neighbors'})
    Indy_jaccard_empty=list(nx.jaccard_coefficient(Indy_projection))
    Indy_df_jaccard = pd.DataFrame(Indy_jaccard_empty)
    Indy_df_jaccard = Indy_df_jaccard.rename(columns={0:'investor_a',1:'investor_b',2:'jaccard'})
    if seq == 1:
        Indy_assort = nx.degree_mixing_matrix(Indy_projection, nodes=Indy_projection.nodes)
        plt.figure(figsize=(10,10))
        assort_map = sns.heatmap(Indy_assort[0:50,0:50])   
        degree_freq = nx.degree_histogram(Indy_projection)
        degree_prob = np.divide(degree_freq,len(Indy_bottom))
        degrees = range(len(nx.degree_histogram(Indy_projection)))
        plt.figure(figsize=(10,10))
        plt.loglog(degrees, degree_prob, 'go')
        plt.xlabel('Degree')
        plt.ylabel('P(k)')
    Indy_node_degree = list(Indy_projection.degree)
    Indy_degree = pd.DataFrame(Indy_node_degree, columns=['node', 'degree']).sort_values(by=['degree'])
    Indy_all_bridges = pd.DataFrame((list(nx.bridges(Indy_projection))), columns=['investor_a', 'investor_b'])
    Indy_all_bridges['bridge'] = 1
    dist = pd.DataFrame(nx.shortest_path_length(Indy_projection)).values
    geo_iter = range(0, len(dist))
    geo_pair = []
    for i in geo_iter:
         a = dist[i,0]
         b = dist[i,1].keys()
         geo = dist[i,1].values()
         paired_dist=pd.DataFrame(columns=['investor_a', 'investor_b', 'geodesic'])
         paired_dist.investor_b = b
         paired_dist.geodesic = geo
         paired_dist.investor_a = a 
         geo_pair.append(paired_dist)
    merged = pd.concat(geo_pair)
    merged['self']=np.where(merged['investor_a'] == merged['investor_b'], np.nan,1)
    merged_pair_geo = merged.dropna(subset=['self']).drop(columns='self')
    merged_pair_geo.geodesic= (1/merged_pair_geo['geodesic']).round(2)
    
    #Merge
    econ2 = pd.merge(econ1, Indy_commons, how='left', on=['investor_a', 'investor_b'])
    econ3 = pd.merge(econ2, Indy_df_jaccard, how='left', on=['investor_a', 'investor_b'])
    econ4 = pd.merge(econ3, Indy_all_bridges, how='left', on=['investor_a', 'investor_b'])
    econ5 = pd.merge(econ4, merged_pair_geo, how='left', on=['investor_a', 'investor_b'])
    econ6 = econ5.fillna(0)
    Dep_projection=nx.bipartite.generic_weighted_projected_graph(Dep_bipart, Dep_bottom)
    Dep_sorted_edges=sorted(list((Dep_projection.edges(data=True))))
    Dep_edge_weight=pd.DataFrame(Dep_sorted_edges, columns=["investor_a","investor_b", "link"])
    Dep_edge_weight.link=[d['weight'] for d in Dep_edge_weight.link]
    Dep_edge_weight["link"].apply(type);
    econ = pd.merge(econ6, Dep_edge_weight, how='left', on=['investor_a', 'investor_b'])
    econ = econ.fillna(0)
    econ_binary = econ
    econ_binary['link']=np.where(econ['link'] >= 1, 1, 0)
    return econ_binary
                                                                                            