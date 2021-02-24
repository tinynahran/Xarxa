# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 12:11:07 2021

@author: nahran
"""

#Structure: Data Cleaning, Creating Cross Sections and Logit Regression

# Part I: Data Cleaning

import pandas as pd
df_fundround=pd.read_csv(r'E:\Xarxa1\funding_rounds-munge.csv', header=0, encoding='latin1')
#print(df_fundround.head())
#print(df_fundround.groupby('investment_type').size())
df_vcround=df_fundround.loc[df_fundround['investment_type'].isin(['series_a','series_b','series_c'])]
#check investor_count for NaN
#print(df_vcround.investor_count.isnull().sum())
# NaNs will be checked with appended data
dropcol=(['country_code','type', 'org_uuid', 'lead_investor_uuids'])
df_vcround.drop(dropcol, axis=1, inplace=True)
#print(df_vcround.head())
round_inf = df_vcround.rename(columns={"announced_on":"round_time", "name":"round_company", "investment_type":"type","org_name":"company"})
round_clean = round_inf[['round_company','round_time', 'investor_count']]
pd.to_datetime(round_clean.round_time, infer_datetime_format=True)
dup=round_clean[round_clean.duplicated()]
duplicate=round_clean.loc[round_clean.round_company.isin(dup.round_company)]
duplicate.sort_values(by=['round_company'])
df_investment=pd.read_csv(r'E:\Xarxa1\investments-munge.csv', header=0, encoding='latin1')
dropcol2=(['name','funding_round_uuid','type'])
dfi = df_investment
dfi.drop(dropcol2, axis=1, inplace=True)
round_inv = dfi.rename(columns={"investor_name":"investor","is_lead_investor":"lead", "funding_round_name":"round_company"})
dftotal = pd.merge(round_clean, round_inv, on=['round_company'])
dftotal.reset_index(drop=True)
dftotal.round_time = pd.to_datetime(dftotal.round_time, infer_datetime_format=True, yearfirst=True)
timebin = pd.Series(pd.date_range('1968', freq='Y', periods =53))
#time_hist = dftotal.round_time.hist(bins=timebin, title='Deals Per Year');
#plt.savefig('deal_number.png')
cen21 = dftotal.round_time > '2000'
dftotal.round_time = dftotal.round_time[cen21]
#dftotal.isnull().sum()
dftot = dftotal.dropna(axis=0, subset=['round_time', 'investor'])
dropcol3=(['lead','investor_type'])
dftot.drop(dropcol3, axis=1, inplace=True)
dft=dftot.dropna(axis=0, subset=['investor_count'])
df=dft.reset_index(drop=True)

#Checking Max Investor Count
df.loc[df.investor_count ==df.investor_count.max()]
#df.investor_count.hist(bins = 95, density=1, cumulative=1, title='Deal Sizes');
#plt.savefig('deal_size.png')

df.round_time=df.round_time.dt.year
df_final=df[df.investor_count <=5] #Discarding larger deals than 6. Warning: Arbitrary
df_final.reset_index()
df_final.drop(columns=['investor_count'])
df_final.to_csv('df_final.csv', index=False)

# Setting the sequential networks
import networkx as nx
from networkx.algorithms import bipartite #B=(U,V,E)
#Generate Cumulative Bipartite Graphs
Years = list(range(2000, 2019))
#Discard 2019 for incomplete yearly data
#Slice into 6 sub-periods (to avoid huge computation): Regress->>
    #2000-2002 on 2003, 
    #2003-2005 on 2006, 
    #2006-2008 on 2009,
    #2009-2011 on 2012, 
    #2012-2014 on 2015, 
    #2015-2017 on 2018,
#Names: Indy1, Dep1, cinfra
#Logit Model: 1: Dep is binary; 2 Dep is ordinal
#print(Years)

