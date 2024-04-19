#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as smt
pd.set_option('display.max_columns', None)
import pprint 
from IPython.display import display, clear_output
from ipywidgets import Layout
import copy
import ast
import yaml
import matplotlib.ticker as ticker
from ipywidgets import widgets, interactive

import panel as pn
import math
pn.extension()
from typing import Dict, List

#pip install widgetsnbextension
pp = pprint.PrettyPrinter(indent=4)

import pprint 
pp = pprint.PrettyPrinter(indent=4)
import warnings
warnings.filterwarnings("ignore")
#import waterfall_chart

from statsmodels.tsa.seasonal import seasonal_decompose

from data_transform import *


def format_dict_plot(dict_plot, dict_input):
    
    d2 = copy.deepcopy(dict_plot)
    
    plot_type = dict_input['chart_type'].__name__
#     print(plot_type)
    
    if 'dict_changes' in dict_input:   
        for change in dict_input['dict_changes'].keys():

                d2[plot_type][change] = dict_input['dict_changes'][change] 

    for dict_key in d2['default'].keys():

            if dict_key not in d2[plot_type]:

                d2[plot_type][dict_key] = d2['default'][dict_key] 
#     print(d2)            
    return d2
            
    


# In[6]:


def line_plot(dict_plot,dict_input):
    

    #chart_package =  dict_input['chart_package']

    dict_chart = format_dict_plot(dict_plot,dict_input)
    df_new = dict_input['df_input']
    
    index_start = dict_input['index_range'][0]
    index_end = dict_input['index_range'][1]
    
    if index_start == 0:
        df_new = df_new.iloc[-index_end:,:]
    else:
        df_new = df_new.iloc[-index_end:-index_start,:]
    
    x_axis = dict_input['col_x_axis'][0]
    
    y_axis_cols = dict_input['col_y_axis']
    x_label = dict_input['col_x_axis'][0]
    y_label = dict_chart['line_plot']['y_label']
    title = dict_input['title']
    fig_len = dict_chart['line_plot']['fig_length']
    fig_breadth = dict_chart['line_plot']['fig_breadth']
    legend = dict_chart['line_plot']['legend']
    legend_fontsize = dict_chart['line_plot']['legend_fontsize']

    legend_location = dict_chart['line_plot']['legend_location']
    title_fontsize = dict_chart['line_plot']['title_fontsize']
    x_label_fontsize = dict_chart['line_plot']['x_label_fontsize']
    y_label_fontsize = dict_chart['line_plot']['y_label_fontsize']
    x_ticks_fontsize = dict_chart['line_plot']['x_ticks_fontsize']
    y_ticks_fontsize = dict_chart['line_plot']['y_ticks_fontsize']
    color = dict_chart['line_plot']['color']
    gridlines = dict_chart['line_plot']['gridlines']
    
    if len(y_axis_cols)!=1:
        p = df_new.plot(x=x_axis, y=y_axis_cols, kind="line",color=color)
        
    else:
        p = df_new.plot(x=x_axis, y=y_axis_cols, kind="line")
    
    p.set_xlabel(x_label, fontsize = x_label_fontsize)
    p.set_ylabel(y_label, fontsize = y_label_fontsize)
    
    if title == 'Enter plot title':
        plt.title('Comparsion: '+' - '.join(y_axis_cols),fontsize = title_fontsize)

    else:
        plt.title(title,fontsize = title_fontsize)

    plt.yticks(fontsize = y_ticks_fontsize)
    plt.xticks(fontsize = x_ticks_fontsize)
    
    plt.xticks(np.arange(0,index_end-index_start,1),df_new[x_axis].to_list(),rotation='vertical')
    plt.legend(fontsize=legend_fontsize,loc=legend_location)
    
    p.grid(visible=gridlines)


    plt.gcf().set_size_inches(fig_len, fig_breadth)
    plt.show()    


# In[7]:


#-------------------------------------------Bar Plot------------------------------------------

def prepare_data_barplot(df):
        #display(df)    
    df_bar = pd.DataFrame(df[df._get_numeric_data().columns].sum(axis=0),columns = ['Total Sum'])

    return df_bar


def bar_plot(dict_plot,dict_input):
    
#     chart_package =  dict_input['chart_package']

    dict_chart = format_dict_plot(dict_plot,dict_input)
    #print(dict_plot)
    df = dict_input['df_input']
    index_start = dict_input['index_range'][0]
    index_end = dict_input['index_range'][1]
    
    if index_start == 0:
        df = df.iloc[-index_end:,:]
    else:
        df = df.iloc[-index_end:-index_start,:]
    
    #print(df)
    y_axis_cols = dict_input['col_y_axis']
    
    df = prepare_data_barplot(df)
    
    fig_len = dict_chart['bar_plot']['fig_length']
    fig_breadth = dict_chart['bar_plot']['fig_breadth']
    legend = dict_chart['bar_plot']['legend']
    legend_fontsize = dict_chart['bar_plot']['legend_fontsize']
    legend_location = dict_chart['bar_plot']['legend_location']
    title_fontsize = dict_chart['bar_plot']['title_fontsize']
    color = dict_chart['bar_plot']['color']
    
    x_label = dict_chart['bar_plot']['x_label']
    y_label = dict_chart['bar_plot']['y_label']
    x_label_fontsize = dict_chart['bar_plot']['x_label_fontsize']
    y_label_fontsize = dict_chart['bar_plot']['y_label_fontsize']
    #x_label_fontsize = dict_chart['bar_plot']['x_label_fontsize']

    x_ticks_fontsize = dict_chart['bar_plot']['x_ticks_fontsize']
    y_ticks_fontsize = dict_chart['bar_plot']['y_ticks_fontsize']
    title = title = dict_input['title']
    rotation = dict_chart['bar_plot']['rotation']
    gridlines = dict_chart['bar_plot']['gridlines']
    
    df = df.T
    #print(df)
    if color == None:
        #p = df.loc[y_axis_cols].plot(kind='bar')#color=['black', 'red', 'green', 'blue', 'cyan']
        p = df.plot.bar(y=y_axis_cols)
    else:
        
        p = df.plot.bar(y=y_axis_cols,color = color)


    if title == 'Enter plot title':
        plt.title('Comparsion: '+' - '.join(y_axis_cols),fontsize = title_fontsize)

    else:
        plt.title(title,fontsize = title_fontsize)

    plt.yticks(fontsize = x_ticks_fontsize)
    plt.xticks(fontsize = y_ticks_fontsize,rotation=rotation)
    plt.legend(prop={'size': legend_fontsize}, loc=legend_location)
    p.set_xlabel(x_label, fontsize = x_label_fontsize)
    p.set_ylabel(y_label, fontsize = y_label_fontsize) 

    p.grid(visible=gridlines)

    plt.gcf().set_size_inches(fig_len, fig_breadth)
    plt.show()    


# In[8]:


def histogram(dict_plot,dict_input):
    
    
    clear_output()
    dict_chart = format_dict_plot(dict_plot, dict_input)
    
    df = dict_input['df_input']
             
    index_start = dict_input['index_range'][0]
    index_end = dict_input['index_range'][1]
    
    if index_start == 0:
        df = df.iloc[-index_end:,:]
    else:
        df = df.iloc[-index_end:-index_start,:]
    
    y_axis_cols = dict_input['col_y_axis']
    x_label = dict_chart['histogram']['x_label']
    y_label = dict_chart['histogram']['y_label']
    fig_len = dict_chart['histogram']['fig_length']
    fig_breadth = dict_chart['histogram']['fig_breadth']
    title_fontsize = dict_chart['histogram']['title_fontsize']
    x_label_fontsize = dict_chart['histogram']['x_label_fontsize']
    y_label_fontsize = dict_chart['histogram']['y_label_fontsize']
    x_ticks_fontsize = dict_chart['histogram']['x_ticks_fontsize']
    y_ticks_fontsize = dict_chart['histogram']['y_ticks_fontsize']

    binwidth = dict_chart['histogram']['binwidth']
    bins = dict_input['bins']
    

    kde = dict_chart['histogram']['kde']
    color = dict_chart['histogram']['color']
    title = dict_input['title']
    gridlines = dict_chart['histogram']['gridlines']

    for i in range(len(y_axis_cols)) :
        plt.figure(i)
        p = sns.histplot(data=df, x=y_axis_cols[i], kde = kde, bins = bins, binwidth = binwidth,color=color)
        p.set_xlabel(y_axis_cols[i], fontsize = x_label_fontsize)        
        p.set_ylabel(y_label, fontsize = y_label_fontsize)
        
            
        if title == 'Enter plot title':
            plt.title('Distribution: ' + y_axis_cols[i],fontsize = title_fontsize)

        else:
            plt.title(title,fontsize = title_fontsize)

        plt.yticks(fontsize = y_ticks_fontsize)
        plt.xticks(fontsize = x_ticks_fontsize)
        plt.grid(visible=gridlines)
        #plt.rc('font', size=10)
        plt.ticklabel_format(useMathText=True)
        #plt.rc('font', size=x_ticks_fontsize)
       # plt.yticks(np.arange(min(df[i]), max(df[i]),max(df[i])/(max(df[i])-min(df[i]))))
#         plt.xticks(np.arange(-3, 3, 0.5))
#         import pylab
#         pylab.tight_layout()


        plt.gcf().set_size_inches(fig_len, fig_breadth)
       
        plt.show()

 


# In[9]:


def string_to_dict_conversion(str_dict_changes):
    
    str_dict_changes = str_dict_changes.replace("\n","")
    str_dict_changes = str_dict_changes.replace("'", "\"")
    str_dict_changes = str_dict_changes.replace("#","")
    str_dict_changes = yaml.load(str_dict_changes)
    
    return str_dict_changes


# In[10]:


def prepare_barplot_data(df,granularity='Month',operation='sum'):
    """
    granularity: should be column name to group the data(e.g. Month, Year, Year-Month)

    operation: operation to be applied on grouped data(e.g. sum, mean)
    """
    
    if operation == 'sum': 
        df_bar = df.groupby(granularity).sum()
    if operation == 'mean':
        df_bar = df.groupby(granularity).mean()
    
    return df_bar



##Waterfall chart for spends


def waterfall_plot(dict_plot,dict_input):

   dict_chart = format_dict_plot(dict_plot, dict_input)

   x_axis = dict_input['col_x_axis'][0]
   y_axis_cols = dict_input['col_y_axis']
   x_label = dict_input['col_x_axis'][0]

   df = dict_input['df_input']

   fig_len = dict_chart['waterfall_plot']['fig_length']
   fig_breadth = dict_chart['waterfall_plot']['fig_breadth']
   title_fontsize = dict_chart['waterfall_plot']['title_fontsize']
   x_label_fontsize = dict_chart['waterfall_plot']['x_label_fontsize']
   y_label_fontsize = dict_chart['waterfall_plot']['y_label_fontsize']
   x_ticks_fontsize = dict_chart['waterfall_plot']['x_ticks_fontsize']
   y_ticks_fontsize = dict_chart['waterfall_plot']['y_ticks_fontsize']
   rotation_value = dict_chart['waterfall_plot']['rotation_value']
   sorted_value = dict_chart['waterfall_plot']['sorted_value']
   threshold = dict_chart['waterfall_plot']['threshold']
   y_axis_unit = dict_chart['waterfall_plot']['y_axis_unit']
   decimal_place = dict_chart['waterfall_plot']['decimal_place']
   net_label = dict_chart['waterfall_plot']['net_label']
   other_label = dict_chart['waterfall_plot']['other_label']
   blue_color = dict_chart['waterfall_plot']['blue_color']
   green_color =  dict_chart['waterfall_plot']['green_color']
   red_color= dict_chart['waterfall_plot']['red_color']
   gridlines = dict_chart['waterfall_plot']['gridlines']
#         title = dict_chart['waterfall_plot']['title']
   y_label = dict_chart['waterfall_plot']['y_label']
   x_label = dict_chart['waterfall_plot']['x_label']
   title = dict_input['title']

   if title == 'Enter plot title':
       title = 'Contribution Comparison' 
   short_y_axis_cols = [var.split('_asl_')[0] for var in y_axis_cols]    

  	
   waterfall_chart.plot(short_y_axis_cols,
                    df.loc[df['Variable'].isin(y_axis_cols),x_axis].values,
                    rotation_value=rotation_value, 
                    sorted_value=sorted_value,
                    threshold=threshold, 
                    formatting=str(y_axis_unit)+ "{:,."+str(decimal_place)+"f}",
                    net_label=net_label,
                    other_label=other_label,
                    blue_color=blue_color, 
                    green_color=green_color, red_color=red_color);


   plt.title(title, fontsize=title_fontsize)
   plt.xticks(fontsize=x_ticks_fontsize)
   plt.yticks(fontsize=y_ticks_fontsize)
   plt.ylabel(y_label, fontsize=y_label_fontsize)
#     plt.annotate(str(y_axis_unit)+ "{:,."+str(decimal_place)+"f}",fontsize=20)
   plt.gcf().set_size_inches(fig_len, fig_breadth)
   sns.despine()
   plt.grid(visible = gridlines)
   plt.show()

def boxplot(dict_plot,dict_input):

   dict_chart = format_dict_plot(dict_plot,dict_input)
   
   df = dict_input['df_input']
#     print(dict_input['dict_changes'])    
#     index_start = round(dict_input['index_range'][0])
#     index_end = round(dict_input['index_range'][1])
#     df = df.iloc[index_start:index_end,:]
       
   #if chart_package == 'seaborn':
   

   y_axis_cols = dict_input['col_y_axis']

   fig_len = dict_chart['boxplot']['fig_length']
   fig_breadth = dict_chart['boxplot']['fig_breadth']
   title_fontsize = dict_chart['boxplot']['title_fontsize']
   x_label_fontsize = dict_chart['boxplot']['x_label_fontsize']
   y_label_fontsize = dict_chart['boxplot']['y_label_fontsize']
   x_ticks_fontsize = dict_chart['boxplot']['x_ticks_fontsize']
   y_ticks_fontsize = dict_chart['boxplot']['y_ticks_fontsize']
   color = dict_chart['boxplot']['color']
   title = dict_input['title']
   orientation = dict_chart['boxplot']['orientation']
   gridlines = dict_chart['boxplot']['gridlines']
   plt.ioff()
   
   for i in y_axis_cols:
       plt.figure(i)

       p = sns.boxplot(x = df[i], orient= orientation,color=color)
       
       if title == 'Enter plot title':
           plt.title('Outlier Detection: '+i,fontsize = title_fontsize)
   
       else:
           plt.title(title,fontsize = title_fontsize)

       plt.gcf().set_size_inches(fig_len, fig_breadth)
       plt.yticks(fontsize = y_ticks_fontsize)
       plt.xticks(fontsize = x_ticks_fontsize)
#         #plt.ticklabel_format(useMathText=True)
#         plt.rc('font', size=x_ticks_fontsize)
#         plt.legend(prop={'size': legend_fontsize}, loc=legend_location)
       plt.grid(visible = gridlines)
       
       plt.show()
       
       
def violin_plot(dict_plot,dict_input):
   clear_output()
   
   df = dict_input['df_input']
   
   chart_package = 'seaborn'

   dict_chart = format_dict_plot(dict_plot, dict_input)
   
   y_axis_cols = dict_input['col_y_axis']
   

   fig_len = dict_chart['violin_plot']['fig_length']
   fig_breadth = dict_chart['violin_plot']['fig_breadth']
   title_fontsize = dict_chart['violin_plot']['title_fontsize']
   x_label_fontsize = dict_chart['violin_plot']['x_label_fontsize']
   y_label_fontsize = dict_chart['violin_plot']['y_label_fontsize']
   x_ticks_fontsize = dict_chart['violin_plot']['x_ticks_fontsize']
   y_ticks_fontsize = dict_chart['violin_plot']['y_ticks_fontsize']
#    gridlines = dict_chart['violin_plot']['gridlines']
   color = dict_chart['violin_plot']['color']
   title = dict_input['title']
   
   try:
       x_axis = dict_input['dict_changes']['x_axis']
       hue = dict_input['dict_changes']['hue']
   except:
       
       pass
   
   legend_fontsize = dict_chart['violin_plot']['legend_fontsize']
   legend_location = dict_chart['violin_plot']['legend_location']

   vertical = dict_chart['violin_plot']['vertical']

   if chart_package == 'matplotlib': 
       
       for i in y_axis_cols:

           plt.figure(i)
           p = plt.violinplot(df[i],vert=vertical,quantiles=[0,.25,.5,.75,1])#orientation
#                 
           plt.title(title,fontsize = title_fontsize)
           plt.gcf().set_size_inches(fig_len, fig_breadth)
           plt.yticks(fontsize = x_ticks_fontsize)
           plt.xticks(fontsize = y_ticks_fontsize)
           
           plt.grid()#visibility=gridlines)

           plt.show()

   else: 
       try:
           if 'x_axis' in dict_input['dict_changes']:
               if 'hue' in dict_input['dict_changes']:
                   x_label = dict_input['dict_changes']['x_axis']
   #         try:

                   for i in y_axis_cols:

                       plt.figure(i)
                       p = sns.violinplot(data = df,x = x_axis,y = i,orient="v",color=color,                                          hue= hue,                                         split=True,linewidth=4)#orientation

                       p.set_xlabel(x_label, fontsize = x_label_fontsize)        

                       if title == 'Enter plot title':
                           plt.title('Outlier Detection: '+i,fontsize = title_fontsize)

                       else:
                           plt.title(title,fontsize = title_fontsize)

                       plt.gcf().set_size_inches(fig_len, fig_breadth)
                       plt.yticks(fontsize = y_ticks_fontsize)
                       plt.xticks(fontsize = x_ticks_fontsize)
                       plt.grid()#visibility=gridlines)

                       plt.show()

       except:
           for i in y_axis_cols:

               plt.figure(i)
               plt.gcf().set_size_inches(fig_len, fig_breadth)
               p = sns.violinplot(data = df,y = i,orient="v",color=color,                                  split=True,linewidth=4)#orientation

           

               if title == 'Enter plot title':
                           plt.title('Outlier Detection: '+i,fontsize = title_fontsize)

               else:
                   plt.title(title,fontsize = title_fontsize)

               plt.gcf().set_size_inches(fig_len, fig_breadth)
               plt.yticks(fontsize = y_ticks_fontsize)
               plt.xticks(fontsize = x_ticks_fontsize)
               plt.grid()#visibility=gridlines)
               #plt.clf()
               plt.show()

       
       #         except:
           
#             for i in y_axis_cols:

#                 plt.figure(i)
#                 p = sns.violinplot(data = df,y = i,orient="v",color=color,\
#                                    split=True,linewidth=4)#orientation

#                 #p.set_xlabel(x_label, fontsize = x_label_fontsize)        

#                 plt.title(title,fontsize = title_fontsize)
#                 plt.gcf().set_size_inches(fig_len, fig_breadth)
#                 plt.yticks(fontsize = x_ticks_fontsize)
#                 plt.xticks(fontsize = y_ticks_fontsize)
#                 plt.grid(visibility=gridlines)
#                 plt.clf()
#                 #plt.show()
               

               

   #     plt.yticks(fontsize = x_ticks_fontsize)
   #     plt.xticks(fontsize = y_ticks_fontsize)
   #     plt.grid()
   #     #plt.legend(prop={'size': legend_fontsize}, loc=legend_location)#loc=4,
   #     plt.gcf().set_size_inches(fig_len, fig_breadth)
   #     plt.show()


# In[ ]:





# In[12]:


def ts_decomposition_plot(dict_plot,dict_input):


    dict_chart = format_dict_plot(dict_plot, dict_input) 

    x_axis = dict_input['col_x_axis'][0]
    y_axis_cols = dict_input['col_y_axis']
    x_label = dict_input['col_x_axis'][0]
    df = dict_input['df_input']

    fig_length = dict_chart['ts_decomposition_plot']['fig_length']
    fig_breadth = dict_chart['ts_decomposition_plot']['fig_breadth']
    title_fontsize = dict_chart['ts_decomposition_plot']['title_fontsize']
    x_label_fontsize = dict_chart['ts_decomposition_plot']['x_label_fontsize']
    y_label_fontsize = dict_chart['ts_decomposition_plot']['y_label_fontsize']
    x_ticks_fontsize = dict_chart['ts_decomposition_plot']['x_ticks_fontsize']
    y_ticks_fontsize = dict_chart['ts_decomposition_plot']['y_ticks_fontsize']
    period = dict_chart['ts_decomposition_plot']['period']
    extrapolate = dict_chart['ts_decomposition_plot']['extrapolate']
    title = dict_input['title']
    title_fontsize = dict_chart['ts_decomposition_plot']['title_fontsize']
    
    index_start = dict_input['index_range'][0]
    index_end = dict_input['index_range'][1]
    
#     if index_start == 0:
#         df_new = df_new.iloc[-index_end:,:]
#     else:
#         df_new = df_new.iloc[-index_end:-index_start,:]


    gridlines = dict_chart['ts_decomposition_plot']['gridlines']
    
    for y_axis in y_axis_cols:
        
        result = seasonal_decompose(df[y_axis], model='additive',period=period,extrapolate_trend=extrapolate)

        observed = pd.DataFrame(result.observed.values,columns = ['Observed'],index = result.observed.index)
        season = pd.DataFrame(result.seasonal.values,columns = ['Season'],index = result.seasonal.index)
        trend = pd.DataFrame(result.trend.values,columns = ['Trend'],index = result.trend.index)
        resid = pd.DataFrame(result.resid.values,columns = ['Residual'],index = result.resid.index)
        observed = observed.reset_index(drop=True)
        observed = pd.concat([df[x_axis],observed],axis=1)
        season = season.reset_index(drop=True)
        season = pd.concat([df[x_axis],season],axis=1)
        trend = trend.reset_index(drop=True)
        trend = pd.concat([df[x_axis],trend],axis=1)
        resid = resid.reset_index(drop=True)
        resid = pd.concat([df[x_axis],resid],axis=1)
        
        if index_start == 0:
            observed = observed.iloc[-index_end:,:]
            season = season.iloc[-index_end:,:]
            trend = trend.iloc[-index_end:,:]
            resid = resid.iloc[-index_end:,:]
        else:
            observed = observed.iloc[-index_end:-index_start,:]
            season = season.iloc[-index_end:-index_start,:]
            trend = trend.iloc[-index_end:-index_start,:]
            resid = resid.iloc[-index_end:-index_start,:]

        fig, (ax1, ax2,ax3,ax4) = plt.subplots(4,sharex=True)
        
        if title == 'Enter plot title':
            fig.suptitle('TS Decomposition: '+y_axis,fontsize =title_fontsize )
            
        else:
            fig.suptitle(title,fontsize =title_fontsize )

   
        ax1.plot(season['Month-Year'],observed['Observed'])
        ax2.plot(season['Month-Year'],season['Season'])
        ax3.plot(season['Month-Year'],trend['Trend'])
        ax4.plot(season['Month-Year'],resid['Residual'])
#         plt.xticks(np.arange(0,df.shape[0],1),season[x_axis].to_list(),rotation='vertical',fontsize=x_ticks_fontsize)
        plt.xticks(np.arange(0,index_end-index_start,1),season[x_axis].to_list(),rotation='vertical',fontsize=x_ticks_fontsize)

#         plt.xticks(np.arange(0,index_end-index_start,1),df.index.to_list(),rotation='vertical',fontsize = x_ticks_fontsize)
       
        ax1.set_title('Observed')
        ax2.set_title('Season')
        ax3.set_title('Trend')
        ax4.set_title('Residual')
#         ax1.grid(visible = gridlines)
#         ax2.grid(visible = gridlines)
#         ax3.grid(visible = gridlines)
#         ax4.grid(visible = gridlines)

        for a in [ax1, ax2, ax3, ax4]:
            for label in (a.get_yticklabels()):#(a.get_xticklabels() + a.get_yticklabels()
                label.set_fontsize(y_ticks_fontsize)
   
        plt.gcf().set_size_inches(fig_length, fig_breadth)
   
        plt.show()

# In[13]:


##    y-o-y Comparison plots


def yoy_line_plot(dict_plot,dict_input):
    clear_output()
      
    dict_chart = format_dict_plot(dict_plot,dict_input)
    
    
    df = dict_input['df_input']

    y_axis_cols = dict_input['col_y_axis']
    #x_label = dict_input['col_x_axis'][0]
    title = dict_input['title']
    #period = dict_input['period_for_comparison']
    fig_len = dict_chart['yoy_line_plot']['fig_length']
    fig_breadth = dict_chart['yoy_line_plot']['fig_breadth']
    legend = dict_chart['yoy_line_plot']['legend']
    title_fontsize = dict_chart['yoy_line_plot']['title_fontsize']
    x_label_fontsize = dict_chart['yoy_line_plot']['x_label_fontsize']
    x_ticks_fontsize = dict_chart['yoy_line_plot']['x_ticks_fontsize']
    y_ticks_fontsize = dict_chart['yoy_line_plot']['y_ticks_fontsize']
    y_label_fontsize = dict_chart['yoy_line_plot']['y_label_fontsize']
    color = dict_chart['yoy_line_plot']['color']
    x_label = dict_chart['yoy_line_plot']['x_label']
    y_label = dict_chart['yoy_line_plot']['y_label']
    legend_location = dict_chart['yoy_line_plot']['legend_location']
    legend_fontsize = dict_chart['yoy_line_plot']['legend_fontsize']
    x_axis = dict_chart['yoy_line_plot']['x_axis']
    gridlines = dict_chart['yoy_line_plot']['gridlines']
   
    for i in y_axis_cols:
        p = sns.lineplot(data=df, x=x_axis, y=i, hue='Year',palette=color);
        p.set_xlabel('Month', fontsize = x_label_fontsize)

        p.set_ylabel(y_label, fontsize = y_label_fontsize) 
            
        if title == 'Enter plot title':
            plt.title('y-o-y comparsion: '+i,fontsize = title_fontsize)
    
        else:
            plt.title(title,fontsize = title_fontsize)

        
        plt.legend(fontsize=legend_fontsize,loc=legend_location)

        plt.yticks(fontsize = y_ticks_fontsize)
        plt.xticks(fontsize = x_ticks_fontsize)
#         plt.xticks(np.arange(0,df.shape[0],1),df[x_axis].to_list(),rotation='vertical',fontsize=x_ticks_fontsize)

        p.grid(visible=gridlines)

        plt.gcf().set_size_inches(fig_len, fig_breadth)
        plt.show()  

    


# In[14]:


def area_plot(dict_plot,dict_input):

    dict_chart = format_dict_plot(dict_plot, dict_input)
#     if chart_package == 'matplotlib': 

    x_axis = dict_input['col_x_axis'][0]
    y_axis_cols = dict_input['col_y_axis']
    x_label = dict_input['col_x_axis'][0]
    title = dict_input['title']
    df = dict_input['df_input']
    var_name = dict_input['var_name']

    index_start = round(dict_input['index_range'][0])
    index_end = round(dict_input['index_range'][1])
#     df = df.iloc[index_start:index_end,:]
    if index_start == 0:
        df = df.iloc[-index_end:,:]
    else:
        df = df.iloc[-index_end:-index_start,:]
    
    fig_len = dict_chart['area_plot']['fig_length']
    fig_breadth = dict_chart['area_plot']['fig_breadth']
    title_fontsize = dict_chart['area_plot']['title_fontsize']
    x_label_fontsize = dict_chart['area_plot']['x_label_fontsize']
    y_label_fontsize = dict_chart['area_plot']['y_label_fontsize']
    x_ticks_fontsize = dict_chart['area_plot']['x_ticks_fontsize']
    y_ticks_fontsize = dict_chart['area_plot']['y_ticks_fontsize']
    title = dict_input['title']

    legend_fontsize = dict_chart['area_plot']['legend_fontsize']
    legend_location = dict_chart['area_plot']['legend_location']
    color = dict_chart['area_plot']['color']
    transparency = dict_chart['area_plot']['transparency']
    gridlines = dict_chart['area_plot']['gridlines']
    
    df = df.set_index(x_axis)

    y_axis_cols.remove(x_axis)
    p = df[y_axis_cols].plot.area(stacked=False,alpha=transparency,color=color,linewidth=2);#['black','#fdfd96','r','c']
    plt.xticks(np.arange(0,index_end-index_start,1),df.index.to_list(),rotation='vertical',fontsize = x_ticks_fontsize)
    
    p.set_xlabel(x_label, fontsize = x_label_fontsize)        

    plt.yticks(fontsize = y_ticks_fontsize)

#     plt.ticklabel_format(axis='y',useMathText=True)

    plt.grid(visible=gridlines)
    if title == 'Enter plot title':
        plt.title('ASL comparsion:'+var_name,fontsize = title_fontsize)
    
    else:
        plt.title(title,fontsize = title_fontsize)
    plt.legend(prop={'size': legend_fontsize}, loc=legend_location)#loc=4,
    plt.gcf().set_size_inches(fig_len, fig_breadth)
    plt.show()


# In[15]:


def generate_csv_for_stacked_waterfall(df,path):
        df_waterfall = pd.DataFrame(df.columns.T)
        df_waterfall['Upper_channel'] = ''
        df_waterfall['Category'] = ''
        df_waterfall['Negative'] = ''
        df_waterfall['Short_name'] = ''
        
        df_waterfall.columns = ['Variable','Upper_channel','Category','Negative','Short_name']
        
        return df_waterfall
    


# In[16]:


def set_widgets(dict_plot, list_df_input, *args):
   
    pn.extension()
    def on_button_clicked1(b):
        
        with output1: 
    
            if 'select_chart' in args:
                ##Chart type selection

                select_chart = widgets.Dropdown(
                    options= list(dict_list_plots.keys()), #['All']
                                                            #value='All',
                    description='Select chart:'
                )
                select_package = widgets.Dropdown(
                    options= list_packages, 
                    description='Select package:'
                )
                display(select_chart)
                
                #for waterfall chart
            if 'select_variables' in args:
                
                list_variables = list(dict_select_dataframe[select_df.value]['Variable'])
                select_vars = widgets.SelectMultiple(
                description="Select vars",
                options=list_variables,layout= Layout(width='70%', height='80px'))
                display(select_vars)
                
            if 'waterfall_column' in args:
                
                df_columns = dict_select_dataframe[select_df.value].columns
                waterfall_column = widgets.SelectMultiple(
                description="Select column",
                options=df_columns,layout= Layout(width='70%', height='80px'))
                display(waterfall_column)           
                
            if 'select_package' in args:
                ##Chart type selection

                select_package = widgets.Dropdown(
                    options= list_packages, 
                    description='Select package:'
                )
                display(select_package)              

            if 'range_slider' in args:
                
                range_slider = pn.widgets.RangeSlider(
                name='Index Slider', start=0, end=dict_select_dataframe[select_df.value].shape[0], \
                    value=(0, dict_select_dataframe[select_df.value].shape[0]), step=1.0)

                display(range_slider)

            if 'bins' in args:
                bin_slider=widgets.IntSlider(min=5, max=50, step=2,description='Bins')
                bin_slider.style.handle_color = 'lightblue'# can have float slider also 

                display(bin_slider)
    
            if 'select_dependent' in args:
                df_columns = dict_select_dataframe[select_df.value].columns
                dependent_var = widgets.SelectMultiple(
                description="Dependent",
                options=df_columns,layout= Layout(width='70%', height='80px'))

                display(dependent_var)
                
            if 'x_axis_cols' in args:

                df_columns = dict_select_dataframe[select_df.value].columns
                select_x_axis_cols = widgets.SelectMultiple(
                description="Select x-axis",
                options=df_columns,layout= Layout(width='70%', height='80px'))

                display(select_x_axis_cols)
                
            if 'correlation_slider' in args:
                correlation_slider = pn.widgets.RangeSlider(
                name='Correlation range', start=0, end=1, \
                    value=(0, 1), step=0.05)
                pn.extension()
                display(correlation_slider)
                
            if 'correlation_range' in args:
                min_range = widgets.Text(description='Min limit')
                max_range = widgets.Text(description='Max limit')
                Box = widgets.HBox([min_range,max_range])
                display(Box)
#                 display(max_range)
            
            if 'top_features' in args:
                top_features=widgets.IntSlider(min=1, max=20, step=1,description='#Top features')
                top_features.style.handle_color = 'lightblue'# can have float slider also 

                display(top_features)
            
            if 'y_axis_cols' in args:

                df_columns = dict_select_dataframe[select_df.value].columns
                select_y_axis_cols = widgets.SelectMultiple(
                description="Select y-axis",
                options=df_columns,layout= Layout(width='70%', height='80px'))
                display(select_y_axis_cols)
                
            if 'asl' in args:
                retention_slider=widgets.FloatSlider(min=0, max=1, step=0.05,description='Retention')
                retention_slider.style.handle_color = 'lightblue'# can have float slider 
                shape_slider=widgets.FloatSlider(min=0, max=10, step=0.5,description='Shape')
                shape_slider.style.handle_color = 'lightblue'# can have float slider also 
                steepness_slider = widgets.Text(description='Steepness')
#                 steepness_slider=widgets.FloatSlider(min=0, max=100, step=5,description='Steepness')
#                 steepness_slider.style.handle_color = 'lightblue'# can have float slider also 
                lag_slider=widgets.IntSlider(min=0, max=5, step=1,description='Lag')
                lag_slider.style.handle_color = 'lightblue'# can have float slider also 

                display(retention_slider)
                display(shape_slider)
                display(steepness_slider)
                display(lag_slider)

 
            if 'select_ASL' in args:
                select_asl = widgets.SelectMultiple(
                description="Choose ASL",
                options=['All','Raw','Adstock','Saturation','Lag'],layout= Layout(width='30%', height='80px'))
                display(select_asl)
                
                #Textarea
            if 'title_area' in args:
                title_area=widgets.Textarea(
                    description='Title:',
                    value='Enter plot title',
                )
                display(title_area)
        #         widgets_chosen.append(title_area.value)
            
            if 'changes_area' in args:
                #Dict changes 
                changes_area = widgets.Textarea(
                    description='Changes',
                    value='Enter dict_changes'
                )

                display(changes_area)
        #         widgets_chosen.append(changes_area.value)
            
            def on_button_clicked(b):
                with output:
                    if 'select_df' in args:
                        dict_input['df_input'] = dict_select_dataframe[select_df.value]

                    if 'range_slider' in args:
                        slider_vals = [int(round(range_slider.value[0])),int(round(range_slider.value[1]))]

                        dict_input['index_range'] = slider_vals

                    if 'bins' in args:
                        dict_input['bins'] = bin_slider.value
#                         print(bin_slider.value)

                    if 'select_chart' in args:    
                        dict_input['chart_type'] = select_chart.value
        
                    if 'select_package' in args:
                        dict_input['chart_package'] = select_package.value
                    
                    if 'select_dependent' in args:
                        dict_input['dependent_var'] = dependent_var.value
                        
                    if 'x_axis_cols' in args:
                        dict_input['col_x_axis']= list(select_x_axis_cols.value)
                    
                    if 'y_axis_cols' in args:
                        dict_input['col_y_axis'] = list(select_y_axis_cols.value)
                        
                    if 'select_dependent' in args:
                        dict_input['dependent_var'] = dependent_var.value
                              
                    if 'correlation_slider' in args: #For checking correlation
                        dict_input['min_cor'], dict_input['max_cor'] = correlation_slider.value[0],                         correlation_slider.value[1]
                        
                    if 'correlation_range' in args:
                        if min_range.value=='' and max_range.value=='':
                            dict_input['min_cor'], dict_input['max_cor'] = 0,1      
                        if min_range.value=='' and max_range.value!='':
                            dict_input['min_cor'], dict_input['max_cor'] = 0,float(max_range.value)
                        if min_range.value!='' and max_range.value=='':
                            dict_input['min_cor'], dict_input['max_cor'] = float(min_range.value),1    
                        if min_range.value!='' and max_range.value!='':    
                            dict_input['min_cor'], dict_input['max_cor'] = float(min_range.value), float(max_range.value)

                            
                    if 'top_features' in args: #For checking correlation
                        dict_input['top'] = top_features.value
                         
                    if 'asl' in args:     #For ASL comparison
                        dict_params_asl = {
                                                'retention': 0,
                                                'shape': 0,
                                                'steepness': 0,
                                                'lag': 0,
                                                'adstock_col': None, 
                                                'saturation_col': None,
                                                'lag_col': None,
                                                'lst_new_col_names_ads': None,
                                                'lst_new_col_names_sat': None,
                                                'lst_new_col_names_lag': None,
                                                'keep_input_col': True, 
                                                'keep_intermediate_col': True
                                            }
                        path_base = os.path.join(os.path.expanduser('~'), 'Desktop', 'MMM_ADKR', 'ASL_testing')
                        folder_output = "/output/"
                        x_axis = list(select_x_axis_cols.value)[0]
                        var_name = list(select_y_axis_cols.value)[0]
                        dict_input['var_name'] = var_name
                        df_raw = dict_input['df_input'][[var_name]]
                        dict_params_asl['retention'] = retention_slider.value
                        dict_params_asl['shape'] = shape_slider.value
                        if steepness_slider.value == '':
                            dict_params_asl['steepness'] = 100
                        else:
                            dict_params_asl['steepness'] = float(steepness_slider.value)
                        dict_params_asl['lag'] = lag_slider.value

                        df_asl = pd.DataFrame()
                        df_asl = asl_col(dict_select_dataframe[select_df.value], 		var_name,                                                                         dict_params_asl)

                        df_asl = pd.concat([dict_select_dataframe[select_df.value][x_axis],df_asl],axis=1)
#                         df_asl = df_asl[list(df_asl.columns)[::-1]]
#                         display(df_asl)
                       
                    if 'select_ASL' in args:
                        chosen_asl = select_asl.value
                        list_asl = [0]
                        if 'All' in chosen_asl:
                            list_asl = [0,1,2,3,4]
                            pass
                        if 'Raw' in chosen_asl:
                            list_asl.append(1)
                        if 'Adstock' in chosen_asl:
                            list_asl.append(2)
                        if 'Saturation' in chosen_asl:
                            list_asl.append(3)
                        if 'Lag' in chosen_asl:
                            list_asl.append(4)
                        if len(chosen_asl)==0:  
                            list_asl = [0,1,2,3,4]
                        list_asl = list(set(list_asl))    

                        df_asl = df_asl.iloc[:,list_asl]
                        dict_input['df_input'] = df_asl
                        dict_input['col_y_axis'] = list(df_asl.columns) 
                 
                    if 'waterfall_column' in args:  #waterfall specific only
                        dict_input['col_x_axis'] = list(waterfall_column.value)
                        
                    if 'select_variables' in args:  #waterfall specific only
                        dict_input['col_y_axis'] = list(select_vars.value)
               
                    if 'title_area' in args: 
                        dict_input['title'] = title_area.value

                    if 'changes_area' in args:
#                         dict_input['dict_changes'] = {}
#                         print(dict_input['dict_changes'])
                        changes_done = string_to_dict_conversion(changes_area.value)
                        if type(changes_done) == dict:
                            dict_input['dict_changes'] = changes_done 
#                         else:
#                             dict_input['dict_changes'] = {}           
                    
                    if 'line_plot' in args:
                        dict_input['chart_type'] = line_plot
                        line_plot(dict_plot,dict_input)
                    
                    if 'yoy_line_plot' in args:
                        dict_input['chart_type'] = yoy_line_plot
                        yoy_line_plot(dict_plot,dict_input)
        
                    if 'bar_plot' in args:
                        dict_input['chart_type'] = bar_plot
                        bar_plot(dict_plot,dict_input)

                    if 'histogram' in args:
                        dict_input['chart_type'] = histogram
                        histogram(dict_plot,dict_input)
                    
                    if 'boxplot' in args:
                        dict_input['chart_type'] = boxplot
                        boxplot(dict_plot,dict_input)
                        
                    
                    if 'violin_plot' in args:
                        dict_input['chart_type'] = violin_plot
                        violin_plot(dict_plot,dict_input)    
                        
                    if 'waterfall_plot' in args:
                        dict_input['chart_type'] = waterfall_plot
                        waterfall_plot(dict_plot,dict_input)    
                        
                    if 'area_plot' in args:
                        dict_input['chart_type'] = area_plot
                        area_plot(dict_plot,dict_input)    
                    if 'ts_decomposition_plot' in args:
                        dict_input['chart_type'] = ts_decomposition_plot
                        ts_decomposition_plot(dict_plot,dict_input)    
                    
                    if 'correlation_plot' in args:    
                        dict_input['chart_type'] = correlation_plot
                        correlation_plot(dict_plot,dict_input)    
                        
#                     dict_input['dict_changes'] = {}

            ## Button
            button = widgets.Button(description="Plot")
            output = widgets.Output()
            
            display(button, output)

            button.on_click(on_button_clicked)
    

    if 'select_df' in args:
        select_df = widgets.Dropdown(
            options= list_df_input, #['All']
                                                    #value='All',
            description='Select dataframe:'
        )

        display(select_df)
#         widgets_chosen.append(select_df.value)
## Button for showing dataframe specific widgets
    button1 = widgets.Button(description="Select df")
    output1 = widgets.Output()
    
    display(button1, output1)

    button1.on_click(on_button_clicked1)

#     return dict_input
       


# In[34]:


def spends_comparison_waterfall(list_df_input):
    
    button0 = widgets.Button(description="Select df")
    output0 = widgets.Output()
    
    def on_button_clicked0(e):
        
        def on_button_clicked_wf1(e):
        
            def on_button_clicked_wf2(f):
#                 print('reached on button clicked 2')
                df_uc = di[di['Upper_channel'].isin(list(select_upper_channel.value))]

                selected_vars = list(df_uc['Variable'].unique())

                #Filtering those columns from main df
                dff =  df[selected_vars]

                sp = pd.DataFrame(dff.sum(axis=0),columns=[category])
                sp = sp.reset_index()
                sp.columns = ['Variable',category]

                sp = pd.merge(df_uc,sp,how='inner')
                sp.drop('Category',axis=1,inplace=True)

                sp1 = sp.pivot_table(index='Upper_channel',columns = ['Variable'])
                sp1.fillna(0,inplace=True)

                sp1 = sp1.xs(category, axis=1, drop_level=True)

                sp2 = sp1.reset_index()
                sp2 = sp2.replace(0,np.nan).dropna(axis=1,how="all")
                sp2 = sp2.fillna(0)
           
                selected_upper_channels = list(select_upper_channel.value)
                negative_upper_channels = list(di[di['Negative']=='Yes']['Upper_channel'].unique())
#                 print('negative_upper_channels 1st',negative_upper_channels)
                negative_upper_channels = [ch for ch in selected_upper_channels if ch in negative_upper_channels]
                selected_upper_channels_copy = selected_upper_channels.copy()
#                 print('selected_upper_channels_copy',selected_upper_channels_copy)
#                 print('negative_upper_channels 2nd',negative_upper_channels)
                
                for i in range(len(negative_upper_channels)):
                    selected_upper_channels_copy.remove(negative_upper_channels[i])
                positive_upper_channels = selected_upper_channels_copy   

                dict_input['df_final'] = sp2
                dict_input['chart_type'] = stacked_waterfall_plot
                dict_input['positive_upper_channels'] = positive_upper_channels
                dict_input['negative_upper_channels'] = negative_upper_channels
                
#                 print('negative_upper_channels',dict_input['negative_upper_channels'])
#                 print('positive_upper_channels',dict_input['positive_upper_channels'])
#                 print('sp2')
#                 display(sp2)
                stacked_waterfall_plot(dict_plot,dict_input)
#--------------------------------------------------------------------------------

            category = list(select_category.value)[0]
            dict_input['category'] = category
            di = dict_input['di']
#             display(di)
#             print(category,'category')
            di = di[di['Category']==category]
#             display(di)

            unique_upper_channels = list(di['Upper_channel'].unique())
            #remove nan from categories
            if np.nan in unique_upper_channels:
                unique_upper_channels.remove(np.nan)
#             print('unique_upper_channels',unique_upper_channels)    
                        
            select_upper_channel = widgets.SelectMultiple(
            description="Upper-channels",
            options = unique_upper_channels,layout= Layout(width='30%', height='80px'))

            display(select_upper_channel)
            
            button_wf2 = widgets.Button(description="plot")
            output_wf2 = widgets.Output()

            display(button_wf2, output_wf2)

            button_wf2.on_click(on_button_clicked_wf2)

#------------------------------------------------------------------------------------------- 
        
        dict_input['df_input'] = dict_select_dataframe[select_df.value]
#         print(dict_input['df_input']['CRM_Kakao_Message_Spending'].sum())
        df_columns = dict_select_dataframe[select_df.value].columns
        path = dict_plot['stacked_waterfall_plot']['path']
        df = dict_select_dataframe[select_df.value]
        df_input_waterfall = generate_csv_for_stacked_waterfall(df,path)

        df_input_waterfall.to_csv(path+'/Waterfall_chart_input.csv')
        print('Waterfall_chart_input.csv has been downloaded!')
        print('\n')
        print('Please fill up the Waterfall_chart_input.csv and save in \n ./Input folder')

#         check = "No" 

#         while check!="y":
#         check = input("Enter y when done")   
#         if check != 'y':
#             check = input("Enter y when done")
            
#         print(path)
#         print(check)
        
#         print(path + '/Input/Waterfall_chart_input.csv')
#         print(os.listdir(path+'/Input'))
        
        if len(os.listdir(path+'/Input'))!=0:
            
            di = pd.read_csv(path + '/Input/Waterfall_chart_input.csv',index_col=0)
           #----------------------------------------------------------------------------------- 
            ## Getting channel names and their shortnames(reqd. while plotting)
            list_names = list(di['Short_name'])
            list_channels = list(di['Variable'])
            dict_short_names = {}
            for i in range(len(list_channels)):
                if str(list_names[i]) == 'nan':
                    dict_short_names[list_channels[i]] = list_channels[i]
                else:
                    dict_short_names[list_channels[i]] = list_names[i]
            
            dict_input['dict_short_names'] = dict_short_names
            
            #---------------------------------------------------------------------------------
            
            dict_input['di'] = di
            category_list = list(di['Category'].unique()) 
            if np.nan in category_list:
                category_list.remove(np.nan)
                
            ##Will show category
            select_category = widgets.SelectMultiple(
            description="Category",
            options = category_list,layout= Layout(width='30%', height='80px'))

            display(select_category)

            button_wf1 = widgets.Button(description="Select category")
            output_wf1 = widgets.Output()

            display(button_wf1, output_wf1)

            button_wf1.on_click(on_button_clicked_wf1)

        
    #---------------------------------------------------------------------    
    select_df = widgets.Dropdown(
            options= list_df_input,
                                                    
            description='Select dataframe:'
        )

    display(select_df)
        
    display(button0, output0)

    button0.on_click(on_button_clicked0)
    


# In[40]:


##stacked_waterfall chart for spends

from typing import Dict, List

def stacked_waterfall_plot(dict_plot,dict_input):
    
    def get_data_stacked_waterfall(df: pd.DataFrame) -> Dict[str, List[int]]:
        """Given a dataframe, transforms it to a dictionary which will be used for
        rendering a stacked bar chart.

        Args:
            df(pd.DataFrame): Dataframe which contains data for the chart

        Returns:
            Dict, which will be used for rendering a stacked bar chart
        """
        row_count = len(df.index)
        # Underscore is pre-pended to Base to ensure that it doesn't show up in the
        
        
        data = {
            '_Base': [0] * row_count,
        }

        columns = df.columns[1: ]
        #Ordr is to be changed beforehand, bcz of base calculation
        pos = dict_input['positive_upper_channels']
        neg = dict_input['negative_upper_channels']
        
        index_order = pos + neg #+ ['Total']
#         print('index_order',index_order)
        df = df.set_index('Upper_channel')
        index_order = pos + neg #+ ['Total']
#         df_for_plotting = df_for_plotting.reindex(index_order)
        df = df.reset_index()
#         display('after resetting indexes',df)
        
    #     for change in ['Up', 'Down']:
        for column in columns:
    #         data[column + ' ' + change] = [0] * row_count
              data[column] = [0] * row_count  

        for column in columns:
            for idx, value in enumerate(df[column]):
                if value >= 0:
    #                 data[column + ' Up'][idx] = df.loc[idx, column]
                      data[column][idx] = df.loc[idx, column]
                else:
                    data[column][idx] = df.loc[idx, column]
    #                 data[column + ' Down'][idx] = df.loc[idx, column]


        # calculate base
        data = pd.DataFrame(data)
#         print('initial data before re-ordering')

# #         df_for_plotting = df_for_plotting.reindex(index_order)
#         display(data)
        
        list_sum_neg = []
        sums = list(data.sum(axis=1))
#         print('sums',sums)
        for i in range(data.shape[0]):
            sum_neg = 0

            for j in range(1,data.shape[1]):
                if data.iloc[i,j]<0:
                    sum_neg += data.iloc[i,j]
            list_sum_neg.append(sum_neg)        
#         print('list_sum_neg',list_sum_neg)
        base = [0]*data.shape[0]
        for i in range(1,data.shape[0]):

            base[i] = base[i-1] - list_sum_neg[i-1] + sums[i-1] + list_sum_neg[i] 

        data['_Base'] = base    
#         print('data from get data()')
#         display(data)
        return data

    
    def render_chart_stacked_waterfall(data: Dict[str, List[int]], df: pd.DataFrame,dict_chart,dict_input) -> None:
        """Given data and df, it renders a stacked bar (waterfall) chart using the
        data.

        Args:
            data(Dict[str, List[int]]): Data which will be used for rendering the
                chart
            df(pd.DataFrame): Dataframe for some auxiliary operations

        Returns:
            None
        """
        fig_len = dict_chart['stacked_waterfall_plot']['fig_length']
        fig_breadth = dict_chart['stacked_waterfall_plot']['fig_breadth']
        title_fontsize = dict_chart['stacked_waterfall_plot']['title_fontsize']
        x_label_fontsize = dict_chart['stacked_waterfall_plot']['x_label_fontsize']
        y_label_fontsize = dict_chart['stacked_waterfall_plot']['y_label_fontsize']
        x_ticks_fontsize = dict_chart['stacked_waterfall_plot']['x_ticks_fontsize']
        y_ticks_fontsize = dict_chart['stacked_waterfall_plot']['y_ticks_fontsize']
        rotation = dict_chart['stacked_waterfall_plot']['x_ticks_rotation']

        gridlines = dict_chart['stacked_waterfall_plot']['gridlines']
    #     title = dict_chart['stacked_waterfall_plot']['title']
        y_label = dict_chart['stacked_waterfall_plot']['y_label']
        x_label = dict_chart['stacked_waterfall_plot']['x_label']
        legend_fontsize = dict_chart['stacked_waterfall_plot']['legend_fontsize']
#         print('dict_input keys',dict_input.keys())
#         print(dict_input['positive_upper_channels'])
#         print(dict_input['negative_upper_channels'])
        
        # get the first column
        particular_names = df.loc[:,'Upper_channel']
#         print('df start of render func')
#         display(df)
#         print('particular_names',particular_names)
#         display(df,'df')
#         display()
        X_AXIS = [particular for particular in particular_names]
#         print('X_AXIS',X_AXIS)
        index = pd.Index(X_AXIS, name='Upper Channel')
#         colors = ['#ffffff00', '#00ff00', '#93f693', '#ff0000', '#f79696']
        
        df_for_plotting = data.copy()
#         print('data')
#         print(data)
        df_for_plotting = df_for_plotting.set_index(index)
        
        #Adding last column of total
#         df_for_plotting['Total'] = 0
#         df_for_plotting = df_for_plotting.append({'Upper_channel':'Total','Total':0},ignore_index = True)
#         df_for_plotting.fillna(0,inplace=True)
#         df_for_plotting.iloc[-1,-1]=df_for_plotting.iloc[:,1:].sum().sum()
#         print('before df_for_plottingTotal calculation')
#         display(df_for_plotting)
#         df_for_plotting.to_csv('waterfall_before_total_calculation.csv')
        df_for_plotting['Total'] = 0
        df_for_plotting.loc['Total','_Base']=0
        df_for_plotting = df_for_plotting.fillna(0)
        df_for_plotting.loc['Total','Total'] = sum(df_for_plotting.sum()) - df_for_plotting['_Base'].sum()
        df_for_plotting = df_for_plotting.abs()
        
#         pos = [col for col in pos if ]
        ##Ordering the bars: positive>>negative>>Total
    
#     #Ordr is to be changed beforehand, bcz of base calculation
#         index_order = pos + neg + ['Total']
#         df_for_plotting = df_for_plotting.reindex(index_order)
        
#         colors = ['#ffffff00','#E50000','#ff0000','#fa8072',(.223,.166,.147,.2),(0.1,0.9,0.1,0.2),(0.0,0.9,0.1,0.4),(0.0,0.9,0.1,0.6),(0.0,0.9,0.1,0.8),(0.0,0.9,0.1,1)]
    
        colors_all_pos = [(.0223,.566,.0147),(.030,.86,.049),'greenyellow',(.173,.223,.136,0.07),(.0165,.222,.003),(0.3,0.86,0.10),(0.3,0.86,0.049),(0.3,0.86,0.49,0.5),(0.3,0.86,0.49,0.7),(0.3,0.86,0.49,0.9)]
        colors_all_neg = ['#E50000','#FA8072','#ff0000']
    #     white_color = '#ffffff00'
        
        dict_count = {}

        b = df_for_plotting.copy()
        for i in range(len(b.index)): #add -1 here for total
            count_channels = 0

            for j in range(1,len(b.columns)-1) : #add -1 here
                if b.iloc[i,j]!=0:
                    count_channels+=1

            dict_count[b.index[i]] = count_channels    

    #grey for total bar
    
        list_color = ['#ffffff00']  #white color for base
        pos = dict_input['positive_upper_channels']
        neg = dict_input['negative_upper_channels']
        
        
#         print('pos',pos)
#         print('neg',neg)
        for ch in pos:
            counts = dict_count[ch]
            list_color.extend(colors_all_pos[:counts])

        for ch in neg:
            counts = dict_count[ch]
            list_color.extend(colors_all_neg[:counts])
            
        for i in range(len(b.index)):
            list_color.append('#808080')
         #for 'Total color':grey
#         print('dict_count',dict_count)
#         print('list_color',list_color)
#----------------------------------------------------------------------------------------------------------
        ##Replacing names to shortnames
        dict_short_names = dict_input['dict_short_names']
        list_channels = list(dict_short_names.keys())
        cols_to_exclude = [col for col in df_for_plotting.columns if col not in list_channels]

        for i in cols_to_exclude:
            dict_short_names[i] = i
        df_for_plotting.columns = df_for_plotting.columns.map(dict_short_names)
       
     #-----------------------------------------------------------------------------   
        #For labelling
        e = df_for_plotting.copy()
        
        labels = []
#         print('e columns',e.columns)
        for i in range(1,len(e.columns)):
            labels.extend(len(e.index)*[e.columns[i]])
        labels.append('Total')
#         df_for_plotting.to_csv('df_for_plotting.csv')
#         print('labels',labels)
#         print(len(labels),len(labels))
#         print('list of colors')
#         print(list_color)
        di = dict_input['di']
        category = dict_input['category']
        list_channels = list(di['Variable'])


#-----------------------------------------------------------------------------------------------------------
#         display('df_for_plotting just before plotting')
#         display(df_for_plotting)
        ax = df_for_plotting.plot(
            kind='bar',
            stacked=True,
            figsize=(fig_len, fig_breadth),
            color=list_color
        )
        ax.set_ylabel(category)

        row_count = len(X_AXIS)
        # ignore the base texts
        idx = 0
        for p in ax.patches[row_count:]:
            width, height = p.get_width(), p.get_height()
    
            x, y = p.get_xy()

            if height:
                ax.text(
                    x + width / 2, 
                    y + height / 2, 
                    labels[idx],#labels[idx]
                    horizontalalignment='center', 
                    verticalalignment='center',
                    color='black',
                    fontsize=25
                )
            idx+=1  
            
        rotation = dict_chart['stacked_waterfall_plot']['x_ticks_rotation']
        plt.xticks(rotation=rotation)
        plt.legend('')
        plt.xticks(fontsize=x_ticks_fontsize)
        plt.yticks(fontsize=y_ticks_fontsize)
        plt.ylabel(y_label, fontsize=y_label_fontsize)
        plt.xlabel(x_label, fontsize=x_label_fontsize)
        plt.gcf().set_size_inches(fig_len, fig_breadth)
#         print('y_label_fontsize',y_label_fontsize)
#         print('x_label_fontsize',x_label_fontsize)
        plt.title(dict_input['category']+' Comparison',fontsize = title_fontsize)
        plt.tight_layout()
        plt.grid(visible=gridlines)
        plt.show()
        
#         return df_for_plotting

    dict_chart = format_dict_plot(dict_plot, dict_input)

    df = dict_input['df_input']
    sp2 = dict_input['df_final']
    
    data = get_data_stacked_waterfall(sp2)
    
    render_chart_stacked_waterfall(data,sp2,dict_chart,dict_input)
    
#     plt.title(title, fontsize=title_fontsize)


# In[19]:


def correlation_plot(dict_plot, dict_input):
    
    dict_chart = format_dict_plot(dict_plot, dict_input)
    
    
    dependent_var = dict_input['dependent_var']
    
    x_axis = dict_input['col_x_axis'][0]

    df = dict_input['df_input']
    
    fig_length = dict_chart['correlation_plot']['fig_length']
    fig_breadth = dict_chart['correlation_plot']['fig_breadth']
    title_fontsize = dict_chart['correlation_plot']['title_fontsize']
    x_label_fontsize = dict_chart['correlation_plot']['x_label_fontsize']

    x_ticks_fontsize = dict_chart['correlation_plot']['x_ticks_fontsize']
    x_ticks_rotation = dict_chart['correlation_plot']['x_ticks_rotation']
    

    cmap = dict_chart['correlation_plot']['cmap']

    title_fontsize = dict_chart['correlation_plot']['title_fontsize']
    
    title = dict_input['title']
    
    asl_files_path = dict_chart['correlation_plot']['asl_files_path'] + x_axis + '.csv'
    df_asl = pd.read_csv(asl_files_path,index_col=0)
    target_var = 'KPI_Total_Sales'
    
    cor =  df_asl.corrwith(df[target_var])
    cor = pd.DataFrame(cor,index=df_asl.columns.to_list(),columns = ['Correlation'])
    min_cor = dict_input['min_cor']
    max_cor = dict_input['max_cor']
    top = dict_input['top']

    cor = cor[(cor['Correlation']>min_cor) & (cor['Correlation']<max_cor) ]
    cor = cor.nlargest(top,'Correlation')

    x_ticks_fontsize = dict_chart['correlation_plot']['x_ticks_fontsize']
    x_ticks_rotation = dict_chart['correlation_plot']['x_ticks_rotation']
    cmap = dict_chart['correlation_plot']['cmap']#crest
    bar_label_fontsize = dict_chart['correlation_plot']['bar_label_fontsize']

    fig, ax = plt.subplots(figsize=(fig_length, fig_breadth))
    sns.heatmap([cor['Correlation']],cmap=cmap,annot=True,fmt=".3g",annot_kws={'size': bar_label_fontsize},linewidths=5) #"crest"
    ax.set_xticklabels(cor.index.to_list(),rotation=x_ticks_rotation,fontsize=x_ticks_fontsize)#'vertical'
    ax.set_yticklabels('o',fontsize=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=x_ticks_fontsize)

    if title=='Enter plot title':
        plt.title('Correlation: '+list(dependent_var)[0]+" & "+x_axis,fontsize = title_fontsize)
    else:
        plt.title(title,fontsize = title_fontsize)



# In[20]:


def plot_kpi_and_channels(list_df_input):

    clear_output()
    pn.extension() #slider doesn't work sometimes without this
    
    set_widgets(dict_plot,list_df_input,'select_df','range_slider','x_axis_cols','y_axis_cols','title_area','changes_area','line_plot')

def comparison_bar_chart(list_df_input):
    
    set_widgets(dict_plot,list_df_input,'select_df','range_slider','y_axis_cols','title_area','changes_area','bar_plot')

def check_distribution(list_df_input):
    
    set_widgets(dict_plot,list_df_input,'select_df','range_slider','bins','y_axis_cols','title_area','changes_area','histogram') 

def check_outliers_boxplot(list_df_input):
    set_widgets(dict_plot,list_df_input,'select_df','y_axis_cols','title_area','changes_area','boxplot')
      
def check_outliers_violinplot(list_df_input):
    set_widgets(dict_plot,list_df_input,'select_df','y_axis_cols','title_area','changes_area','violin_plot')    

def yoy_comparison_plot(list_df_input):
    set_widgets(dict_plot,list_df_input,'select_df','y_axis_cols','title_area','changes_area','yoy_line_plot')
    
    
def asl_comparison(list_df_input):
    set_widgets(dict_plot,list_df_input,'select_df','range_slider','x_axis_cols','y_axis_cols','asl','select_ASL','title_area','changes_area','area_plot')    

def prepare_data_waterfall_plot(df):
           
    df_bar = pd.DataFrame(df[df._get_numeric_data().columns].sum(axis=0),columns = ['Total Sum'])

    return df_bar

def check_contributions(list_df_input):

    clear_output()
    set_widgets(dict_plot,list_df_input,'select_df','select_variables','waterfall_column','title_area','changes_area','waterfall_plot')    


def check_correlation(list_df_input):
    set_widgets(dict_plot,list_df_input,'select_df','select_dependent','x_axis_cols','correlation_range',               'top_features','title_area','changes_area','correlation_plot')


# In[21]:


def time_series_decomposition(list_df_input):
    set_widgets(dict_plot,list_df_input,'select_df','x_axis_cols','y_axis_cols','range_slider','title_area','changes_area','ts_decomposition_plot')


# In[22]:


#---------------------------------------Variable comparison---------------------------------------------------

## Button - var_comparison

def variable_comparison(list_df_input):

    def variable_comparison_internal(list_df_input):
        button_var_comparison = widgets.Button(description="Select Plot")
        output_var_comparison = widgets.Output()

        select_plot_var_comparison = widgets.Dropdown(
            options= ['Line Plot','Bar Plot'], #['All']
                                                    #value='All',
            description='Select Plot'
        )


        def on_button_clicked_var_comparison(e):

            with output_var_comparison:
                clear_output()

                if select_plot_var_comparison.value == 'Line Plot':
                    plot_kpi_and_channels(list_df_input)
                elif select_plot_var_comparison.value == 'Bar Plot':
                    comparison_bar_chart(list_df_input)

        display(select_plot_var_comparison)

        display(button_var_comparison,output_var_comparison)

        button_var_comparison.on_click(on_button_clicked_var_comparison)    
        
    variable_comparison_internal(list_df_input)
            


# In[23]:


#---------------------------------------Variable comparison---------------------------------------------------

## Button - var_comparison

def outlier_detection(list_df_input):

    def outlier_detection_internal(list_df_input):
        button_outlier_detection = widgets.Button(description="Select Plot")
        output_outlier_detection = widgets.Output()

        select_plot_outlier_detection = widgets.Dropdown(
            options= ['Box Plot','Violin Plot'], 
            description='Select Plot'
        )

        def on_button_clicked_outlier_detection(f):

            with output_outlier_detection:
                clear_output()
                if select_plot_outlier_detection.value == 'Box Plot':
                    check_outliers_boxplot(list_df_input)
                elif select_plot_outlier_detection.value == 'Violin Plot':
                    check_outliers_violinplot(list_df_input)

        display(select_plot_outlier_detection)

        display(button_outlier_detection,output_outlier_detection)

        button_outlier_detection.on_click(on_button_clicked_outlier_detection)    
        
    outlier_detection_internal(list_df_input)
            


# # Dict Plot

# In[24]:


dict_plot = {
   
    'default':
      { 
        
        'x_label_fontsize':30,
        'y_label_fontsize':28,
        'x_ticks_fontsize':30,
        'y_ticks_fontsize':30,
        'fig_length':40,
        'fig_breadth':12,
        'legend':True,
        'x_label':None,
        'y_label':None,
        'title_fontsize':40,
        'legend_location':'upper right',
          'legend_fontsize':30,
          'color':None,
          'gridlines':True,
          
          'vertical': False, ##Violin Plot,
          'orientation': "v", ##Violin Plot,
          'transparency':0.6,
          #'bins':50,
          'binwidth':None
              
    },
    'line_plot':
    {'gridlines':True
        
    },
              
    'histogram':{
        'y_label':'Count',
        'x_label':None,
        'kde':False,
#        'bins':20,#'auto',
        'binwidth':None
    },
              
    'boxplot':{
         'fig_length':20,
        'fig_breadth':5,
        'orientation': "h",
        'title_fontsize':20,
        'x_label_fontsize':5,
        'x_ticks_fontsize':20,
        'gridlines':True
    },
     'area_plot':{
         
         'transparency':0.2,
         'legend_location':'best',
         'legend_fontsize':30,
         'color':['black','orange','red','c'],#['black','#fdfd96','red','c']
         'x_ticks_fontsize':30,
         'gridlines':True
     },
       'violin_plot':{
        'fig_length':15,
        'fig_breadth':5,
          'vertical': False,
           'orientation': "v",
           'gridlines':False,
           'title_fontsize':25,
           'y_ticks_fontsize':15
     },
    'waterfall_plot':{   ##Non stacked: for contribution
        'rotation_value':90, 
         'sorted_value' : True, #sort values as per absolute magnitude
         'threshold' : 0.01, #0.1 means points with > 10% of start value will be present in the chart 
         'y_axis_unit': "",
         'decimal_place': 2,   
         'net_label' : 'Total \ncontribution', #Name for net value
         'other_label':'Small \nchanges',      #Name for small changes(total)
         'blue_color' :'slateblue',            #change blue color with some other color
         'green_color' :'greenyellow',                #change green color with some other color
         'red_color':'red',                    ##change red color with some other color
         'gridlines':True,
         'title': "Contributions Comparison",
         'y_label':"Contribution",
         'x_label':"Variable",
         'x_label_fontsize':30,
         'y_label_fontsize':30,
         'x_ticks_fontsize':30,
         'y_ticks_fontsize':30,
         'title_fontsize':50,
         'fig_length':40,
         'fig_breadth':150

     },
    'stacked_waterfall_plot':{  #Spends/Impressions comparison
        'path':'./Stacked_waterfall',
        'fig_length':40,
        'fig_breadth':18,
        'x_ticks_rotation': 90,
        'legend_fontsize':20,
        'x_ticks_fntsize':35,
        'y_ticks_fntsize':35,
        'gridlines':True
    },
    
    'ts_decomposition_plot':{
        'period' :12,
        'fig_length':14,
        'fig_breadth':10,
        'extrapolate':True,
        'x_ticks_fontsize':12,
        'y_ticks_fontsize':12,
        'title_fontsize':25,
        'x_label':'Upper channel',
        'gridlines':True
        
    },
    
    'bar_plot':{'rotation':0,
    'legend_location':'best',
    'gridlines':True
        
    },
    'yoy_line_plot':{
        'x_axis':'Month',
        'color':['r','g','b','y'],
        'legend_fontsize':22,
        'x_ticks_fontsize':25,
        'gridlines':True
        
    },
    'correlation_plot':{
        'asl_files_path':'./Transformed_onebyone/Transformed_monthly_level/',
        'x_ticks_rotation':90,
        'cmap':"Blues",#"inferno",#"Blues", #crest
        'x_ticks_fontsize' : 12,
        'bar_label_fontsize':17,
         'fig_length':20,
        'fig_breadth':4,
        'title_fontsize':15

#         'top_correlated':10
    }
    
}

dict_input = {}


# # Variables Comparison (Line & Bar Plot)

# In[43]:


# variable_comparison(list_df_input)


# # Spends/Impressions Comparison (Waterfall chart)

# In[44]:


# spends_comparison_waterfall(list_df_input)


# # y-o-y Comparison

# In[45]:


# yoy_comparison_plot(list_df_input)


# # Check Distribution

# In[46]:


# check_distribution(list_df_input)


# # Outlier detection

# In[47]:


# outlier_detection(list_df_input)


# # Contributions

# In[48]:


# check_contributions(list_df_input)


# # ASL Comparison

# In[49]:


# asl_comparison(list_df_input)


# # Time Series Decomposition

# In[50]:


# time_series_decomposition(list_df_input)


# 
# # Correlation

# In[51]:


# check_correlation(list_df_input)


# # Update dict_plot

# In[76]:


def update_paths_in_dict_plot(asl_files_path, path_save_stacked_waterfall_input):
    path = path_save_stacked_waterfall_input
    dict_plot['correlation_plot']['asl_files_path'] = asl_files_path
    dict_plot['stacked_waterfall_plot']['path'] = path


# In[ ]:




