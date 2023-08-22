import time
import numpy as np
import pandas as pd
import os 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def date_time():
    '''
    get date and time in string format '_yymmdd_hhmm'
    at the moment the function is called.
    '''
    from datetime import date, datetime
    
    today = date.today()
    now = datetime.now() 

#     return today.strftime("%Y%m%d")[2:] +'_'+ now.strftime("%H%M")
    return today.strftime("%Y%m%d")[2:] + now.strftime("%H%M")


def classification_reports_difference(cr1, cr2):
    
    cr_diff = cr1.set_index('prdtypecode').subtract(cr2.set_index('prdtypecode')).reset_index()
    
    ## plot cr_diff
    art = sns.color_palette()

    fig, axs = plt.subplots(1,2,figsize = (5.833,9),gridspec_kw={'width_ratios': [3.0, 2.0]})

    sns.heatmap(cr_diff.set_index('prdtypecode')[['precision', 'recall', 'f1-score']], annot = True, 
                cmap='RdBu', vmin = -0.1, vmax = 0.1, ax = axs[0], cbar = False)

    sns.barplot(data = cr1, x = 'support', y='prdtypecode', color = 'grey', alpha = 0.75, ax = axs[1])

    axs[0].xaxis.set_ticks_position('top')
    axs[0].xaxis.set_tick_params(length = 0)
    axs[0].xaxis.set_label_position('top')
    axs[0].set_ylabel('Product Type Code')

    axs[1].set_xticks([0,500,1000,1500,2000])
    axs[1].set_xticklabels([0,'',1000,'',2000], fontsize=12)
    axs[1].yaxis.set_tick_params(labelleft=False)
    axs[1].set_ylabel('')
    axs[1].set_xlabel('Nb of observations')

    plt.subplots_adjust(wspace=0.02, hspace=0);
    
    return None