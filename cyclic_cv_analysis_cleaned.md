This notebook is for Daniels, Zhong, and Balsam (In Prep): Tracking of Sinusoidal Time is Not Scalar Invariant

The goal of the present work was to replicate some of the work of Staddon and colleagues using a more tradtional timing task (FI schedule) and determine whether timing is scalar invariant. This analysis is a wip and subject to change. 


```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os
import re
import pymysql
from sqlalchemy import create_engine
from datetime import date
pd.set_option('display.max_rows', 10000)
import warnings
warnings.filterwarnings('ignore')
import scipy.optimize as optimization
from scipy import signal
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.deterministic import Fourier
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 0.5
```


```python
#Establish connection to database

engine1 = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user="root", pw="u$watchmenR15!", db = "cfi_e1"))
engine2 = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user="root", pw="u$watchmenR15!", db = "cfi_e3_mz"))
```


```python
#Read in data
tracked_data_e1 = pd.read_sql("SELECT * FROM latandbpdata",con=engine1)
irt_data_e1 = pd.read_sql("SELECT * FROM irtdata", con=engine1)
rr_data_e1 = pd.read_sql("SELECT * FROM rrdata", con=engine1)
#Read in session notes
sessionNotes_e1 = pd.read_sql("SELECT * FROM sessions",con=engine1)

#Read in data
tracked_data_e2 = pd.read_sql("SELECT * FROM latandbpdata",con=engine2)
irt_data_e2 = pd.read_sql("SELECT * FROM irtdata", con=engine2)
rr_data_e2 = pd.read_sql("SELECT * FROM rrdata", con=engine2)
#Read in session notes
sessionNotes_e2 = pd.read_sql("SELECT * FROM sessions",con=engine2)
```


```python
all_racked_data = pd.concat([tracked_data_e1,tracked_data_e2],axis=0)
```

Response rate as a function of time in FI analysis
Select baseline data equivalent to similar amounts of training (36 sessions in total)


```python
rr_base20_base40 = rr_data_e1[(rr_data_e1['Session']<=63) & (rr_data_e1['Session']>=51)]
#find all Base20 mice
rr_base20_base40.loc[(rr_base20_base40['SubjectID']==21) | (rr_base20_base40['SubjectID']==23) | (rr_base20_base40['SubjectID']==26) | (rr_base20_base40['SubjectID']==28) | (rr_base20_base40['SubjectID']==30) | (rr_base20_base40['SubjectID']==32) | (rr_base20_base40['SubjectID']==33) | (rr_base20_base40['SubjectID']==35),'Group']='Base20'
rr_base20_base40.loc[(rr_base20_base40['SubjectID']==22) | (rr_base20_base40['SubjectID']==24) | (rr_base20_base40['SubjectID']==25) | (rr_base20_base40['SubjectID']==27) | (rr_base20_base40['SubjectID']==29) | (rr_base20_base40['SubjectID']==31) | (rr_base20_base40['SubjectID']==34) | (rr_base20_base40['SubjectID']==36),'Group']='Base40'
#Replace session 50 with 63 because of operant chamber malfunction
rr_base20_base40[(rr_base20_base40['SubjectID']>=29) & (rr_base20_base40['Session']==50)] = rr_base20_base40[(rr_base20_base40['SubjectID']>=29) & (rr_base20_base40['Session']==63)]
#Drop session 63 because not needed for baseline
rr_base20_base40 = rr_base20_base40[rr_base20_base40['Session']<=62]
```


```python
#Organize baseline of e2 data (Base 60)
rr_base60 = rr_data_e2[(rr_data_e2['Session']<=62) & (rr_data_e2['Session']>= 51)]
#Add group identifier
rr_base60['Group' ] = 'Base60'
```


```python
#concatenate all data
all_rr_base = pd.concat([rr_base20_base40,rr_base60],axis=0)
```


```python
#Add FI information to RR
for subj in all_rr_base.SubjectID.unique():
    for sess in all_rr_base[all_rr_base['SubjectID']==subj].Session.unique():
        all_rr_base.loc[(all_rr_base['SubjectID']==subj) & (all_rr_base['Session']==sess),'ProgFI'] = np.repeat(all_racked_data[(all_racked_data['SubjectID']==subj) & (all_racked_data['Session']==sess)].ProgFI.to_numpy(),8)
        
all_rr_base['LogRate'] = np.log(all_rr_base.Rate+0.001)
```


```python
#Add ascending and dscending information to lat and bp data
for subj in all_racked_data.SubjectID.unique():
    for sess in all_racked_data[all_racked_data['SubjectID']==subj].Session.unique():
        if all_racked_data[(all_racked_data['SubjectID']==subj) & (all_racked_data['Session']==sess)].ProgFI.to_numpy()[0] - all_racked_data[(all_racked_data['SubjectID']==subj) & (all_racked_data['Session']==sess)].ProgFI.to_numpy()[1] == 0:
            all_racked_data.loc[(all_racked_data['SubjectID']==subj) & (all_racked_data['Session']==sess),'Direction'] = 'Static'
        elif all_racked_data[(all_racked_data['SubjectID']==subj) & (all_racked_data['Session']==sess)].ProgFI.to_numpy()[0] - all_racked_data[(all_racked_data['SubjectID']==subj) & (all_racked_data['Session']==sess)].ProgFI.to_numpy()[1] < 0:
            all_racked_data.loc[(all_racked_data['SubjectID']==subj) & (all_racked_data['Session']==sess),'Direction'] = 'Ascending'
        else:
            all_racked_data.loc[(all_racked_data['SubjectID']==subj) & (all_racked_data['Session']==sess),'Direction'] = 'Descending'
```


```python
print(all_racked_data)
```

           index  SubjectID  Session  SessionCode  Trial  ProgFI  ObtFI  Latency  \
    0          0         21        7            1      1    5.00   7.09     4.70   
    1          1         21        7            1      2    5.00   6.04     1.93   
    2          2         21        7            1      3    5.00   5.40     0.92   
    3          3         21        7            1      4    5.00   7.94     0.92   
    4          4         21        7            1      5    5.00   5.90     1.04   
    ...      ...        ...      ...          ...    ...     ...    ...      ...   
    81064     56        181      193            4     57   22.94  23.91     1.22   
    81065     57        181      193            4     58   31.46  31.54     1.87   
    81066     58        181      193            4     59   44.73  44.95     3.82   
    81067     59        181      193            4     60   61.46  61.51     4.20   
    81068     60        181      193            4     61   80.00  80.04     6.47   
    
           Breakpoint  Direction  
    0            4.70     Static  
    1            3.65     Static  
    2            2.37     Static  
    3            3.09     Static  
    4            1.59     Static  
    ...           ...        ...  
    81064       16.43  Ascending  
    81065        2.18  Ascending  
    81066        5.92  Ascending  
    81067       13.83  Ascending  
    81068       11.71  Ascending  
    
    [185989 rows x 10 columns]
    


```python
#Calculate response rate/bin for each FI
fi_rr = all_rr_base.LogRate.groupby([all_rr_base['SubjectID'],all_rr_base['Group'],all_rr_base['ProgFI'],all_rr_base['Bin']]).mean()
fi_rr = fi_rr.reset_index()
fi_rr['ExpRate'] = np.exp(fi_rr.LogRate) - 0.001
fi_rr['UnBin'] = (fi_rr.Bin+1)*(fi_rr.ProgFI*(1/8))
fi_rr['PropBin'] = fi_rr.Bin/7
for subj in fi_rr.SubjectID.unique():
    for FI in fi_rr.ProgFI[fi_rr['SubjectID']==subj].unique():
        fi_rr.loc[(fi_rr['SubjectID']==subj) & (fi_rr['ProgFI']==FI), 'PropRate'] = fi_rr.ExpRate[(fi_rr['SubjectID']==subj) & (fi_rr['ProgFI']==FI)]/np.max(fi_rr.ExpRate[(fi_rr['SubjectID']==subj) & (fi_rr['ProgFI']==FI)])

```


```python
racked_base20_base40 = tracked_data_e1[(tracked_data_e1['Session']<=63) & (tracked_data_e1['Session']>=51)]
#find all Base20 mice
racked_base20_base40.loc[(racked_base20_base40['SubjectID']==21) | (racked_base20_base40['SubjectID']==23) | (racked_base20_base40['SubjectID']==26) | (racked_base20_base40['SubjectID']==28) | (racked_base20_base40['SubjectID']==30) | (racked_base20_base40['SubjectID']==32) | (racked_base20_base40['SubjectID']==33) | (racked_base20_base40['SubjectID']==35),'Group']='Base20'
racked_base20_base40.loc[(racked_base20_base40['SubjectID']==22) | (racked_base20_base40['SubjectID']==24) | (racked_base20_base40['SubjectID']==25) | (racked_base20_base40['SubjectID']==27) | (racked_base20_base40['SubjectID']==29) | (racked_base20_base40['SubjectID']==31) | (racked_base20_base40['SubjectID']==34) | (racked_base20_base40['SubjectID']==36),'Group']='Base40'
#Replace session 50 with 63 because of operant chamber malfunction
racked_base20_base40[(racked_base20_base40['SubjectID']>=29) & (racked_base20_base40['Session']==50)] = racked_base20_base40[(racked_base20_base40['SubjectID']>=29) & (racked_base20_base40['Session']==63)]
#Drop session 63 because not needed for baseline
racked_base20_base40 = racked_base20_base40[racked_base20_base40['Session']<=62]
```


```python
#Organize baseline of e2 data (Base 60)
racked_base60 = tracked_data_e2[(tracked_data_e2['Session']<=62) & (tracked_data_e2['Session']>= 51)]
#Add group identifier
racked_base60['Group' ] = 'Base60'
```


```python
#recombine data again
all_racked_base = pd.concat([racked_base20_base40,racked_base60],axis=0)
all_racked_base['LogLat'] = np.log(all_racked_base.Latency)
all_racked_base['LogBP'] = np.log(all_racked_base.Breakpoint)
```


```python
for subj in all_racked_base.SubjectID.unique():
    for sess in all_racked_base[all_racked_base['SubjectID']==subj].Session.unique():
        if all_racked_base[(all_racked_base['SubjectID']==subj) & (all_racked_base['Session']==sess)].ProgFI.to_numpy()[0] - all_racked_base[(all_racked_base['SubjectID']==subj) & (all_racked_base['Session']==sess)].ProgFI.to_numpy()[1] == 0:
            all_racked_base.loc[(all_racked_base['SubjectID']==subj) & (all_racked_base['Session']==sess),'Direction'] = 'Static'
        elif all_racked_base[(all_racked_base['SubjectID']==subj) & (all_racked_base['Session']==sess)].ProgFI.to_numpy()[0] - all_racked_base[(all_racked_base['SubjectID']==subj) & (all_racked_base['Session']==sess)].ProgFI.to_numpy()[1] < 0:
            all_racked_base.loc[(all_racked_base['SubjectID']==subj) & (all_racked_base['Session']==sess),'Direction'] = 'Ascending'
        else:
            all_racked_base.loc[(all_racked_base['SubjectID']==subj) & (all_racked_base['Session']==sess),'Direction'] = 'Descending'
```


```python
experimental_data = all_racked_base
for subj in experimental_data.SubjectID.unique():
    for sess in experimental_data[experimental_data['SubjectID']==subj].Session.unique():
        Progtemp = experimental_data[(experimental_data['SubjectID']==subj) & (experimental_data['Session']==sess)].ProgFI.to_numpy()
        BPtemp = experimental_data[(experimental_data['SubjectID']==subj) & (experimental_data['Session']==sess)].Breakpoint.to_numpy()
        Lattemp = experimental_data[(experimental_data['SubjectID']==subj) & (experimental_data['Session']==sess)].Latency.to_numpy()
        if Progtemp[0] - Progtemp[1] > 0:
            experimental_data.loc[(experimental_data['SubjectID']==subj) & (experimental_data['Session']==sess),'ProgFI']= np.flip(Progtemp)
            experimental_data.loc[(experimental_data['SubjectID']==subj) & (experimental_data['Session']==sess),'Breakpoint'] = np.flip(BPtemp)
            experimental_data.loc[(experimental_data['SubjectID']==subj) & (experimental_data['Session']==sess),'Latency'] = np.flip(Lattemp)
print(experimental_data)
```

           index  SubjectID  Session  SessionCode  Trial  ProgFI  ObtFI  Latency  \
    42944      0         21       51            4      1   20.00  20.09     3.06   
    42945      1         21       51            4      2   24.64  24.83     0.51   
    42946      2         21       51            4      3   28.82  29.07     5.40   
    42947      3         21       51            4      4   32.14  32.30     4.43   
    42948      4         21       51            4      5   34.27  34.74     4.40   
    ...      ...        ...      ...          ...    ...     ...    ...      ...   
    25249     56        184       62            4     57   17.20  17.59     3.48   
    25250     57        184       62            4     58   23.59  24.08     3.18   
    25251     58        184       62            4     59   33.55  33.77     4.86   
    25252     59        184       62            4     60   46.09  46.13    15.00   
    25253     60        184       62            4     61   60.00  60.02    31.50   
    
           Breakpoint   Group    LogLat     LogBP  Direction  
    42944        8.74  Base20  1.118415  2.167910  Ascending  
    42945        5.11  Base20 -0.673345  1.631199  Ascending  
    42946        9.01  Base20  1.686399  2.198335  Ascending  
    42947       17.08  Base20  1.488400  2.837908  Ascending  
    42948       11.47  Base20  1.481605  2.439735  Ascending  
    ...           ...     ...       ...       ...        ...  
    25249        8.71  Base60  1.247032  2.164472  Ascending  
    25250        6.90  Base60  1.156881  1.931521  Ascending  
    25251       19.21  Base60  1.581038  2.955431  Ascending  
    25252       15.00  Base60  2.708050  2.708050  Ascending  
    25253       31.50  Base60  3.449988  3.449988  Ascending  
    
    [17568 rows x 13 columns]
    


```python
experimental_data['NormBP'] = experimental_data.Breakpoint
experimental_data['NormFI'] = experimental_data.ProgFI
baseFI_means = experimental_data.Breakpoint.groupby([experimental_data['Group'],experimental_data['SubjectID'],experimental_data['Trial']]).mean().reset_index()
for subj in experimental_data.SubjectID.unique():
    #norm BPs first
    temp = experimental_data.Breakpoint[experimental_data['SubjectID']==subj]
    temp2 = baseFI_means.Breakpoint[baseFI_means['SubjectID']==subj].iloc[0]
    temp3 = temp/temp2
    experimental_data.loc[experimental_data['SubjectID']==subj,'NormBP'] = temp3
    
    #norm ProgFI second
    temp = experimental_data.ProgFI[experimental_data['SubjectID']==subj]
    temp2 = temp/temp.iloc[0]
    experimental_data.loc[experimental_data['SubjectID']==subj,'NormFI'] = temp2
print(experimental_data)
```

           index  SubjectID  Session  SessionCode  Trial  ProgFI  ObtFI  Latency  \
    42944      0         21       51            4      1   20.00  20.09     3.06   
    42945      1         21       51            4      2   24.64  24.83     0.51   
    42946      2         21       51            4      3   28.82  29.07     5.40   
    42947      3         21       51            4      4   32.14  32.30     4.43   
    42948      4         21       51            4      5   34.27  34.74     4.40   
    ...      ...        ...      ...          ...    ...     ...    ...      ...   
    25249     56        184       62            4     57   17.20  17.59     3.48   
    25250     57        184       62            4     58   23.59  24.08     3.18   
    25251     58        184       62            4     59   33.55  33.77     4.86   
    25252     59        184       62            4     60   46.09  46.13    15.00   
    25253     60        184       62            4     61   60.00  60.02    31.50   
    
           Breakpoint   Group    LogLat     LogBP  Direction    NormBP    NormFI  
    42944        8.74  Base20  1.118415  2.167910  Ascending  0.994878  1.000000  
    42945        5.11  Base20 -0.673345  1.631199  Ascending  0.581673  1.232000  
    42946        9.01  Base20  1.686399  2.198335  Ascending  1.025612  1.441000  
    42947       17.08  Base20  1.488400  2.837908  Ascending  1.944223  1.607000  
    42948       11.47  Base20  1.481605  2.439735  Ascending  1.305635  1.713500  
    ...           ...     ...       ...       ...        ...       ...       ...  
    25249        8.71  Base60  1.247032  2.164472  Ascending  0.312149  0.286667  
    25250        6.90  Base60  1.156881  1.931521  Ascending  0.247282  0.393167  
    25251       19.21  Base60  1.581038  2.955431  Ascending  0.688448  0.559167  
    25252       15.00  Base60  2.708050  2.708050  Ascending  0.537570  0.768167  
    25253       31.50  Base60  3.449988  3.449988  Ascending  1.128897  1.000000  
    
    [17568 rows x 15 columns]
    

First we want to know whether the mean breakpoint as a function of trial in the session is scalar invariant. The graphs below are a visual test. Actual test will be conducted in R

***UPDATE WITH TEST***


```python
fig1 = plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
sns.lineplot(data=experimental_data,x="Trial",y="Breakpoint",style="Group", markers = True, color="tab:blue",markersize=8,linewidth=2,legend=None)
plt.ylabel("Breakpoint (s)")
plt.twinx()
sns.lineplot(data=experimental_data,x="Trial",y="ProgFI",style="Group", color="tab:orange",legend=None)
plt.ylabel("PRogrammed FI (S)")
plt.subplot(1,2,2)
sns.lineplot(data=experimental_data,x="Trial",y="NormBP",style="Group", markers = True, color="tab:blue",markersize=8,linewidth=2)
plt.legend(fontsize=18,loc="best")
plt.ylabel("Normalized Breakpoint")
plt.ylim(0,2.5)
plt.twinx()
sns.lineplot(data=experimental_data,x="Trial",y="NormFI",style="Group", color="tab:orange",legend=None)
plt.ylabel("Programmed FI / Base FI")
plt.ylim(0,2.5)
plt.tight_layout()
plt.show()
fig1.savefig("CyclicFig1.png")
```


    
![png](output_21_0.png)
    



```python
#Calculate response rate/bin for each FI
fi_rr = all_rr_base.LogRate.groupby([all_rr_base['SubjectID'],all_rr_base['Group'],all_rr_base['ProgFI'],all_rr_base['Bin']]).mean()
fi_rr = fi_rr.reset_index()
fi_rr['ExpRate'] = np.exp(fi_rr.LogRate) - 0.001
fi_rr['UnBin'] = (fi_rr.Bin+1)*(fi_rr.ProgFI*(1/8))
fi_rr['PropBin'] = fi_rr.Bin/7
for subj in fi_rr.SubjectID.unique():
    for FI in fi_rr.ProgFI[fi_rr['SubjectID']==subj].unique():
        fi_rr.loc[(fi_rr['SubjectID']==subj) & (fi_rr['ProgFI']==FI), 'PropRate'] = fi_rr.ExpRate[(fi_rr['SubjectID']==subj) & (fi_rr['ProgFI']==FI)]/np.max(fi_rr.ExpRate[(fi_rr['SubjectID']==subj) & (fi_rr['ProgFI']==FI)])

```

Next we asked if the scalar invariance observed in the previous graphs would be observed in the response function where response rates are plotted as a function of time for each FI in the sine wave. 


```python
fig2 = plt.figure(figsize=(16,18),tight_layout=True)
for i,group in enumerate(fi_rr.Group.unique(),start=1):
    temp = fi_rr[fi_rr['Group']==group]
    plt.subplot(3,1,i)
    plt.rc('font',size=24)
    sns.lineplot(data=temp,x='UnBin',y='ExpRate',hue='ProgFI',palette="colorblind",markers=True,dashes=False,marker='o',ci=68,markersize=8,linewidth=2)
    plt.ylim([0,5])
    plt.xlim(0,105)
    plt.xlabel('Time in FI (s)')
    plt.ylabel('Responses/second')
    plt.legend(fontsize=18)
plt.show()
fig2.savefig('CyclicFig2.png')
```


    
![png](output_24_0.png)
    



```python
fig3 = plt.figure(figsize=(16,18),tight_layout=True)
for i,group in enumerate(fi_rr.Group.unique(),start=1):
    temp = fi_rr[fi_rr['Group']==group]
    plt.subplot(3,1,i)
    plt.rc('font',size=24)
    sns.lineplot(data=temp,x='PropBin',y='PropRate',hue='ProgFI',palette="colorblind",markers=True,dashes=False,marker='o',ci=68,markersize=8,linewidth=2)
    plt.ylim([0,1])
    plt.xlim(0,1)
    plt.xlabel('Normalized FI')
    plt.ylabel('Normalized Response Rate')
    plt.legend(fontsize=18)
plt.show()
fig3.savefig('CyclicFig3.png')
```


    
![png](output_25_0.png)
    


Here we see clear violations of the scalar property. Rather than quantify these response functions, which are fairly noisy for the relatively short intervals, we decided to look at breakpoints as a function of FI. Thus providing a parallel analysis. 


```python
#cv as a function of interval (for breakpoints)
subject_mean = all_racked_base.groupby([all_racked_base['SubjectID'],all_racked_base['Group'],all_racked_base['ProgFI']]).mean().reset_index()
subject_std = all_racked_base.groupby([all_racked_base['SubjectID'],all_racked_base['Group'],all_racked_base['ProgFI']]).std().reset_index()
subject_cv = pd.DataFrame(np.asfarray(subject_std.Breakpoint/subject_mean.Breakpoint),columns=['Breakpoint'])
subject_cv['SubjectID'] = subject_mean.SubjectID
subject_cv['Group'] = subject_mean.Group
subject_cv['ProgFI'] = subject_mean.ProgFI
```


```python
fig4 = plt.figure(figsize=(24,6),tight_layout=True)
for i,group in enumerate(subject_mean.Group.unique(),start=1):
    temp = fi_rr[fi_rr['Group']==group]
    temp_mean = subject_mean[subject_mean['Group']==group]
    temp_std = subject_std[subject_std['Group']==group]
    temp_cv = subject_cv[subject_cv['Group']==group]
    plt.subplot(1,3,i)
    if (i == 2) or (i == 3):
        leg_on = None
    else:
        leg_on = 'brief'
    plt.rc('font',size=24)
    sns.lineplot(data=temp_mean,x='ProgFI',y='Breakpoint',color="tab:blue",markers=True,dashes=False,marker='o',ci=68,markersize=8,linewidth=2,legend=leg_on)
    sns.lineplot(data=temp_std,x='ProgFI',y='Breakpoint',color="tab:orange",markers=True,dashes=False,marker='o',ci=68,markersize=8,linewidth=2,legend=leg_on)
    plt.xlabel('Fixed Interval (s)')
    plt.ylabel('Mean or STD (s)')
    if i == 1:
        plt.legend(['Mean','STD'],loc="upper right")
    plt.ylim([0,50])
    plt.twinx()
    sns.lineplot(data=temp_cv,x='ProgFI',y='Breakpoint',color="black",markers=True,dashes=False,marker='o',ci=68,markersize=8,linewidth=2,legend=leg_on)
    plt.ylim([0,0.8])
    plt.xlim([5,105])
    plt.ylabel('CV')
    if i == 1:
        plt.legend(['CV'],loc="lower right")
plt.show()
fig4.savefig('CyclicFig4.png')
```


    
![png](output_28_0.png)
    



```python
fig5 = plt.figure(figsize=(6,6),tight_layout=True)
plt.rc('font',size=24)
sns.lineplot(data=subject_mean,x='ProgFI',y='Breakpoint',style="Group",color="tab:blue",markers=True,dashes=False,marker='o',ci=68,markersize=8,linewidth=2, legend=None)
sns.lineplot(data=subject_std,x='ProgFI',y='Breakpoint',style="Group",color="tab:orange",markers=True,dashes=False,marker='o',ci=68,markersize=8,linewidth=2, legend=None)
plt.xlabel('Fixed Interval (s)')
plt.ylabel('Mean or STD (s)')
plt.ylim([0,50])
plt.xlim([5,105])
plt.twinx()
sns.lineplot(data=subject_cv,x='ProgFI',y='Breakpoint',style="Group",color="black",markers=True,dashes=False,marker='o',ci=68,markersize=8,linewidth=2, legend=None)
plt.ylim([0,0.8])
plt.ylabel('CV')
plt.show()
fig5.savefig('CyclicFig5.png')
```


    
![png](output_29_0.png)
    


Here we see clearly that the CV is not constant across FIs. Indicating that despite the mean breakpoint appearing scalar invariant, it is in fact not. Normalized variance appears to take on an inverted U-shape. 

***UPDATE WITH TEST***


```python

```
