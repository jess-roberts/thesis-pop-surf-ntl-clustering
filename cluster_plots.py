import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import os

def findCSVs(csvDir,csvName):
    # criteria to find the .csvs
    search_criteria = str(csvName)+'*'
    CSVs = os.path.join(csvDir, search_criteria)
    # taking all the files that match and globbing together
    all_CSVs = glob.glob(CSVs)
    print(len(all_CSVs),'.csv files found')
    return all_CSVs

CSVs = findCSVs('./','output_cluster_')
sorted_CSVs = sorted(CSVs, key=lambda x: int(x[-7:-4]))
print(sorted_CSVs)

# Setting up figure plot parameters
tfont = {'fontname':'Charter', 'fontweight':'bold','pad':'2'}
indexes = [231,232,234,235,236]
classes = ['A','B','C','D','E']
x_params = [False, False, True, True, True]
y_params = [True, True, False, False, False]

# Open plot
fig = plt.figure(figsize=(8,6))
for csv, index, classs, xp, yp in zip(sorted_CSVs,indexes,classes,x_params,y_params):
    names=['POP','NTL']
    color = ['red','blue']
    # Read in cluster data
    df1 = pd.read_csv(csv,names=names)
    # Add subplot
    ax = fig.add_subplot(index)
    # subplot formatting
    ax.set_title(classs, **tfont)
    ax.set_yscale('log')
    ax.set_ylim(1,10000000)
    # Fill subplot with histogram data
    df1.plot.hist(ax=plt.gca(),bins=255,alpha=0.5,histtype='step', color=color,range=(0,255),legend=False)
    # Change font
    ax.set(ylabel=None)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Charter') for label in labels]

    index += 1

# Figure formatting
fig.text(0.5, 0.02, 'Value Distribution', ha='center', fontname='Charter',fontsize=10)
fig.text(0.04, 0.5, 'Frequency (log 10)', va='center', rotation='vertical', fontname='Charter', fontsize=10)
plt.subplots_adjust(wspace=0.3,hspace=0.5)
mpl.rc('font',family='Charter')
plt.legend()
# Export
plt.savefig('./cluster_data_plots.png',dpi=500,transparent=True)
print('Cluster plot figure saved')
plt.close()