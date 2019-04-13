#from os import listdir
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

#from mpl_toolkits.mplot3d import axes3d

from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")
#from Pfeature.Pfeature import aac_wp,atc_wp
'''
path = "./data/"
input_file = listdir(path)
'''


std = list("ACDEFGHIKLMNPQRSTVWY")

def aac_wp(file,out):
    filename, file_extension = os.path.splitext(file)
    f = open(out, 'w')
    #sys.stdout = f
    df1 = pd.read_csv(file, header = None)
    df = pd.DataFrame(df1[0].str.upper())	
    zz = df.iloc[:,0]
    f.write("A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y,\n")
    for j in zz:
        line = ""
        for i in std:
            count = 0
            for k in j:
                temp1 = k
                if temp1 == i:
                    count += 1
                composition = (count/len(j))*100
            line +=("%.2f"%composition+",")
        f.write(line+"\n")
    f.truncate()
    
def plotgraph(resultlist,title=""):
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    '''
    plt.figure()
    for i in sorted(resultlist.keys()):
        '''
        x = resultlist[i].T[0]
        y = resultlist[i].T[1]
        z = resultlist[i].T[2]
        
        ax.scatter(x,y,z,label='Cluster ' + str(i))
        '''
        plt.scatter(resultlist[i].T[0],resultlist[i].T[1],label='Cluster ' + str(i))
        #plt.show()
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig("./figures/"+title)
    plt.show()

'''
for file in input_file:
    aac_wp(path+file,'./out/'+file)

sys.stdout = oldstdout
'''
def completefun(path,title="",numclusters=3):
    datapoints = np.array(pd.read_csv(path))
    print ("Kmeans Clustering Running...")
    kmeans = KMeans(n_clusters=numclusters, random_state=0).fit(datapoints)
    
    tnsepoints = TSNE(n_components=2).fit_transform(datapoints)
    clusters = {}
    for i in range(len(kmeans.labels_)):
        if(kmeans.labels_[i] not in clusters):
            clusters[kmeans.labels_[i]] = []
        clusters[kmeans.labels_[i]].append(i)
    for i in clusters.keys():
        clusters[i] = tnsepoints[clusters[i]]
    plotgraph(clusters,"Kmeans_"+title)
    
    
    print ("Hierarchal Clustering Running...")
    #datapoints = np.array(pd.read_csv(path))
    clustering = AgglomerativeClustering(n_clusters=numclusters).fit(datapoints)
    #datapoints = TSNE(n_components=3).fit_transform(datapoints)
    clusters = {}
    for i in range(len(clustering.labels_)):
        if(clustering.labels_[i] not in clusters):
            clusters[clustering.labels_[i]] = []
        clusters[clustering.labels_[i]].append(i)
    for i in clusters.keys():
        clusters[i] = tnsepoints[clusters[i]]
    plotgraph(clusters,"Hierarchal_"+title)
    print ("Clusters using Kmeans: %d\nClusters using Hierarchical: %d"%(len(set(kmeans.labels_)),len(set(clustering.labels_))))

num_cluster = int(input("Enter Number of Clusters for K-Means (AAC Data): "))
print ("Running on AAC Composition...")
completefun("./aacinput/final_amino_acid_result.csv"," AAC",numclusters=num_cluster)
num_cluster = int(input("Enter Number of Clusters for K-Means (ATC Data): "))
print ("Running on ATC Composition...")
completefun("./atomiccomposition/final_amino_acid_result.csv"," ATC",numclusters=num_cluster)
#    
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#x = datapoints.T[0]
#y = datapoints.T[1]
#z = datapoints.T[2]
#ax.scatter(x, y, z, cmap='viridis', linewidth=0.5);
    
print ("Clusters outputs generated in ./figures directory to see outputs copy images using docker cp command")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
