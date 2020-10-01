#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
from sklearn import datasets
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
import gdal
import osr

def rast2array(filename):
    '''
    Read a geotiff in to RAM
    '''
    # open a dataset object
    ds=gdal.Open(str(filename))
    # could use gdal.Warp to reproject if wanted?

    # read data from geotiff object
    nX=ds.RasterXSize             # number of pixels in x direction
    nY=ds.RasterYSize             # number of pixels in y direction
    # read data. Returns as a 2D numpy array
    red_data = ds.GetRasterBand(1).ReadAsArray(0,0,nX,nY)
    green_data = ds.GetRasterBand(2).ReadAsArray(0,0,nX,nY)
    blue_data = ds.GetRasterBand(3).ReadAsArray(0,0,nX,nY)
    # extract spatial parameters
    transform_ds = ds.GetGeoTransform() # extract geolocation information
    xOrigin=transform_ds[0]       # coordinate of x corner
    yOrigin=transform_ds[3]       # coordinate of y corner
    pixelWidth=transform_ds[1]    # resolution in x direction
    pixelHeight=transform_ds[5]   # resolution in y direction
    ds=None
    return red_data, green_data, blue_data, nX, nY, xOrigin, yOrigin, pixelWidth, pixelHeight

def writeTiff(data,nX, nY, xOrigin, yOrigin, pixelWidth, pixelHeight, dest):
        """
            Create output raster
        """
        # set geolocation information
        geotransform = (xOrigin, pixelWidth, 0, yOrigin, 0, pixelHeight)
        # load data in to geotiff object
        dst_ds = gdal.GetDriverByName('GTiff').Create(dest, nX, nY, 1, gdal.GDT_Float32)
        dst_ds.SetGeoTransform(geotransform)    # specify coords
        srs = osr.SpatialReference()            # establish encoding
        srs.ImportFromEPSG(3857)                # encoding the EPSG
        dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
        dst_ds.GetRasterBand(1).SetNoDataValue(np.nan)  # set no data value
        dst_ds.GetRasterBand(1).WriteArray(data)  # write red image to the raster
        dst_ds.FlushCache()                     # write to disk
        dst_ds = None
        print('Image written to',dest)

def removeNaN(array_in):
    """
        Function to remove NaN values from array
    """
    nan_array = np.isnan(array_in)
    not_nan_array = ~nan_array 
    array_out = array_in[not_nan_array]
    return array_out

# Import the 3 arrays (x=POP, y=SURF, z=NTL)
x1, y2, z3, nX, nY, xOrigin, yOrigin, pixelWidth, pixelHeight  = rast2array('../../RGB_output/malawi_RGB_data_only.tif')
x1_flat = x1.flatten()
y2_flat = y2.flatten()
z3_flat = z3.flatten()
print('Original size:',len(x1_flat))

# Remove no data (NaN) pixels from the arrays
print('-- Removing no data --')
x1_data = removeNaN(x1)
y2_data = removeNaN(y2)
z3_data = removeNaN(z3)
print('New size:',len(x1_data))

# Combine the 3 bands to create an array of (x=POP, y=SURF, z=NTL) pixels
print('-- Combining all arrays --')
X = np.array(list(zip(x1_data, y2_data, z3_data))).reshape(len(x1_data), 3) # all arrays combined 

# Set up variables to separate these pixels according to their y(SURF) value
nbua_list = []
bua_list = []
bad_list = []
X_class = x1_data.copy()
print('-- Separating data according to binary layer --')
i = -1
for pixs in X:
    i += 1
    if pixs[1] == 0:
    # Classed as not built up area
        # Separate
        nbua_list.append([pixs[0],pixs[2]])
        # Reclass raster array as 100
        X_class[i] = 100
    
    elif pixs[1] in (254,255):
    # Classed as built up area
        # Separate
        bua_list.append([pixs[0],pixs[2]])
        # Reclass raster array as 200
        X_class[i] = 200
    else:
        bad_list.append([pixs[0],pixs[1],pixs[2]])

# Check to see if any invalid values were thrown up
if len(bad_list) > 1:
    print('You got a problem with your surface data boss')
    print(bad_list)

print(len(nbua_list),'non-built up pixels found')
print(len(bua_list),'built-up pixels found')

# Dict of options
# [0] = data_sample, [1] = data, [2] = class assigned, [3] = name to use
opt_dict = {'Built_Up_Area': [bua_list[::10].copy(), bua_list.copy(), 200, 'Built Up Area'],
            'Non-Built_Up_Area': [nbua_list[::100].copy(), nbua_list.copy(), 100, 'Not Built Up Area']}
fargs = {'fontname':'Charter'}

j = 0
# Run teh following on both Buit Up (surf) and not built up (not surf)
for opt in opt_dict.keys():
    
    # Select the relevant surf/nosurf parameters to use
    data_sample = opt_dict[opt][0]
    data = opt_dict[opt][1]
    classed = opt_dict[opt][2]
    name = opt_dict[opt][3]

    # Take data sample
    X_sample = np.array(data_sample).reshape(len(data_sample),2)
    X_use = np.array(data).reshape(len(data),2)

    # Set up range up clusters to test
    range_n_clusters = np.arange(2,10,1)

    clusters_nums = []
    cluster_avgs = []
    print('-- Performing clustering on',opt,'pixels --')
    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=100)
        cluster_labels = clusterer.fit_predict(X_sample)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X_sample, cluster_labels)
        print("For n_clusters =", n_clusters,
            "the average silhouette_score is :", silhouette_avg)
        # Store the silhouette score per cluster
        clusters_nums.append(n_clusters)
        cluster_avgs.append(silhouette_avg)
    
    # Create figure of average silhouette scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(clusters_nums,cluster_avgs,c='r')
    plt.title(name,**fargs)
    plt.ylabel('Average Silhouette Scores',**fargs)
    plt.xlabel('Clusters (k)',**fargs)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Charter') for label in labels]
    plt.savefig('./avg_silhouette_'+str(opt)+'.png',dpi=300)
    plt.close()

    ####### PERFORMING CLUSTERING #######
    # Number of clusters k
    opt_clust = clusters_nums[cluster_avgs.index(max(cluster_avgs))]
    print('Optimal number of clusters is',opt_clust)
    # Clustering model to use
    model = KMeans(n_clusters=opt_clust, random_state=100)

    # Set up plot
    colors = np.array(["gold", "green", "blue", "magenta", "red"])
    fig2 = plt.figure(figsize=(3, 4))
    plt.xlabel('Population Density (POP)')
    plt.ylabel('Nightlights (NTL)')
    
    # Fit cluster model
    print('-- Fitting cluster model --')
    model.fit(X_use)

    # Labels is the cluster classification of the array 
    labels = model.labels_
    centers = model.cluster_centers_ 

    # Plot clustered data
    data = plt.scatter(X_use[:, 0], X_use[:, 1], c=colors[labels.astype(int)+j], alpha=0.5)
    centers_list = enumerate(centers)
    j =+ 3

    for i, c in enumerate(centers):
        # Centroid extraction
        clust_num = "Cluster number: "+str(i)
        clust_cent = " & centroid: ("+str(int(c[0]))+','+str(int(c[1]))+')'
        print(clust_num)
        print(clust_cent)

    # For each cluster (100 NOSURF or 200SURF) add the cluster number (i)
    # e.g. 100 NO SURF will have clusters with labels 100, 101, 102 
    # e.g. 200 SURF will have clusters with labels 200, 201, 202 
    label_types = [i+classed for i in labels] 
    # For every value classed as either 100 (no surf) or 200 (surf) 
    # fill with the new class labels (100, 101, 201 etc)
    X_class[np.where(X_class == classed)] = label_types

    # Write out clustered plot figure
    plt.title('POP-NTL Clusters for '+str(opt))
    plt.savefig('./cluster_RB_'+str(opt)+'_output.png',dpi=300)
    print('Cluster figure saved')
    plt.tight_layout()
    plt.close()


###### Write out clustered geotiff #####
# Copy original shape of geotiff array
arr_2_fill = x1.copy()
print('-- Filling raster with clustered values --')
# Where the geotiff has data, fill with the new classed data
arr_2_fill[np.isnan(x1) == False] = X_class
# Write out newly classed geotiff
writeTiff(arr_2_fill, nX, nY, xOrigin, yOrigin, pixelWidth, pixelHeight, './clustered.tif')
print('Cluster geotiff saved')

