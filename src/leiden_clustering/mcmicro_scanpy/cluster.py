import re
import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

'''
Parse arguments.
Input file is required.
'''
def parseArgs():
    parser = argparse.ArgumentParser(description='Cluster cell types using latent vectors of pixel image patches.')
    parser.add_argument('-i', '--input', help="Input CSV of latent vector data for cells", type=str, required=True)
    parser.add_argument('-o', '--output', help='The directory to which output files will be saved', type=str, required=False)
    parser.add_argument('-k', '--neighbors', help='the number of nearest neighbors to use when clustering. The default is 30.', default=30, type=int, required=False)
    parser.add_argument('-r', '--resolution', help='the resolution controls the coarseness of the clustering. Higher values lead to more clusters. The default is 1.0.', default=1.0, type=float, required=False)
    parser.add_argument('-c', '--method', help='Include a column with the method name in the output files.', action="store_true", required=False)
    parser.add_argument('-y', '--config', help='A yaml config file that states whether the input data should be log/logicle transformed.', type=str, required=False)
    parser.add_argument('--force-transform', help='Log transform the input data. If omitted, and --no-transform is omitted, log transform is only performed if the max value in the input data is >1000.', action='store_true', required=False)
    parser.add_argument('--no-transform', help='Do not perform Log transformation on the input data. If omitted, and --force-transform is omitted, log transform is only performed if the max value in the input data is >1000.', action='store_true', required=False)
    args = parser.parse_args()
    return args


'''
Get input data file name
'''
def getDataName(path):
    fileName = path.split('/')[-1] # get filename from end of input path
    dataName = fileName[:fileName.rfind('.')] # get data name by removing extension from file name
    return dataName

'''
Write PATCHES_FILE from leidenCluster() adata
'''
def writePatches(adata):
    print("Writing Patches...")
    patches = pd.DataFrame(adata.obs[PATCH_ID].astype(int)) # extract patch IDs to dataframe
    patches[CLUSTER] = adata.obs[LEIDEN] # extract and add cluster assignments to patches dataframe

    # add in method column if requested
    if args.method:
        patches[METHOD] = SCANPY

    patches.to_csv(f'{output}/{patches_file}', index=False)

'''
Write CLUSTERS_FILE from leidenCluster() adata
'''
def writeClusters(adata):
    print("Writing Clusters...")
    clusters = pd.DataFrame(columns=adata.var_names, index=adata.obs[LEIDEN].cat.categories)   
    clusters.index.name = CLUSTER # name indices as cluster column
    for cluster in adata.obs.leiden.cat.categories: # this assumes that LEIDEN = 'leiden' if the name is changed, replace it for 'leiden' in this line
        clusters.loc[cluster] = adata[adata.obs[LEIDEN].isin([cluster]),:].X.mean(0)
    
    # add in method column if requested
    if args.method:
        clusters[METHOD] = SCANPY

    clusters.to_csv(f'{output}/{clusters_file}')

'''
Get max value in dataframe.
'''
def getMax(df):
    return max([n for n in df.max(axis = 0)])

'''
Cluster data using the Leiden algorithm via scanpy
'''

def leidenCluster(input_file):
    print("Starting leidenCluster()...")

    data = pd.read_csv(input_file)
    columns = list(data.columns)
    if columns.index(PATCH_ID) != 0: # if patch ID column is included but not first, move it to the front
        columns.insert(0, columns.pop(columns.index(PATCH_ID)))
    data = data[columns]
    # save cleaned data to csv
    data.to_csv(f'{output}/{clean_data_file}', index=False)

    sc.settings.verbosity = 3 # print out information
    print(f'{output}/{clean_data_file}')
    adata_init = sc.read(f'{output}/{clean_data_file}', cache=True) # load input data

    # move PATCH_ID info into .obs
    # this assumes that 'PATCH_ID' is the first column in the csv
    adata_init.obs[PATCH_ID] = adata_init.X[:,0]
    adata = ad.AnnData(np.delete(adata_init.X, 0, 1), obs=adata_init.obs, var=adata_init.var.drop([PATCH_ID]))
    
    print("Started writing config")
    # log transform the data according to parameter. If 'auto,' transform only if the max value >1000. Don't do anything if transform == 'false'. Write transform decision to yaml file.
    if transform == 'true':
        sc.pp.log1p(adata, base=10)
        writeConfig(True)
    elif transform == 'auto' and getMax(adata.X) > 1000:
        sc.pp.log1p(adata, base=10)
        writeConfig(True)
    else:
        writeConfig(False)
    print("Finished writing config")
    
    # compute neighbors and cluster
    sc.pp.neighbors(adata, n_neighbors=args.neighbors, n_pcs=50, use_rep='X') # compute neighbors, using the first 10 principle components and the number of neighbors provided in the command line. Default is 30.
    sc.tl.leiden(adata, key_added = LEIDEN, resolution=args.resolution) # run leidan clustering. default resolution is 1.0
    print("Finished leiden clustering")
    
    # write patch/cluster information to 'PATCHES_FILE'
    writePatches(adata)

    # write cluster mean feature expression to 'CLUSTERS_FILE'
    writeClusters(adata)

'''
Write to a yaml file whether the data was transformed or not.
'''
def writeConfig(transformed):
    qcExists = os.path.exists('qc')
    if not qcExists: 
        os.mkdir('qc')
    with open('qc/config.yml', 'a') as f:
        f.write('---\n')
        if transformed:
            f.write('transform: true')
        else:
            f.write('transform: false')

'''
Read config.yml file contents.
'''
def readConfig(file):
    f = open(file, 'r')
    lines = f.readlines()

    # find line with 'transform:' in it
    for l in lines:
        if 'transform:' in l.strip():
            transform = l.split(':')[-1].strip() # get last value after colon

    return transform

'''
Main.
'''
if __name__ == '__main__':
    args = parseArgs() # parse arguments

    # get user-defined output dir (strip last slash if present) or set to current
    if args.output is None:
        output = '.'
    elif args.output[-1] == '/':
        output = args.output[:-1]
    else:
        output = args.output

    # assess log transform parameter
    if args.force_transform and not args.no_transform:
        transform = 'true'
    elif not args.force_transform and args.no_transform:
        transform = 'false'
    elif args.config is not None:
        transform = readConfig(args.config)
    else:
        transform = 'auto'

    # constants
    PATCH_ID = 'CellID' # column name holding patch IDs
    CLUSTER = 'Cluster' # column name holding cluster number
    LEIDEN = 'leiden' # obs name for cluster assignment
    METHOD = 'Method' # name of column containing the method for clustering
    SCANPY = 'Scanpy' # the name of this method
    
    # output file names
    data_prefix = getDataName(args.input) # get the name of the input data file to add as a prefix to the output file names
    clean_data_file = f'{data_prefix}-clean.csv' # name of output cleaned data CSV file
    clusters_file = f'{data_prefix}-clusters.csv' # name of output CSV file that contains the mean expression of each latent vector, for each cluster
    patches_file = f'{data_prefix}-patches.csv' # name of output CSV file that contains each patch ID and it's assigned cluster
    

    print("Entering Leiden function...")

    # cluster using scanpy implementation of Leiden algorithm
    leidenCluster(args.input)
