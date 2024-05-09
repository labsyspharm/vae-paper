# mcmicro-scanpy
An MCMICRO module implementation of scanpy for clustering cells using the Leiden algorithm.

Example usage:
```
docker run --rm -v "$PWD":/data labsyspharm/mc-scanpy:1.0.1 python3 /app/cluster.py -i /data/unmicst-exemplar-001.csv -o /data/ -c
```

## Output Files
- `cells.csv` contains the cluster assignment for each cell
- `clusters.csv` contains each clusters' mean values for every feature 
(if the max feature value is >1000 then the values will be log transformed for clustering and remain transformed in this output file)

## Paramenter Reference
```
usage: cluster.py [-h] -i INPUT [-o OUTPUT] [-m MARKERS] [-k NEIGHBORS] [-c] [-y CONFIG] [--force-transform] [--no-transform]

Cluster cell types using mcmicro marker expression data.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input CSV of mcmicro marker expression data for cells
  -o OUTPUT, --output OUTPUT
                        The directory to which output files will be saved
  -m MARKERS, --markers MARKERS
                        A text file with a marker on each line to specify which markers to use for clustering
  -k NEIGHBORS, --neighbors NEIGHBORS
                        the number of nearest neighbors to use when clustering. The default is 30.
  -c, --method          Include a column with the method name in the output files.
  -y CONFIG, --config CONFIG
                        A yaml config file that states whether the input data should be log/logicle transformed.
  --force-transform     Log transform the input data. If omitted, and --no-transform is omitted, log transform is only performed if the max value in the input data is >1000.
  --no-transform        Do not perform Log transformation on the input data. If omitted, and --force-transform is omitted, log transform is only performed if the max value in the input data is >1000.
```
