# QGCN

QGCN method for graph classification

required packages:
- pytorch (optional use of GPU)
- numpy

### How to use 
To use this package you will need to provide the following files as input
-> Graphs csv file - files that contains the graphs for input and their labels
  The format of the file is flexible, but it must contain headers for any columns, and the there must be a column provided for:
  - graph id
  - source node id
  - destination node id
  - label id (every graph id can be attached to only one label)
- External data file - external data for every node (Optional)
    The format of this file is also flexible, but it must contain headers for any columns, and the there must be a column provided for:
    **note!! every node must get a value**
    - graph id
    - node id
    - column for every external feeture (if the value is not numeric then it can be handeled with embeddings)
    
Example for such files:
graph csv file:
g_id,src,dst,label
6678,_1,_2,i
6678,_1,_3,i
6678,_2,_4,i
6678,_3,_5,i

External data file:
g_id,node,charge,chem,symbol,x,y
6678,_1,0,1,C,4.5981,-0.25
6678,_2,0,1,C,5.4641,0.25
6678,_3,0,1,C,3.7321,0.25
6678,_4,0,1,C,6.3301,-0.25

-> Parameters file for each part of the algorithm. Example files can be found in "params" directory (different for binary/multiclass). Notice that if an external file is not 
provided, you should put the associated parameters as None.

Once you have these files, to the directory 