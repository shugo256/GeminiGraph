# Hypergraph extension of Gemini
A computation-centric distributed hypergraph processing system.

## Quick Start
Gemini uses **MPI** for inter-process communication and **libnuma** for NUMA-aware memory allocation.
A compiler supporting **OpenMP** and **C++11** features (e.g. lambda expressions, multi-threading, etc.) is required.

Implementations of five graph analytics applications (PageRank, Connected Components, Single-Source Shortest Paths, Breadth-First Search, Betweenness Centrality) and seven hypergraph applications (PageRank, Connected Components, Single-Source Shortest Paths, Breadth-First Search, Betweenness Centrality, K-Core Decomposition, Maximal Independent Set) are inclulded in the *toolkits/* directory.

To build:
```
make
```

The input parameters of these applications are as follows:
```
# graph applications
./toolkits/pagerank [path] [vertices] [iterations]
./toolkits/cc [path] [vertices]
./toolkits/sssp [path] [vertices] [root]
./toolkits/bfs [path] [vertices] [root]
./toolkits/bc [path] [vertices] [root]

# hypergraph applications
./toolkits/hyper_pagerank [path] [vertices] [hyperedges] ([iterations]=20)
./toolkits/hyper_cc [path] [vertices] [hyperedges]
./toolkits/hyper_sssp [path] [vertices] [hyperedges] ([root]=0)
./toolkits/hyper_bfs [path] [vertices] [hyperedges] ([root]=0)
./toolkits/hyper_bc [path] [vertices] [hyperedges] ([root]=0)
./toolkits/hyper_kcore [path] [vertices] [hyperedges]
./toolkits/hyper_mis [path] [vertices] [hyperedges]

```

*[path]* gives the path of an input graph/hypergraph, i.e. a file stored on a *shared* file system, consisting of *|E|* \<source vertex (or hyperedge) id, destination vertex (or hyperedge) id, edge data\> tuples in binary. For hypergraph inputs, the system treats hyperdges just like vertices.
*[vertices]* gives the number of vertices *|V|*, 
and *[hyperedges]* gives the number of hyperedges *|U|*.
Vertex IDs are represented with 32-bit integers ranges from *0* to *|V|-1*, 
and Hyperedge IDs are also 32-bit integers but ranges from *|V|* to *|V|+|E|-1*.
Edge data can be omitted for unweighted graphs (e.g. the above applications except SSSP and HyperSSSP).

Note: CC makes the input graph undirected by adding a reversed edge to the graph for each loaded one; SSSP and HyperSSSP uses *float* as the type of weights.

If Slurm is installed on the cluster, you may run jobs like this, e.g. 20 iterations of PageRank on the *twitter-2010* graph:
```
srun -N 8 ./toolkits/pagerank /path/to/twitter-2010.binedgelist 41652230 20
```

## Modification from Gemini
See [1ab23bd](https://github.com/shugo256/GeminiGraph/commit/1ab23bd993302eaa1ddc8faf507302d7fe73b566)
* *README.md*: added hypergraph extension info
* *Makefile*: added hypergraph apps to the mix
* *.gitignore*: ignore executables inside `toolkits/`
* *atomic.h*: changed `write_add` to return the updated value and added `write_sub`
* *bitmap.h*: added `Bitmap::reset_bit`
* *graph.hpp*: added `Graph::filter_from`


## Resources

Xiaowei Zhu, Wenguang Chen, Weimin Zheng, and Xiaosong Ma.
Gemini: A Computation-Centric Distributed Graph Processing System.
12th USENIX Symposium on Operating Systems Design and Implementation (OSDI '16).

