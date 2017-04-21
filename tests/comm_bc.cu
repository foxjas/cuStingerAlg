#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>
#include <queue>
#include <functional>
#include <vector>
#include <unordered_map>
#include <utility>
#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"
#include "algs.cuh"

#include "static_betweenness_centrality/bc.cuh"

using namespace cuStingerAlgs;


#define CUDA(call, ...) do {                        \
        cudaError_t _e = (call);                    \
        if (_e == cudaSuccess) break;               \
        fprintf(stdout,                             \
                "CUDA runtime error: %s (%d)\n",    \
                cudaGetErrorString(_e), _e);        \
                return -1;                                  \
} while (0)


char* getOption(const char* option, int argc, char **argv) {
  for (int i = 1; i < argc-1; i++) {
      if (strcmp(argv[i], option) == 0)
          return argv[i+1];
  }
  return NULL;
}

class CompareDist
{
public:
    bool operator()(pair<int,int> n1,pair<int,int> n2) {
        return n1.second > n2.second;
    }
};

// extracts top k vertices by degree
void topKVertices(int K, int n, vertexId_t *off, vertexId_t *out) {
    priority_queue<pair<int, int>, vector<pair<int, int>>,CompareDist> min_heap;
    length_t deg;
    length_t maxDeg;
    for(int v=1; v<n;v++){
        deg = off[v+1]-off[v];
        if (min_heap.size() >= K) {
            std::pair<int, int> p = min_heap.top();
            maxDeg = p.second;
            if (deg > maxDeg) {
                min_heap.pop();
                min_heap.push(std::make_pair(v, deg));
            }
        } else {
            min_heap.push(std::make_pair(v, deg));
        }
    }

    vertexId_t v;
    for (int i=0; i<K; i++) {
        v = min_heap.top().first;
        out[i] = v;
        min_heap.pop();
    }
}

vector<vector<vertexId_t>> parseInfomapCommunities(char *fpath, int nv) {

	vector<vector<vertexId_t>> communities;
    FILE *com_fp = fopen(fpath, "r");
    const int MAX_CHARS = 1000;
    char temp[MAX_CHARS];
    vertexId_t vid;
    int cid, prev_cid=-1;
    vector<vertexId_t> curr_comm;
    char* written = fgets(temp, MAX_CHARS, com_fp);
    while (written != NULL && *temp == '#') { // skip comments
        written = fgets(temp, MAX_CHARS, com_fp);
    }
    while (written != NULL) {
        sscanf(temp, "%d %d %*s\n", (vertexId_t*)&vid, (int*)&cid);
        if (prev_cid == -1)
        	prev_cid = cid;
        if (cid != prev_cid && curr_comm.size()) {
        	communities.push_back(curr_comm);
        	curr_comm.clear();
        	prev_cid = cid;
        }
        curr_comm.push_back(vid);
        written = fgets(temp, MAX_CHARS, com_fp);
    }

    // handle remaining elements
    if (curr_comm.size()) {
    	communities.push_back(curr_comm);
    }

    fclose(com_fp);
	return communities;
}

void subgraphCSR(vector<vertexId_t> const &community, length_t *off, vertexId_t *adj,
		length_t **off_sub, vertexId_t **adj_sub)
{
    unordered_map<vertexId_t, vertexId_t> relabel_map;
    vertexId_t nv = community.size();
    vector<length_t> degrees(nv, 0);
    // fill in degrees for new subgraph
    for (vertexId_t i=0; i<nv; i++) {
    	relabel_map[community[i]] = i;
    	degrees[i] = off[community[i+1]]-off[community[i]];
    }

    // offsets of new subgraph
    length_t *off_new = (length_t*)malloc(sizeof(length_t)*(nv+1));
    off_new[0]=0;
    for(vertexId_t v=0; v<nv;v++)
        off_new[v+1]=off_new[v]+degrees[v];

    for(vertexId_t v=0; v<nv;v++)
        degrees[v]=0;

    // adjacencies of new subgraph
    length_t ne = off_new[nv];
    vertexId_t *adj_new = (vertexId_t*)malloc(sizeof(vertexId_t)*ne);
    for (vertexId_t src : community) {
    	vertexId_t relabeled_src = relabel_map[src];
    	for (length_t k = off[src]; k<off[src+1]; k++) {
    		vertexId_t dest = adj[k];
    		vertexId_t relabeled_dest = relabel_map[dest];
    		adj_new[off_new[relabeled_src]+degrees[relabeled_src]++] = relabeled_dest;
    	}
    }

    *off_sub = off_new;
    *adj_sub = adj_new;
}

int main(const int argc, char *argv[]){
    int device=0;
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    length_t nv, ne,*off;
    vertexId_t *adj;

    bool isDimacs,isSNAP,isRmat,isMarket;
    string filename(argv[1]);
    isDimacs = filename.find(".graph")==std::string::npos?false:true;
    isSNAP   = filename.find(".txt")==std::string::npos?false:true;
    isRmat   = filename.find("kron")==std::string::npos?false:true;
    isMarket = filename.find(".mtx")==std::string::npos?false:true;
    bool undirected = hasOption("--undirected", argc, argv);
    char *comm_file = getOption("-p", argc, argv); // communities file path

    if(isDimacs){
        readGraphDIMACS(argv[1],&off,&adj,&nv,&ne,undirected);
    }
    else if(isSNAP){
        readGraphSNAP(argv[1],&off,&adj,&nv,&ne,undirected);
    }
    else if(isMarket){
        readGraphMatrixMarket(argv[1],&off,&adj,&nv,&ne,undirected);
    }
    else{ 
        cout << "Unknown graph type" << endl;
    }

    size_t i1 = filename.find_last_of("/");
    size_t i2 = filename.find_last_of(".");
    string rawname = filename.substr(i1+1, i2-i1-1);

    cudaEvent_t ce_start,ce_stop;
    cuStinger custing(defaultInitAllocater,defaultUpdateAllocater);

    cuStingerInitConfig cuInit;
    cuInit.initState =eInitStateCSR;
    cuInit.maxNV = nv+1;
    cuInit.useVWeight = false;
    cuInit.isSemantic = false;
    cuInit.useEWeight = false;
    cuInit.csrNV            = nv;
    cuInit.csrNE            = ne;
    cuInit.csrOff           = off;
    cuInit.csrAdj           = adj;
    cuInit.csrVW            = NULL;
    cuInit.csrEW            = NULL;

//    custing.initializeCuStinger(cuInit);
//
//    float totalTime;
//
//    // Finding k-largest vertices
//    int K = 100;
//    cout << "K: " << K << endl;
//    vertexId_t *topKV = new vertexId_t[K];
//    topKVertices(K, nv, off, topKV);
//
//    float *bc = (float *)calloc(nv, sizeof(float));
//    StaticBC sbc(K, topKV, bc);
//    sbc.Init(custing);
//    sbc.Reset();
//
//    start_clock(ce_start, ce_stop);
//    sbc.Run(custing);
//    totalTime = end_clock(ce_start, ce_stop);
//    cout << "App: BC, Graph: " << rawname << endl;
//    cout << "Total time: " << totalTime << endl;
//    free(bc);
//
//    custing.freecuStinger();

    vector<vector<vertexId_t>> communities = parseInfomapCommunities(comm_file, nv);
//    int i = 1;
//    for (vector<vertexId_t> comm : communities) {
//    	printf("%d: %d\n", i++, comm.size());
//    }

    free(off);
    free(adj);
//    delete[] topKV;
    return 0;
}

