#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>
#include <assert.h>
#include <queue>
#include <functional>
#include <vector>
#include <unordered_map>
#include <unordered_set>
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
		length_t **off_sub, vertexId_t **adj_sub, vertexId_t *NV, length_t *NE)
{
    unordered_map<vertexId_t, vertexId_t> relabel_map;
    vertexId_t nv = community.size();
    vector<length_t> degrees(nv, 0);
    vector<pair<vertexId_t, vertexId_t>> edges;

    for (vertexId_t i=0; i<nv; i++)
    	relabel_map[community[i]] = i;

    int ecount = 0;
    unordered_map<vertexId_t, vertexId_t>::const_iterator m;
    for (vertexId_t src : community) {
    	for (length_t k = off[src]; k<off[src+1]; k++) {
    		vertexId_t dest = adj[k];
    		m = relabel_map.find(dest);
    		if (m != relabel_map.end()) {
    			edges.push_back(make_pair(src, dest));
    			ecount += 1;
    		}
    	}
    }

    // fill in degrees array
    vertexId_t relabeledSrcId, relabeledDestId;
    for (int i=0; i<edges.size(); i++) {
        relabeledSrcId = relabel_map[edges[i].first];
        degrees[relabeledSrcId]++;
    }

    // offsets of new subgraph
    length_t *off_new = (length_t*)malloc(sizeof(length_t)*(nv+1));
    off_new[0]=0;
    for(vertexId_t v=0; v<nv;v++) {
        off_new[v+1]=off_new[v]+degrees[v];
        degrees[v]=0;
    }

    // adjacencies of new subgraph
    length_t ne = off_new[nv];
    assert(ne == ecount);
    vertexId_t *adj_new = (vertexId_t*)malloc(sizeof(vertexId_t)*ne);
    for (int i=0; i<edges.size(); i++) {
        relabeledSrcId = relabel_map[edges[i].first];
        relabeledDestId = relabel_map[edges[i].second];
		adj_new[off_new[relabeledSrcId]+degrees[relabeledSrcId]++] = relabeledDestId;
    }

    *off_sub = off_new;
    *adj_sub = adj_new;
    *NV = nv;
    *NE = ne;
}

void printCommunityInfo(vector<vector<vertexId_t>> communities, length_t *off, vertexId_t *adj) {
    int nv_sub;
    int ne_sub;
    length_t *off_sub;
    vertexId_t *adj_sub;

    printf("id nv ne\n");
    length_t ne_total = 0;
    vertexId_t nv_total = 0;
    for (int i=0; i<communities.size(); i++) {
    	vector<vertexId_t> comm = communities[i];
        subgraphCSR(comm, off, adj, &off_sub, &adj_sub, &nv_sub, &ne_sub);
        printf("%d %d %d\n", i+1, nv_sub, ne_sub);
        ne_total += ne_sub;
        nv_total += nv_sub;
    }
    printf("%d %d\n", nv_total, ne_total);
    free(off_sub);
    free(adj_sub);
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

    vector<vector<vertexId_t>> communities = parseInfomapCommunities(comm_file, nv);

    cudaEvent_t ce_start,ce_stop;
    cuStinger custing(defaultInitAllocater,defaultUpdateAllocater);
    cuStingerInitConfig cuInit;
    cuInit.initState = eInitStateCSR;
    cuInit.useVWeight = false;
    cuInit.isSemantic = false;
    cuInit.useEWeight = false;
    cuInit.csrVW = NULL;
    cuInit.csrEW = NULL;
    int nv_sub;
    int ne_sub;
    length_t *off_sub;
    vertexId_t *adj_sub;
    float *bc;
    float time;
    float totalTime = 0.0;
//    printCommunityInfo(communities, off, adj);
    for (int i=426; i<427; i++) {
    	vector<vertexId_t> comm = communities[i];
        subgraphCSR(comm, off, adj, &off_sub, &adj_sub, &nv_sub, &ne_sub);
        cuInit.maxNV = nv_sub+1;
        cuInit.csrNV = nv_sub;
        cuInit.csrNE = ne_sub;
        cuInit.csrOff = off_sub;
        cuInit.csrAdj = adj_sub;
        custing.initializeCuStinger(cuInit);
        printf("nv_sub: %d\n", nv_sub);
        printf("Off: ");
        for (int j=0; j<nv_sub+1; j++) {
        	printf("%d ", off_sub[j]);
        }
        printf("\n");

        printf("ne_sub: %d\n", ne_sub);
        printf("Adj: ");
        for (int j=0; j<ne_sub; j++) {
        	printf("%d ", adj_sub[j]);
        	assert(adj_sub[j] >= 0 && adj_sub[j] < nv_sub);
        }
        printf("\n");

		bc = (float *)calloc(nv, sizeof(float));
		StaticBC sbc(bc);
		sbc.Init(custing);
		sbc.Reset();
		start_clock(ce_start, ce_stop);
		sbc.Run(custing);
		time = end_clock(ce_start, ce_stop);
		printf("%d %f\n", i+1, time);
		totalTime += time;
		free(off_sub);
		free(adj_sub);
		free(bc);
        custing.freecuStinger();
    }
    printf("Total time for %d communities: %f\n", communities.size(), totalTime);

    free(off);
    free(adj);
//    delete[] topKV;
    return 0;
}

