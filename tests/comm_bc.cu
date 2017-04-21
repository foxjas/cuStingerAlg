#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>
#include <queue>
#include <functional>
#include <vector>
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

    custing.initializeCuStinger(cuInit);

    float totalTime;

    // Finding k-largest vertices
    int K = 100;
    cout << "K: " << K << endl;
    vertexId_t *topKV = new vertexId_t[K];
    topKVertices(K, nv, off, topKV);

    float *bc = (float *)calloc(nv, sizeof(float));
    StaticBC sbc(K, topKV, bc);
    sbc.Init(custing);
    sbc.Reset();

    start_clock(ce_start, ce_stop);
    sbc.Run(custing);
    totalTime = end_clock(ce_start, ce_stop);
    cout << "App: BC, Graph: " << rawname << endl;
    cout << "Total time: " << totalTime << endl;
    free(bc);

    custing.freecuStinger();

    free(off);
    free(adj);
    delete[] topKV;
    return 0;
}

