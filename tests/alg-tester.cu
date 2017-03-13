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

#include "static_breadth_first_search/bfs_top_down.cuh"
// #include "static_breadth_first_search/bfs_bottom_up.cuh"
// #include "static_breadth_first_search/bfs_hybrid.cuh"
#include "static_connected_components/cc.cuh"
#include "static_page_rank/pr.cuh"


using namespace cuStingerAlgs;


#define CUDA(call, ...) do {                        \
        cudaError_t _e = (call);                    \
        if (_e == cudaSuccess) break;               \
        fprintf(stdout,                             \
                "CUDA runtime error: %s (%d)\n",    \
                cudaGetErrorString(_e), _e);        \
        return -1;                                  \
    } while (0)


//bool compare(std::pair<vertexId_t, int> a, std::pair<vertexId_t, int> b)
//{
//	return a.second < b.second;
//}

class CompareDist
{
public:
    bool operator()(pair<int,int> n1,pair<int,int> n2) {
        return n1.second > n2.second;
    }
};

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
	isRmat 	 = filename.find("kron")==std::string::npos?false:true;
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
	cuInit.isSemantic = false;  // Use edge types and vertex types
	cuInit.useEWeight = false;
	// CSR data
	cuInit.csrNV 			= nv;
	cuInit.csrNE	   		= ne;
	cuInit.csrOff 			= off;
	cuInit.csrAdj 			= adj;
	cuInit.csrVW 			= NULL;
	cuInit.csrEW			= NULL;

	custing.initializeCuStinger(cuInit);

	
	float totalTime;

//	ccBaseline scc;
//	scc.Init(custing);
//	scc.Reset();
//	start_clock(ce_start, ce_stop);
////	scc.Run(custing);
//	totalTime = end_clock(ce_start, ce_stop);
//	// cout << "The number of iterations           : " << scc.GetIterationCount() << endl;
//	// cout << "The number of connected-compoents  : " << scc.CountConnectComponents(custing) << endl;
//	// cout << "Total time for connected-compoents : " << totalTime << endl;
//	scc.Release();

//	ccConcurrent scc2;
//	scc2.Init(custing);
//	scc2.Reset();
//	start_clock(ce_start, ce_stop);
//    // scc2.Run(custing);
//	totalTime = end_clock(ce_start, ce_stop);
//	// cout << "The number of iterations           : " << scc2.GetIterationCount() << endl;
//	// cout << "The number of connected-compoents  : " << scc2.CountConnectComponents(custing) << endl;
//	// cout << "Total time for connected-compoents : " << totalTime << endl;
//	scc2.Release();


	ccConcurrentLB scc3;
	scc3.Init(custing);
	scc3.Reset();
	start_clock(ce_start, ce_stop);
	scc3.Run(custing);
	totalTime = end_clock(ce_start, ce_stop);
	cout << "App: CC, Graph: " << rawname << endl;
	cout << "The number of iterations: " << scc3.GetIterationCount() << endl;
	cout << "The number of connected-components: " << scc3.CountConnectComponents(custing) << endl;
	cout << "Total time: " << totalTime << endl;
	scc3.Release();

	// ccConcurrentOptimized scc4;
	// scc4.Init(custing);
	// scc4.Reset();
	// start_clock(ce_start, ce_stop);
	// scc4.Run(custing);
	// totalTime = end_clock(ce_start, ce_stop);
	// cout << "The number of iterations           : " << scc4.GetIterationCount() << endl;
	// cout << "The number of connected-compoents  : " << scc4.CountConnectComponents(custing) << endl;
	// cout << "Total time for connected-compoents : " << totalTime << endl; 
	// scc4.Release();

	// Finding k-largest vertices
	int K = 100;
	priority_queue<pair<int, int>, vector<pair<int, int>>,CompareDist> min_heap;
	length_t deg;
	length_t maxDeg;
	for(int v=1; v<nv;v++){
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

	bfsTD bfs;
	bfs.Init(custing);
	for (int i=0; i<K; i++) {
		bfs.Reset();
		bfs.setInputParameters(min_heap.top().first);
		start_clock(ce_start, ce_stop);
		bfs.Run(custing);
		totalTime += end_clock(ce_start, ce_stop);
		min_heap.pop();
	}
	cout << "App: BFS (top-down), Graph: " << rawname << endl;
//	cout << "The number of levels: " << bfs.getLevels() << endl;
//	cout << "The number of elements found: " << bfs.getElementsFound() << endl;
	cout << "Total time: " << totalTime << endl;

	bfs.Release();

	StaticPageRank pr;
	pr.Init(custing);
	pr.Reset();
	pr.setInputParameters(5,0.001);
	start_clock(ce_start, ce_stop);
	pr.Run(custing);
	totalTime = end_clock(ce_start, ce_stop);
	cout << "App: pr, Graph: " << rawname << endl;
	cout << "The number of iterations: " << pr.getIterationCount() << endl;
	cout << "Total time: " << totalTime << endl;
	cout << "Per iteration average: " << totalTime/(float)pr.getIterationCount() << endl;
//	pr.printRankings(custing);

	pr.Release();
	custing.freecuStinger();

	free(off);
	free(adj);
    return 0;	
}

