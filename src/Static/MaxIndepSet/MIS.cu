#include "Static/MaxIndepSet/MIS.cuh"

/* 
 * hash and initialization body sourced from 
 * http://cs.txstate.edu/~burtscher/research/ECL-MIS/ECL-MIS_10.cu
 */

namespace custinger_alg {

typedef unsigned char stattype;
static const stattype in = ~0 - 1;
static const stattype out = 0;

/// Helpers ///
//------------------------------------------------------------------------------
static __device__ unsigned int hash(unsigned int val) {
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

//------------------------------------------------------------------------------

/// Operators ///

__device__ __forceinline__
void VertexInit(const Vertex& src, void* optional_field) {
    auto MIS_data = reinterpret_cast<MISData*>(optional_field);
    const int index = src.id();
    const float avg = (float)MIS_data->edges / MIS_data->nodes;
    const float scaledavg = ((in / 2) - 1) * avg;
    stattype val = in;
    const int degree = src.degree();
    if (degree > 0) {
      float x = degree - (hash(index) * 0.00000000023283064365386962890625f);
      int res = __float2int_rn(scaledavg / (avg + x));
      val = (res + res) | 1;
    }
    MIS_data->values[index] = val;
}

__device__ __forceinline__
void VertexFilter(const Vertex& src, void* optional_field) {
    auto MIS_data = reinterpret_cast<MISData*>(optional_field);
    int v = src.id();
    int v_dest;
    int* values = MIS_data->values;
    if (values[v] & 1) { // neither in nor out
      int i = 0;
      v_dest = src.edge(i).dst();
      while ((i < src.degree()) && ((values[v] > values[v_dest]) || ((values[v] == values[v_dest]) && (v > v_dest)))) {
        i++;
        v_dest = src.edge(i).dst();
      }
      if (i < src.degree()) { // v is not a local maximum
        MIS_data->queue.insert(v);
      } else { // v is a local maximum; process neighbors
        for (int i = 0; i < src.degree(); i++) {
          v_dest = src.edge(i).dst();
          values[v_dest] = out;
        }
        values[src.id()] = in;
      }
    }
}

//------------------------------------------------------------------------------

MIS::MIS(custinger::cuStinger& custinger) :
                                       StaticAlgorithm(custinger),
                                       host_MIS_data(custinger) {
    cuMalloc(host_MIS_data.values, custinger.nV());
    device_MIS_data = register_data(host_MIS_data);
    reset();
}

MIS::~MIS() {
    cuFree(host_MIS_data.values);
}

void MIS::reset() {
    host_MIS_data.queue.clear();
}

void MIS::run() {
    using namespace timer;
    Timer<DEVICE> TM;
    TM.start();

    forAllVertices<VertexInit>(custinger, device_MIS_data);

    forAllVertices<VertexFilter>(custinger, device_MIS_data);
    host_MIS_data.queue.swap();
    printf("host size after VertexFilter: %d\n", host_MIS_data.queue.size());
    while (host_MIS_data.queue.size() > 0) {
      forAllVertices<VertexFilter>(custinger, device_MIS_data);
      host_MIS_data.queue.swap();
      printf("host size after VertexFilter: %d\n", host_MIS_data.queue.size());
    }
    TM.stop();
    TM.print("Main computation body");
}

void MIS::release() {
    cuFree(host_MIS_data.values);
    host_MIS_data.values = nullptr;
}

bool MIS::validate() {
    using namespace graph;
    GraphStd<vid_t, eoff_t> graph(custinger.csr_offsets(), custinger.nV(), custinger.csr_edges(), custinger.nE());
    int* host_values = new int[graph.nV()];
    cuMemcpyToHost(host_MIS_data.values, graph.nV(), host_values);
    const eoff_t* offsets = graph.out_offsets();
    const eoff_t* adjacencies = graph.out_edges();
    int numIn = 0;
    for (int v = 0; v < custinger.nV(); v++) {
      if ((host_values[v] != in) && (host_values[v] != out)) {
        fprintf(stderr, "ERROR: found unprocessed node in graph\n\n");  exit(-1);
      }
      if (host_values[v] == in) {
        numIn += 1;
        for (int i = offsets[v]; i < offsets[v + 1]; i++) {
          if (host_values[adjacencies[i]] == in) {
            fprintf(stderr, "ERROR: found adjacent nodes in MIS\n\n");  
            exit(-1);
          }
        }
      } else {
        int flag = 0;
        for (int i = offsets[v]; i < offsets[v + 1]; i++) {
          if (host_values[adjacencies[i]] == in) {
            flag = 1;
          }
        }
        if (flag == 0) {
          fprintf(stderr, "ERROR: set is not maximal\n\n");
          exit(-1);
        }
      }
    }

    printf("Vertices in set: %d\n", numIn);
    delete[] host_values;
}


} // namespace custinger_alg
