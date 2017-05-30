#include "Static/MaxIndepSet/MIS.cuh"

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
    MIS_data->canRemove[index] = 1;
    // printf("%u\n", MIS_data->values[index]); // VERIFIED
}

/** 
 * Use bit string for canRemove
 */
__device__ __forceinline__
void findMaximums(const Vertex& src, const Edge& edge, void* optional_field) {
    auto MIS_data = reinterpret_cast<MISData*>(optional_field);
    int dst = edge.dst();
    int v = src.id();
    int* values = MIS_data->values;
    int* canRemove = MIS_data->canRemove;
    int greater = ((values[v] > values[dst]) || ((values[v] == values[dst]) && (v > dst)));
    atomicAnd(canRemove+v, greater);
    // canRemove[v] &= greater;
}

__device__ __forceinline__
void findRemovalNeighbors(const Vertex& src, const Edge& edge, void* optional_field) {
    auto MIS_data = reinterpret_cast<MISData*>(optional_field);
    int v = src.id();
    int* canRemove = MIS_data->canRemove;
    int* values = MIS_data->values;
    // if (v<50 && src.edge(0).dst() == edge.dst())
    //   printf("(%d,%d) ", v, canRemove[v]);
    if (canRemove[v]) {
      values[v] = in; // race condition O?
    } else {
      int neighborIn = canRemove[edge.dst()];
      atomicOr(canRemove+v, neighborIn);
      // canRemove[v] += neighborIn;
    }
    if (v<50 && src.edge(0).dst() == edge.dst())
      printf("(%d,%d) ", v, values[v]);
}

// TODO: can be improved if we could do forAllVertices in a queue (less work each round)
__device__ __forceinline__
void removeAndEnqueue(vid_t index, void* optional_field) {
    auto MIS_data = reinterpret_cast<MISData*>(optional_field);
    int* values = MIS_data->values;
    int* canRemove = MIS_data->canRemove;
    if (values[index] & 1) { // v neither in nor out
      if (!canRemove[index]) {
        MIS_data->queue.insert(index);
        canRemove[index] = 1;
      } else {
        values[index] = out;
      }
    } 
}

//------------------------------------------------------------------------------

MIS::MIS(custinger::cuStinger& custinger) :
                                       StaticAlgorithm(custinger),
                                       host_MIS_data(custinger) {
    cuMalloc(host_MIS_data.values, custinger.nV());
    cuMalloc(host_MIS_data.canRemove, custinger.nV());
    device_MIS_data = register_data(host_MIS_data);
    reset();
}

MIS::~MIS() {
    cuFree(host_MIS_data.values);
    cuFree(host_MIS_data.canRemove);
}

void MIS::reset() {
    host_MIS_data.queue.clear();
    forAllVertices<VertexInit>(custinger, device_MIS_data);
}

// TODO: forAllEdges call seems broken; forced to use hacky, slow workaround for enqueueing vertices
void MIS::run() {
    // forAllEdges<findMaximums>(custinger, device_MIS_data);
    // syncDeviceWithHost();
    // forAllEdges<findRemovalNeighbors>(custinger, device_MIS_data);
    // syncDeviceWithHost();
    // forAllnumV<removeAndEnqueue>(custinger, device_MIS_data);
    // host_MIS_data.queue.swap();

    for (int i=0; i<custinger.nV(); i++) {
      host_MIS_data.queue.insert(i);
    }
    printf("Initial queue size: %d\n", host_MIS_data.queue.size()); 

    using namespace timer;
    Timer<DEVICE> TM;
    TM.start();
    while (host_MIS_data.queue.size() > 0) {
      load_balacing.traverse_edges<findMaximums>(host_MIS_data.queue, device_MIS_data);
      // syncDeviceWithHost();
      printf("host size after findMaximums: %d\n", host_MIS_data.queue.size());

      load_balacing.traverse_edges<findRemovalNeighbors>(host_MIS_data.queue, device_MIS_data);
      // syncDeviceWithHost();
      printf("host size after findRemovalNeighbors: %d\n", host_MIS_data.queue.size());

      forAllnumV<removeAndEnqueue>(custinger, device_MIS_data);
      host_MIS_data.queue.swap();
      printf("host size after removeAndEnqueue: %d\n", host_MIS_data.queue.size());
    }
    TM.stop();
    TM.print("Main computation time");
}

void MIS::release() {
    cuFree(host_MIS_data.values, host_MIS_data.canRemove);
    host_MIS_data.values = nullptr;
    host_MIS_data.canRemove = nullptr;
}

bool MIS::validate() {
    using namespace graph;
    GraphStd<vid_t, eoff_t> graph(custinger.csr_offsets(), custinger.nV(), custinger.csr_edges(), custinger.nE());
    int* host_values = new int[graph.nV()];
    cuMemcpyToHost(host_MIS_data.values, graph.nV(), host_values);
    const eoff_t* offsets = graph.out_offsets();
    const eoff_t* adjacencies = graph.out_edges();
    int numIn = 0;
    int numInConflict = 0;
    int numOutConflict = 0;
    for (int v = 0; v < custinger.nV(); v++) {
      if ((host_values[v] != in) && (host_values[v] != out)) {
        fprintf(stderr, "ERROR: found unprocessed node in graph\n\n");  exit(-1);
      }
      if (host_values[v] == in) {
        numIn += 1;
        for (int i = offsets[v]; i < offsets[v + 1]; i++) {
          if (host_values[adjacencies[i]] == in) {
            numInConflict += 1;
            // fprintf(stderr, "ERROR: found adjacent nodes in MIS\n\n");  
            // exit(-1);
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
          // fprintf(stderr, "ERROR: set is not maximal\n\n");
          // exit(-1);
          numOutConflict += 1;
        }
      }
    }

    printf("Vertices in set: %d\n", numIn);
    printf("In conflicts: %d\n", numInConflict);
    delete[] host_values;
}


} // namespace custinger_alg
