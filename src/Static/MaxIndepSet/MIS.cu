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

// forceinline?
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
      val = (res + res) | 1; // ensures least significant bit is 1
    }
    MIS_data->statuses[index] = val;
    // printf("%u\n", MIS_data->statuses[index]); // VERIFIED
}

__device__ __forceinline__
void findMaximums(const Vertex& src, void* optional_field) {
    auto MIS_data = reinterpret_cast<MISData*>(optional_field);
    int missing;
    int v;
    int v_dest;
    int* statuses = MIS_data->statuses;
    do {
      missing = 0;
      v = src.id();
      if (statuses[v] & 1) { // neither in nor out
        int i = 0;
        v_dest = src.edge(i).dst();
        while ((i < src.degree()) && ((statuses[v] > statuses[v_dest]) || ((statuses[v] == statuses[v_dest]) && (v > v_dest)))) {
          i++;
          v_dest = src.edge(i).dst();
        }
        if (i < src.degree()) { // v <= some neighbor's rand value
          missing = 1;  // this thread stays alive
        } else {
          for (int i = 0; i < src.degree(); i++) {
            v_dest = src.edge(i).dst();
            MIS_data->statuses[v_dest] = out;
          }
          MIS_data->statuses[src.id()] = in;
        }
      }
    } while (missing != 0); // is there any guarantee that this thread wouldn't deadlock?
}
//------------------------------------------------------------------------------

MIS::MIS(custinger::cuStinger& custinger) :
                                       StaticAlgorithm(custinger),
                                       host_MIS_data(custinger) {
    cuMalloc(host_MIS_data.statuses, custinger.nV());
    device_MIS_data = register_data(host_MIS_data);
    reset();
}

MIS::~MIS() {
    cuFree(host_MIS_data.statuses);
}

// initialization
void MIS::reset() {
    host_MIS_data.queue.clear();
    syncDeviceWithHost();
    forAllVertices<VertexInit>(custinger, device_MIS_data);
}

// void MIS::set_parameters(vid_t source) {
//     bfs_source = source;
//     host_MIS_data.queue.insert(bfs_source);
//     cuMemcpyToDevice(0, host_MIS_data.distances + bfs_source);
// }

// When to call "syncHostWithDevice()" and "syncDeviceWithHost()"?
void MIS::run() {
    syncHostWithDevice();
    forAllVertices<findMaximums>(custinger, device_MIS_data);
    syncDeviceWithHost();
}

void MIS::release() {
    cuFree(host_MIS_data.statuses);
    host_MIS_data.statuses = nullptr;
}

bool MIS::validate() {
    using namespace graph; // ?
    GraphStd<vid_t, eoff_t> graph(custinger.csr_offsets(), custinger.nV(), custinger.csr_edges(), custinger.nE());
    int* statuses = host_MIS_data.statuses;
    const eoff_t* offsets = graph.out_offsets();
    const eoff_t* adjacencies = graph.out_edges();
    for (int v = 0; v < custinger.nV(); v++) {
      if ((statuses[v] != in) && (statuses[v] != out)) {
        fprintf(stderr, "ERROR: found unprocessed node in graph\n\n");  exit(-1);
      }
      if (statuses[v] == in) {
        for (int i = offsets[v]; i < offsets[v + 1]; i++) {
          if (statuses[adjacencies[i]] == in) {fprintf(stderr, "ERROR: found adjacent nodes in MIS\n\n");  exit(-1);}
        }
      } else {
        int flag = 0;
        for (int i = offsets[v]; i < offsets[v + 1]; i++) {
          if (statuses[adjacencies[i]] == in) {
            flag = 1;
          }
        }
        if (flag == 0) {
          fprintf(stderr, "ERROR: set is not maximal\n\n");  exit(-1);
        }
      }
    }
}


} // namespace custinger_alg
