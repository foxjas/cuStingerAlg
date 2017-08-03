
#include "Static/MaxIndepSet/MIS.cuh"
#include "Support/Device/Timer.cuh"

using namespace timer;
using namespace custinger;
using namespace custinger_alg;

int main(int argc, char* argv[]) {
    using namespace graph::structure_prop;

    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv);

    cuStingerInit custinger_init(graph.nV(), graph.nE(),
                                 graph.out_offsets_ptr(),
                                 graph.out_edges_ptr());

    cuStinger custiger_graph(custinger_init);

    MIS mis(custiger_graph);

    Timer<DEVICE, seconds> TM;
    TM.start();

    mis.run();

    TM.stop();
    TM.print("Maximal Independent Set");

    mis.validate();

}
