
#include "Static/MaxIndepSet/MIS.cuh"

int main(int argc, char* argv[]) {
    using namespace timer;
    using namespace custinger;
    using namespace custinger_alg;

    graph::GraphStd<vid_t, eoff_t> graph; 
    graph.read(argv[1]); // directed/undirected? need undirected for MIS

    cuStingerInit custinger_init(graph.nV(), graph.nE(), graph.out_offsets(),
                                 graph.out_edges());

    cuStinger custiger_graph(custinger_init);
    MIS mis(custiger_graph);
    // bfs_top_down.set_parameters(graph.max_out_degree_vertex());

    Timer<DEVICE> TM;
    TM.start();

    mis.run();

    TM.stop();
    TM.print("Maximal Independent Set");

    mis.validate();

    // auto is_correct = bfs_top_down.validate();
    // std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    // return is_correct;
}
