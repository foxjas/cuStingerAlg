#pragma once

#include "cuStingerAlg.hpp"

namespace custinger_alg {

struct MISData {
    MISData(cuStinger& custinger) : queue(custinger), nodes(custinger.nV()), 
    								edges(custinger.nE()) {} // syntax?

	TwoLevelQueue<vid_t> queue;
    int* values;
    int nodes;
    int edges;
};

class MIS final : public StaticAlgorithm {
	public:
	    explicit MIS(cuStinger& custinger);
	    ~MIS();

		void reset()    override;
		void run()      override;
		void release()  override;
	    bool validate() override;

	private:
		MISData  host_MIS_data;
		MISData* device_MIS_data { nullptr };
};

}
