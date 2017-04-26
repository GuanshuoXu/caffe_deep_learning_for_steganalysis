// Modified from implementation of ReLU layer
// The bn_conv layer relies on top_data for correct backward operations.
// But the following ReLU layer inevitably changes the top_data of BN output.
// This problem is solved by recovering the input data of the ReLU layer.

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/relu_recover_layer.hpp"

namespace caffe {

template<typename Dtype>
void ReLURecoverLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	if (this->phase_ == TRAIN) {
		if (bottom[0] == top[0]) {
			// For in-place computation
			bottom_memory_.ReshapeLike(*bottom[0]);
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(ReLURecoverLayer);
#endif

INSTANTIATE_CLASS(ReLURecoverLayer);
REGISTER_LAYER_CLASS(ReLURecover);

}  // namespace caffe
