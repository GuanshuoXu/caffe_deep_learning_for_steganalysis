#include <vector>

#include "caffe/layers/absolute_mem_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AbsoluteMemLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	bottom_sign_.Reshape(1, 1, 1, bottom[0]->count());
	caffe_gpu_set(bottom[0]->count(), char(0), bottom_sign_.mutable_gpu_data());
}

#ifdef CPU_ONLY
STUB_GPU(AbsoluteMemLayer);
#endif

INSTANTIATE_CLASS(AbsoluteMemLayer);
REGISTER_LAYER_CLASS(AbsoluteMem);

}  // namespace caffe
