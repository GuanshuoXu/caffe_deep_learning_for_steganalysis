#include "caffe/layers/quant_trunc_abs_layer.hpp"

namespace caffe {

template<typename Dtype>
void QuantTruncAbsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(top[0], bottom[0]) << "Only support in-place";
	threshold_ = this->layer_param_.quant_trunc_abs_param().threshold();
}

#ifdef CPU_ONLY
STUB_GPU(QuantTruncAbsLayer);
#endif

INSTANTIATE_CLASS(QuantTruncAbsLayer);
REGISTER_LAYER_CLASS(QuantTruncAbs);

}  // namespace caffe
