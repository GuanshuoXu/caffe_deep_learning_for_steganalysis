#include "caffe/layers/quantize.hpp"

namespace caffe {

template<typename Dtype>
void QuantizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
}

template<typename Dtype>
void QuantizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	top[0]->Reshape(num_, channels_ * 3, height_, width_);
}

#ifdef CPU_ONLY
STUB_GPU(QuantizeLayer);
#endif

INSTANTIATE_CLASS(QuantizeLayer);
REGISTER_LAYER_CLASS(Quantize);
}  // namespace caffe
