// The main body of this batch-normalization implementation is modified from
// https://github.com/ChenglongChen/caffe-windows
// I mainly optimized memory usages.
// In-place only!

#include <algorithm>
#include <vector>

#include "caffe/layers/bn_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void BNConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	decay_ = this->layer_param_.bn_conv_param().decay();
	fix_ = this->layer_param_.bn_conv_param().fix();

	x_std_.Reshape(1, channels_, 1, 1);

	// fill spatial multiplier
	spatial_sum_multiplier_.Reshape(1, 1, height_, width_);
	caffe_set(spatial_sum_multiplier_.count(), Dtype(1),
			spatial_sum_multiplier_.mutable_cpu_data());
	// fill batch multiplier
	batch_sum_multiplier_.Reshape(num_, 1, 1, 1);
	caffe_set(batch_sum_multiplier_.count(), Dtype(1),
			batch_sum_multiplier_.mutable_cpu_data());

	// Check if we need to set up the weights
	if (this->blobs_.size() > 0) {
		LOG(INFO)<< "Skipping parameter initialization";
	}
	else {
		this->blobs_.resize(4);

		// fill scale with 1s
		this->blobs_[0].reset(new Blob<Dtype>(1, channels_, 1, 1));
		caffe_set(channels_, Dtype(1), this->blobs_[0]->mutable_cpu_data());

		// fill shift with 0s
		this->blobs_[1].reset(new Blob<Dtype>(1, channels_, 1, 1));
		caffe_set(channels_, Dtype(0), this->blobs_[1]->mutable_cpu_data());

		// history mean
		this->blobs_[2].reset(new Blob<Dtype>(1, channels_, 1, 1));
		caffe_set(channels_, Dtype(0), this->blobs_[2]->mutable_cpu_data());

		// history variance
		this->blobs_[3].reset(new Blob<Dtype>(1, channels_, 1, 1));
		caffe_set(channels_, Dtype(0), this->blobs_[3]->mutable_cpu_data());

	}	// parameter initialization
	this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Dtype>
void BNConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	top[0]->Reshape(num_, channels_, height_, width_);
}

#ifdef CPU_ONLY
STUB_GPU(BNConvLayer);
#endif

INSTANTIATE_CLASS(BNConvLayer);
REGISTER_LAYER_CLASS(BNConv);
}  // namespace caffe
