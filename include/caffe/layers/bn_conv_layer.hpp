// The main body of this batch-normalization implementation is modified from
// https://github.com/ChenglongChen/caffe-windows
// I mainly optimized memory usages.
// In-place only!

#ifndef CAFFE_BN_CONV_LAYER_HPP_
#define CAFFE_BN_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
class BNConvLayer: public Layer<Dtype> {
public:
	explicit BNConvLayer(const LayerParameter& param) :
			Layer<Dtype>(param), num_(0), channels_(0), height_(0), width_(0), decay_(
					0), fix_(false) {
	}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const {
		return "BNConv";
	}
	virtual inline int ExactNumBottomBlobs() const {
		return 1;
	}
	virtual inline int ExactNumTopBlobs() const {
		return 1;
	}

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		NOT_IMPLEMENTED;
	}
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom) {
		NOT_IMPLEMENTED;
	}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);

	// dimension
	int num_;
	int channels_;
	int height_;
	int width_;
	// decay factor
	Dtype decay_;
	// x_sum_multiplier is used to carry out sum using BLAS
	Blob<Dtype> spatial_sum_multiplier_;
	Blob<Dtype> batch_sum_multiplier_;
	Blob<Dtype> x_std_;
	bool fix_;
};

}  // namespace caffe

#endif  // CAFFE_BATCHNORM_LAYER_HPP_
