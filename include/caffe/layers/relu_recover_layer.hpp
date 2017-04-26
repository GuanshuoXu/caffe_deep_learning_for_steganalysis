// Modified from implementation of ReLU layer
// The bn_conv layer relies on top_data for correct backward operations.
// But the following ReLU layer inevitably changes the top_data of BN output.
// This problem is solved by recovering the input data of the ReLU layer.

#ifndef CAFFE_RELU_RECOVER_LAYER_HPP_
#define CAFFE_RELU_RECOVER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template<typename Dtype>
class ReLURecoverLayer: public NeuronLayer<Dtype> {
public:

	explicit ReLURecoverLayer(const LayerParameter& param) :
			NeuronLayer<Dtype>(param) {
	}

	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const {
		return "ReLURecover";
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

	Blob<Dtype> bottom_memory_; // memory for in-place computation
};

}  // namespace caffe

#endif  // CAFFE_RELU_RECOVER_LAYER_HPP_
