#ifndef CAFFE_FAKE_EMBED_SPATIAL_LAYER_HPP_
#define CAFFE_FAKE_EMBED_SPATIAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
class FakeEmbedSpatialLayer: public Layer<Dtype> {
public:
	explicit FakeEmbedSpatialLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const {
		return "FakeEmbedSpatial";
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
			const vector<Blob<Dtype>*>& bottom) {
		NOT_IMPLEMENTED;
	}

	int num_;
	int channels_;
	int height_;
	int width_;

	int size_;
	Dtype sigma_;
	Blob<Dtype> gaussian_kernel_;
};

}  // namespace caffe

#endif  // CAFFE_FAKE_EMBED_SPATIAL_LAYER_HPP_
