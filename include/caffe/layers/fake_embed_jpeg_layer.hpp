#ifndef CAFFE_FAKE_EMBED_JPEG_LAYER_HPP_
#define CAFFE_FAKE_EMBED_JPEG_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
class FakeEmbedJpegLayer: public Layer<Dtype> {
public:
	explicit FakeEmbedJpegLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const {
		return "FakeEmbedJpeg";
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

	int size_inter_;
	Dtype sigma_inter_;
	int size_intra_;
	Dtype sigma_intra_;
	Blob<Dtype> gaussian_kernel_inter_;
	Blob<Dtype> gaussian_kernel_intra_;
	bool inter_only_;
};

}  // namespace caffe

#endif  // CAFFE_FAKE_EMBED_JPEG_LAYER_HPP_
