// Input: BDCT coefficients
// Output: decompressed JPEG image without the last rounding/truncation step (not mapped to 0,...,255)

#ifndef CAFFE_BDCT_TO_SPATIAL_LAYER_HPP_
#define CAFFE_BDCT_TO_SPATIAL_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
class BdctToSpatialLayer: public Layer<Dtype> {
public:
	explicit BdctToSpatialLayer(const LayerParameter& param) :
			Layer<Dtype>(param), num_(0), height_(0), width_(0), img_h_(0), img_w_(
					0), quality_(0) {
	}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const {
		return "BdctToSpatial";
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

	// dimension
	int num_;
	int height_;
	int width_;
	int img_h_;
	int img_w_;
	Blob<Dtype> quant_matrix_;
	int quality_;
};

}  // namespace caffe

#endif  // CAFFE_BDCT_TO_SPATIAL_LAYER_HPP_
