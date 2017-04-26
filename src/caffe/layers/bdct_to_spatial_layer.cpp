#include "caffe/layers/bdct_to_spatial_layer.hpp"

namespace caffe {

template<typename Dtype>
void BdctToSpatialLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	num_ = bottom[0]->num();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	img_h_ = 8 * bottom[0]->height();
	img_w_ = 8 * bottom[0]->width();

	quality_ = this->layer_param_.bdct_to_spatial_param().quality();
	quant_matrix_.Reshape(1, 1, 1, 64);
	caffe_set(64, Dtype(0), quant_matrix_.mutable_cpu_data());

	// QF must be either 75 or 95
	// Q matrices obtained from Matlab
	if (quality_ == 75) {
		Dtype qm[64] = { 8, 6, 5, 8, 12, 20, 26, 31, 6, 6, 7, 10, 13, 29, 30,
				28, 7, 7, 8, 12, 20, 29, 35, 28, 7, 9, 11, 15, 26, 44, 40, 31,
				9, 11, 19, 28, 34, 55, 52, 39, 12, 18, 28, 32, 41, 52, 57, 46,
				25, 32, 39, 44, 52, 61, 60, 51, 36, 46, 48, 49, 56, 50, 52, 50 };
		for (int i=0; i<64; i++) {
			(quant_matrix_.mutable_cpu_data())[i] = qm[i];
		}
	} else if (quality_ == 95) {
		Dtype qm[64] = { 2, 1, 1, 2, 2, 4, 5, 6, 1, 1, 1, 2, 3, 6, 6, 6, 1, 1,
				2, 2, 4, 6, 7, 6, 1, 2, 2, 3, 5, 9, 8, 6, 2, 2, 4, 6, 7, 11, 10,
				8, 2, 4, 6, 6, 8, 10, 11, 9, 5, 6, 8, 9, 10, 12, 12, 10, 7, 9,
				10, 10, 11, 10, 10, 10 };
		for (int i = 0; i < 64; i++) {
			(quant_matrix_.mutable_cpu_data())[i] = qm[i];
		}
	} else {
		CHECK_EQ(1, 0) << "QF must be either 75 or 95!";
	}

}

template<typename Dtype>
void BdctToSpatialLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	top[0]->Reshape(num_, 1, img_h_, img_w_);
}

#ifdef CPU_ONLY
STUB_GPU(BdctToSpatialLayer);
#endif

INSTANTIATE_CLASS(BdctToSpatialLayer);
REGISTER_LAYER_CLASS(BdctToSpatial);
}  // namespace caffe
