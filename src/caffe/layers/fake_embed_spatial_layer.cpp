#include "caffe/layers/fake_embed_spatial_layer.hpp"
#include <cmath>

namespace caffe {

template<typename Dtype>
void FakeEmbedSpatialLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(top[0], bottom[0])<< "Only support in-place";
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	size_ = this->layer_param_.fake_embed_spatial_param().size();
	CHECK(size_%2==1 && size_>=3);
	CHECK(size_<50);
	sigma_ = this->layer_param_.fake_embed_spatial_param().sigma();
	CHECK(sigma_>0);

	gaussian_kernel_.Reshape(1, 1, size_, size_);
	caffe_set(gaussian_kernel_.count(), Dtype(0), gaussian_kernel_.mutable_cpu_data());

	int siz = (size_ - 1) / 2;
	Dtype sum = 0;
	for (int i = 0; i < size_; i++) {
		for (int j = 0; j < size_; j++) {
			(gaussian_kernel_.mutable_cpu_data())[i*size_+j] = exp((-(i-siz)*(i-siz)+(j-siz)*(j-siz))/(2.0*sigma_*sigma_));
			sum+=(gaussian_kernel_.mutable_cpu_data())[i*size_+j];
		}
	}
	CHECK(sum>0);
	for (int i = 0; i < size_; i++) {
		for (int j = 0; j < size_; j++) {
			(gaussian_kernel_.mutable_cpu_data())[i*size_+j] /= sum;
		}
	}
}

template<typename Dtype>
void FakeEmbedSpatialLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

}

#ifdef CPU_ONLY
STUB_GPU(FakeEmbedSpatialLayer);
#endif

INSTANTIATE_CLASS(FakeEmbedSpatialLayer);
REGISTER_LAYER_CLASS(FakeEmbedSpatial);
}  // namespace caffe
