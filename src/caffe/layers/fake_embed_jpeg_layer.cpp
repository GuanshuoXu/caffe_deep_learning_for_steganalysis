#include "caffe/layers/fake_embed_jpeg_layer.hpp"
#include <cmath>

namespace caffe {

template<typename Dtype>
void FakeEmbedJpegLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(top[0], bottom[0])<< "Only support in-place";
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();

	size_inter_ = this->layer_param_.fake_embed_jpeg_param().size_inter();
	CHECK(size_inter_%2==1 && size_inter_>=3);
	CHECK(size_inter_<50);
	sigma_inter_ = this->layer_param_.fake_embed_jpeg_param().sigma_inter();
	CHECK(sigma_inter_>0);

	gaussian_kernel_inter_.Reshape(1, 1, size_inter_, size_inter_);
	caffe_set(gaussian_kernel_inter_.count(), Dtype(0), gaussian_kernel_inter_.mutable_cpu_data());

	int siz_inter = (size_inter_ - 1) / 2;
	Dtype sum_inter = 0;
	for (int i = 0; i < size_inter_; i++) {
		for (int j = 0; j < size_inter_; j++) {
			(gaussian_kernel_inter_.mutable_cpu_data())[i*size_inter_+j] = exp((-(i-siz_inter)*(i-siz_inter)+(j-siz_inter)*(j-siz_inter))/(2.0*sigma_inter_*sigma_inter_));
			sum_inter+=(gaussian_kernel_inter_.mutable_cpu_data())[i*size_inter_+j];
		}
	}
	CHECK(sum_inter>0);
	for (int i = 0; i < size_inter_; i++) {
		for (int j = 0; j < size_inter_; j++) {
			(gaussian_kernel_inter_.mutable_cpu_data())[i*size_inter_+j] /= sum_inter;
		}
	}

	size_intra_ = 3; // fix to 3x3 for now
	CHECK(size_intra_%2==1 && size_intra_>=3);
	CHECK(size_intra_<50);
	sigma_intra_ = this->layer_param_.fake_embed_jpeg_param().sigma_intra();
	CHECK(sigma_intra_>0);

	gaussian_kernel_intra_.Reshape(1, 1, size_intra_, size_intra_);
	caffe_set(gaussian_kernel_intra_.count(), Dtype(0), gaussian_kernel_intra_.mutable_cpu_data());

	int siz_intra = (size_intra_ - 1) / 2;
	Dtype sum_intra = 0;
	for (int i = 0; i < size_intra_; i++) {
		for (int j = 0; j < size_intra_; j++) {
			(gaussian_kernel_intra_.mutable_cpu_data())[i*size_intra_+j] = exp((-(i-siz_intra)*(i-siz_intra)+(j-siz_intra)*(j-siz_intra))/(2.0*sigma_intra_*sigma_intra_));
			sum_intra+=(gaussian_kernel_intra_.mutable_cpu_data())[i*size_intra_+j];
		}
	}
	CHECK(sum_intra>0);
	for (int i = 0; i < size_intra_; i++) {
		for (int j = 0; j < size_intra_; j++) {
			(gaussian_kernel_intra_.mutable_cpu_data())[i*size_intra_+j] /= sum_intra;
		}
	}

	inter_only_ = this->layer_param_.fake_embed_jpeg_param().inter_only();
}

template<typename Dtype>
void FakeEmbedJpegLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

}

#ifdef CPU_ONLY
STUB_GPU(FakeEmbedJpegLayer);
#endif

INSTANTIATE_CLASS(FakeEmbedJpegLayer);
REGISTER_LAYER_CLASS(FakeEmbedJpeg);
}  // namespace caffe
