// Modified from implementation of ReLU layer
// The bn_conv layer relies on top_data for correct backward operations.
// But the following ReLU layer inevitably changes the top_data of BN output.
// This problem is solved by recovering the input data of the ReLU layer.

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/relu_recover_layer.hpp"

namespace caffe {

template<typename Dtype>
__global__ void ReLURecoverForward(const int n, const Dtype* in, Dtype* out,
		Dtype negative_slope) {
	CUDA_KERNEL_LOOP(index, n)
	{
		out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
	}
}

template<typename Dtype>
void ReLURecoverLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = bottom[0]->count();
	Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

	if (this->phase_ == TRAIN) {
		if (bottom[0] == top[0]) {
			// For in-place computation
			caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
		}
	}

	ReLURecoverForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, bottom_data, top_data, negative_slope);
	CUDA_POST_KERNEL_CHECK
	;
}

template<typename Dtype>
__global__ void ReLURecoverBackward(const int n, const Dtype* in_diff,
		const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
	CUDA_KERNEL_LOOP(index, n)
	{
		out_diff[index] =
				in_diff[index]
						* ((in_data[index] > 0)
								+ (in_data[index] <= 0) * negative_slope);
	}
}

template<typename Dtype>
void ReLURecoverLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int count = bottom[0]->count();
		Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

		ReLURecoverBackward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, top_diff, bottom_data, bottom_diff, negative_slope);
		CUDA_POST_KERNEL_CHECK
		;

		if (this->phase_ == TRAIN) {
			if (bottom[0] == top[0]) {
				// For in-place computation
				caffe_copy(count, bottom_memory_.gpu_data(),
						bottom[0]->mutable_gpu_data());
			}
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ReLURecoverLayer);

}  // namespace caffe
