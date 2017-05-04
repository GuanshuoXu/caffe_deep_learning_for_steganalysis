#include <vector>

#include "caffe/layers/absolute_mem_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void storeSign(const int n, const Dtype* bottom_data, char* bottom_sign_) {
  CUDA_KERNEL_LOOP(index, n) {
	  if (bottom_data[index]>0) {
		  bottom_sign_[index] = 1;
	  }
	  else if (bottom_data[index]<0) {
		  bottom_sign_[index] = -1;
	  }
	  else {
		  bottom_sign_[index] = 0;
	  }
  }
}

template <typename Dtype>
__global__ void backward_kernel(const int n, Dtype* bottom_diff, const char* bottom_sign_) {
  CUDA_KERNEL_LOOP(index, n) {
	  if (bottom_sign_[index]<0) {
		  bottom_diff[index] *= Dtype(-1);
	  }
	  else if (bottom_sign_[index]==0) {
		  bottom_diff[index] = 0;
	  }
  }
}

template <typename Dtype>
void AbsoluteMemLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const int count = bottom[0]->count();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	if (this->phase_==TEST) {
		caffe_gpu_abs(count, bottom[0]->gpu_data(), top_data);
	}
	else {
		storeSign<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, bottom_data, bottom_sign_.mutable_gpu_data());
		CUDA_POST_KERNEL_CHECK;
		caffe_gpu_abs(count, bottom[0]->gpu_data(), top_data);
	}
}

template<typename Dtype>
void AbsoluteMemLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	const int count = top[0]->count();
	if (propagate_down[0]) {
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		backward_kernel<Dtype> <<<CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS>>>(count, bottom_diff, bottom_sign_.gpu_data());
		CUDA_POST_KERNEL_CHECK;
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(AbsoluteMemLayer);


}  // namespace caffe
