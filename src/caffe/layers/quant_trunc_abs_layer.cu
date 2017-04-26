#include "caffe/layers/quant_trunc_abs_layer.hpp"

namespace caffe {

template<typename Dtype>
__global__ void QuantTruncAbsForward(const int n, const int th, const Dtype* in,
		Dtype* out) {
	CUDA_KERNEL_LOOP(index, n)
	{
		if (in[index] >= 0) {
			out[index] = in[index];
		} else {
			out[index] = -in[index];
		}
		if ((out[index] - (int) out[index]) >= 0.5) {
			out[index] = (int) out[index] + 1;
		} else {
			out[index] = (int) out[index];
		}
		if (out[index] > th) {
			out[index] = th;
		}
	}
}

template<typename Dtype>
__global__ void QuantForward(const int n, const Dtype* in, Dtype* out) {
	CUDA_KERNEL_LOOP(index, n)
	{
		int sign = 0;
		if (in[index] >= 0) {
			sign = 1;
			out[index] = in[index];
		} else {
			sign = -1;
			out[index] = -in[index];
		}
		if ((out[index] - (int) out[index]) >= 0.5) {
			out[index] = (int) out[index] + 1;
		} else {
			out[index] = (int) out[index];
		}
		out[index] *= sign;
	}
}

template<typename Dtype>
__global__ void AbsForward(const int n, const Dtype* in, Dtype* out) {
	CUDA_KERNEL_LOOP(index, n)
	{
		if (in[index] >= 0) {
			out[index] = in[index];
		} else {
			out[index] = -in[index];
		}
	}
}

template<typename Dtype>
__global__ void TruncForward(const int n, const int th, const Dtype* in,
		Dtype* out) {
	CUDA_KERNEL_LOOP(index, n)
	{
		if (in[index] > th) {
			out[index] = th;
		}
		if (in[index] < -th) {
			out[index] = -th;
		}
	}
}

template<typename Dtype>
__global__ void QuantTruncForward(const int n, const int th, const Dtype* in,
		Dtype* out) {
	CUDA_KERNEL_LOOP(index, n)
	{
		int sign = 0;
		if (in[index] >= 0) {
			sign = 1;
			out[index] = in[index];
		} else {
			sign = -1;
			out[index] = -in[index];
		}
		if ((out[index] - (int) out[index]) >= 0.5) {
			out[index] = (int) out[index] + 1;
		} else {
			out[index] = (int) out[index];
		}
		if (out[index] > th) {
			out[index] = th;
		}
		out[index] *= sign;
	}
}

template<typename Dtype>
__global__ void QuantAbsForward(const int n, const Dtype* in, Dtype* out) {
	CUDA_KERNEL_LOOP(index, n)
	{
		if (in[index] >= 0) {
			out[index] = in[index];
		} else {
			out[index] = -in[index];
		}
		if ((out[index] - (int) out[index]) >= 0.5) {
			out[index] = (int) out[index] + 1;
		} else {
			out[index] = (int) out[index];
		}
	}
}

template<typename Dtype>
__global__ void TruncAbsForward(const int n, const int th, const Dtype* in, Dtype* out) {
	CUDA_KERNEL_LOOP(index, n)
	{
		if (in[index] >= 0) {
			out[index] = in[index];
		} else {
			out[index] = -in[index];
		}
		if (out[index] > th) {
			out[index] = th;
		}
	}
}

template<typename Dtype>
void QuantTruncAbsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = top[0]->count();
	switch (this->layer_param_.quant_trunc_abs_param().process()) {
	case QuantTruncAbsParameter_ProcessMethod_QUANTTRUNCABS:
		QuantTruncAbsForward<Dtype> <<<CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS>>>(count, threshold_, bottom_data,
				top_data);
		break;
	case QuantTruncAbsParameter_ProcessMethod_QUANT:
		QuantForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, bottom_data, top_data);
		break;
	case QuantTruncAbsParameter_ProcessMethod_ABS:
		AbsForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, bottom_data, top_data);
		break;
	case QuantTruncAbsParameter_ProcessMethod_TRUNC:
		TruncForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, threshold_, bottom_data, top_data);
		break;
	case QuantTruncAbsParameter_ProcessMethod_QUANTTRUNC:
		QuantTruncForward<Dtype> <<<CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS>>>(count, threshold_, bottom_data,
				top_data);
		break;
	case QuantTruncAbsParameter_ProcessMethod_QUANTABS:
		QuantAbsForward<Dtype> <<<CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, top_data);
		break;
	case QuantTruncAbsParameter_ProcessMethod_TRUNCABS:
		TruncAbsForward<Dtype> <<<CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS>>>(count, threshold_, bottom_data,
				top_data);
		break;
	default:
		LOG(FATAL)<< "Unknown process method.";
	}
	CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(QuantTruncAbsLayer);

}  // namespace caffe
