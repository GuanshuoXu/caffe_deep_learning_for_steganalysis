#include <iostream>
#include "caffe/layers/quantize.hpp"

namespace caffe {

template<typename Dtype>
__global__ void Quantize3(const int n, const int H, const int W,
		const Dtype* in, Dtype* out) {
	CUDA_KERNEL_LOOP(index, n)
	{
		int w = index % W;
		int h = index / W % H;
		int n = index / W / H;
		float temp, c;

		temp = in[index] / 1;
		c = (int) temp;
		if (temp >= 0) {
			temp = temp - c >= 0.5 ? c + 1 : c;
		} else {
			temp = temp - c <= -0.5 ? 1 - c : -c;
		}
		if (temp > 4)
			temp = 4;
		out[((n * 3 + 0) * H + h) * W + w] = temp;

		temp = in[index] / 2;
		c = (int) temp;
		if (temp >= 0) {
			temp = temp - c >= 0.5 ? c + 1 : c;
		} else {
			temp = temp - c <= -0.5 ? 1 - c : -c;
		}
		if (temp > 4)
			temp = 4;
		out[((n * 3 + 1) * H + h) * W + w] = temp;

		temp = in[index] / 4;
		c = (int) temp;
		if (temp >= 0) {
			temp = temp - c >= 0.5 ? c + 1 : c;
		} else {
			temp = temp - c <= -0.5 ? 1 - c : -c;
		}
		if (temp > 4)
			temp = 4;
		out[((n * 3 + 2) * H + h) * W + w] = temp;
	}
}

template<typename Dtype>
void QuantizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = bottom[0]->count();

	Quantize3<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, height_, width_, bottom_data, top_data);
	CUDA_POST_KERNEL_CHECK;

	/*std::ofstream myfile5;
	myfile5.open("example5.txt");
	std::ofstream myfile6;
	myfile6.open("example6.txt");
	for (int i=0; i<count/2; i++) {
		myfile5 << (top[0]->cpu_data())[i] << ' ';
		myfile6 << (top[0]->cpu_data())[i+count/2] << ' ';
	}
	myfile5.close();
	myfile6.close();*/

}

INSTANTIATE_LAYER_GPU_FUNCS(QuantizeLayer);
}  // namespace caffe
