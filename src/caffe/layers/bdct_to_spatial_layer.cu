#include "caffe/layers/bdct_to_spatial_layer.hpp"

namespace caffe {

template<typename Dtype>
__global__ void Dequantize(const int n, const int H, const int W,
		const Dtype* quant, const Dtype* in, Dtype* out) {
	CUDA_KERNEL_LOOP(index, n)
	{
		int w = index % W;
		int h = index / W % H;
		int k = index / W / H % 64;
		int n = index / W / H / 64;
		out[(n * H * W + h * W + w) * 64 + k] = in[index] * quant[k];
	}
}

// The Idct function uses the code from: http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html
// fft2d.zip (2006/12/28) file shrtdct.c
// Copyright Takuya OOURA, 1996-2001
template<typename Dtype>
__global__ void Idct(const int n, const Dtype* in, Dtype* out) {
	CUDA_KERNEL_LOOP(index, n)
	{
		int offset = index * 64;

		Dtype C8_1R = 0.49039264020161522456;
		Dtype C8_1I = 0.09754516100806413392;
		Dtype C8_2R = 0.46193976625564337806;
		Dtype C8_2I = 0.19134171618254488586;
		Dtype C8_3R = 0.41573480615127261854;
		Dtype C8_3I = 0.27778511650980111237;
		Dtype C8_4R = 0.35355339059327376220;
		Dtype W8_4R = 0.70710678118654752440;

		Dtype x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
		Dtype xr, xi;

		for (int j = 0; j <= 7; j++) {
			x1r = C8_1R * in[offset + 1 * 8 + j]
					+ C8_1I * in[offset + 7 * 8 + j];
			x1i = C8_1R * in[offset + 7 * 8 + j]
					- C8_1I * in[offset + 1 * 8 + j];
			x3r = C8_3R * in[offset + 3 * 8 + j]
					+ C8_3I * in[offset + 5 * 8 + j];
			x3i = C8_3R * in[offset + 5 * 8 + j]
					- C8_3I * in[offset + 3 * 8 + j];
			xr = x1r - x3r;
			xi = x1i + x3i;
			x1r += x3r;
			x3i -= x1i;
			x1i = W8_4R * (xr + xi);
			x3r = W8_4R * (xr - xi);
			xr = C8_2R * in[offset + 2 * 8 + j]
					+ C8_2I * in[offset + 6 * 8 + j];
			xi = C8_2R * in[offset + 6 * 8 + j]
					- C8_2I * in[offset + 2 * 8 + j];
			x0r = C8_4R * (in[offset + 0 * 8 + j] + in[offset + 4 * 8 + j]);
			x0i = C8_4R * (in[offset + 0 * 8 + j] - in[offset + 4 * 8 + j]);
			x2r = x0r - xr;
			x2i = x0i - xi;
			x0r += xr;
			x0i += xi;
			out[offset + 0 * 8 + j] = x0r + x1r;
			out[offset + 7 * 8 + j] = x0r - x1r;
			out[offset + 2 * 8 + j] = x0i + x1i;
			out[offset + 5 * 8 + j] = x0i - x1i;
			out[offset + 4 * 8 + j] = x2r - x3i;
			out[offset + 3 * 8 + j] = x2r + x3i;
			out[offset + 6 * 8 + j] = x2i - x3r;
			out[offset + 1 * 8 + j] = x2i + x3r;
		}
		for (int j = 0; j <= 7; j++) {
			x1r = C8_1R * out[offset + j * 8 + 1]
					+ C8_1I * out[offset + j * 8 + 7];
			x1i = C8_1R * out[offset + j * 8 + 7]
					- C8_1I * out[offset + j * 8 + 1];
			x3r = C8_3R * out[offset + j * 8 + 3]
					+ C8_3I * out[offset + j * 8 + 5];
			x3i = C8_3R * out[offset + j * 8 + 5]
					- C8_3I * out[offset + j * 8 + 3];
			xr = x1r - x3r;
			xi = x1i + x3i;
			x1r += x3r;
			x3i -= x1i;
			x1i = W8_4R * (xr + xi);
			x3r = W8_4R * (xr - xi);
			xr = C8_2R * out[offset + j * 8 + 2]
					+ C8_2I * out[offset + j * 8 + 6];
			xi = C8_2R * out[offset + j * 8 + 6]
					- C8_2I * out[offset + j * 8 + 2];
			x0r = C8_4R * (out[offset + j * 8 + 0] + out[offset + j * 8 + 4]);
			x0i = C8_4R * (out[offset + j * 8 + 0] - out[offset + j * 8 + 4]);
			x2r = x0r - xr;
			x2i = x0i - xi;
			x0r += xr;
			x0i += xi;
			out[offset + j * 8 + 0] = x0r + x1r;
			out[offset + j * 8 + 7] = x0r - x1r;
			out[offset + j * 8 + 2] = x0i + x1i;
			out[offset + j * 8 + 5] = x0i - x1i;
			out[offset + j * 8 + 4] = x2r - x3i;
			out[offset + j * 8 + 3] = x2r + x3i;
			out[offset + j * 8 + 6] = x2i - x3r;
			out[offset + j * 8 + 1] = x2i + x3r;
		}
	}
}

template<typename Dtype>
__global__ void ModeToSpatial(const int n, const int H, const int W, const Dtype* in,
		Dtype* out) {
	CUDA_KERNEL_LOOP(index, n)
	{
		int dct_w = index % 8;
		int dct_h = index / 8 % 8;
		int blk_w = index / 64 % W;
		int blk_h = index / 64 / W % H;
		int n = index / 64 / W / H;
		out[n*W*H*64+(blk_h*8+dct_h)*W*8+blk_w*8+dct_w] = in[index];
	}
}

template<typename Dtype>
void BdctToSpatialLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = bottom[0]->count();
	const int count_block = num_ * height_ * width_;

	Blob<Dtype> dequantized(num_, 1, height_ * width_, 64);
	caffe_gpu_set(dequantized.count(), Dtype(0),
			dequantized.mutable_gpu_data());
	Blob<Dtype> idcted(num_, 1, height_ * width_, 64);
	caffe_gpu_set(idcted.count(), Dtype(0), idcted.mutable_gpu_data());

	Dequantize<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, height_, width_, quant_matrix_.gpu_data(), bottom_data,
			dequantized.mutable_gpu_data());
	CUDA_POST_KERNEL_CHECK;

	Idct<Dtype> <<<CAFFE_GET_BLOCKS(count_block), CAFFE_CUDA_NUM_THREADS>>>(
			count_block, dequantized.gpu_data(), idcted.mutable_gpu_data());
	CUDA_POST_KERNEL_CHECK;

	ModeToSpatial<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
			height_, width_, idcted.gpu_data(), top_data);
	CUDA_POST_KERNEL_CHECK;

}

INSTANTIATE_LAYER_GPU_FUNCS(BdctToSpatialLayer);
}  // namespace caffe
