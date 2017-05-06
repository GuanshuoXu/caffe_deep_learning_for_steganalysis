#include "caffe/layers/fake_embed_jpeg_layer.hpp"

namespace caffe {

template<typename Dtype>
__global__ void FakeEmbedJpegInterOnly(const int n, const int K, const int H,
		const int W, const Dtype* kernel_inter, const int size_inter,
		const Dtype* rand_location_inter, const Dtype* rand_operation,
		const Dtype* in, Dtype* out) {
	CUDA_KERNEL_LOOP(index, n)
	{
		const int w = index % W;
		const int h = index / W % H;
		const int k = index / W / H % K;
		const int n = index / W / H / K; // n-th stego
		const int index_cover = (((2 * n + 0) * K + k) * H + h) * W + w;
		const int index_stego = (((2 * n + 1) * K + k) * H + h) * W + w;
		const int diff = in[index_cover] - in[index_stego];

		const int siz_inter = (size_inter - 1) / 2;
		const int hstart_inter = max(h - siz_inter, 0);
		const int hend_inter = min(h + siz_inter, H);
		const int wstart_inter = max(w - siz_inter, 0);
		const int wend_inter = min(w + siz_inter, W);
		const int khstart_inter = hstart_inter - h + siz_inter;
		const int khend_inter = hend_inter - h + siz_inter;
		const int kwstart_inter = wstart_inter - w + siz_inter;
		const int kwend_inter = wend_inter - w + siz_inter;

		Dtype cumsum_inter = 0.;
		for (int i = khstart_inter; i < khend_inter; ++i) {
			for (int j = kwstart_inter; j < kwend_inter; ++j) {
				cumsum_inter += kernel_inter[i * size_inter + j];
			}
		}

		int fake_index_cover_inter = 0;
		int fake_index_stego_inter = 0;
		const float thres_inter = rand_location_inter[index] * cumsum_inter;
		cumsum_inter = 0;
		for (int i = khstart_inter; i < khend_inter; ++i) {
			for (int j = kwstart_inter; j < kwend_inter; ++j) {
				cumsum_inter += kernel_inter[i * size_inter + j];
				if (cumsum_inter >= thres_inter) {
					fake_index_cover_inter = (((2 * n + 0) * K + k) * H + h + i
							- siz_inter) * W + w + j - siz_inter;
					fake_index_stego_inter = (((2 * n + 1) * K + k) * H + h + i
							- siz_inter) * W + w + j - siz_inter;
					break;
				}
			}
		}

		if (rand_operation[index] > 0.5) {
			out[fake_index_stego_inter] = in[fake_index_cover_inter] + diff;
		} else {
			out[fake_index_stego_inter] = in[fake_index_cover_inter] - diff;
		}

	}
}

template<typename Dtype>
__global__ void FakeEmbedJpegInterIntra(const int n, const int K, const int H,
		const int W, const Dtype* kernel_inter, const Dtype* kernel_intra,
		const int size_inter, const int size_intra,
		const Dtype* rand_location_inter, const Dtype* rand_location_intra,
		const Dtype* rand_operation, const Dtype* in, Dtype* out) {
	CUDA_KERNEL_LOOP(index, n)
	{
		const int w = index % W;
		const int h = index / W % H;
		const int k = index / W / H % K;
		const int n = index / W / H / K; // n-th stego
		const int index_cover = (((2 * n + 0) * K + k) * H + h) * W + w;
		const int index_stego = (((2 * n + 1) * K + k) * H + h) * W + w;
		const int diff = in[index_cover] - in[index_stego];

		const int siz_inter = (size_inter - 1) / 2;
		const int hstart_inter = max(h - siz_inter, 0);
		const int hend_inter = min(h + siz_inter, H);
		const int wstart_inter = max(w - siz_inter, 0);
		const int wend_inter = min(w + siz_inter, W);
		const int khstart_inter = hstart_inter - h + siz_inter;
		const int khend_inter = hend_inter - h + siz_inter;
		const int kwstart_inter = wstart_inter - w + siz_inter;
		const int kwend_inter = wend_inter - w + siz_inter;
		Dtype cumsum_inter = 0.;
		for (int i = khstart_inter; i < khend_inter; ++i) {
			for (int j = kwstart_inter; j < kwend_inter; ++j) {
				cumsum_inter += kernel_inter[i * size_inter + j];
			}
		}
		int fake_index_h_inter = 0;
		int fake_index_w_inter = 0;
		const float thres_inter = rand_location_inter[index] * cumsum_inter;
		cumsum_inter = 0;
		for (int i = khstart_inter; i < khend_inter; ++i) {
			for (int j = kwstart_inter; j < kwend_inter; ++j) {
				cumsum_inter += kernel_inter[i * size_inter + j];
				if (cumsum_inter >= thres_inter) {
					fake_index_h_inter =  h + i - siz_inter;
					fake_index_w_inter =  w + j - siz_inter;
					break;
				}
			}
		}


		const int dct_w = index % 8;
		const int dct_h = index / 8 % 8;

		const int siz_intra = (size_intra - 1) / 2;
		const int hstart_intra = max(dct_h - siz_intra, 0);
		const int hend_intra = min(dct_h + siz_intra, 8);
		const int wstart_intra = max(dct_w - siz_intra, 0);
		const int wend_intra = min(dct_w + siz_intra, 8);
		const int khstart_intra = hstart_intra - dct_h + siz_intra;
		const int khend_intra = hend_intra - dct_h + siz_intra;
		const int kwstart_intra = wstart_intra - dct_w + siz_intra;
		const int kwend_intra = wend_intra - dct_w + siz_intra;

		Dtype cumsum_intra = 0.;
		for (int i = khstart_intra; i < khend_intra; ++i) {
			for (int j = kwstart_intra; j < kwend_intra; ++j) {
				cumsum_intra += kernel_intra[i * size_intra + j];
			}
		}
		int fake_index_cover = 0;
		int fake_index_stego = 0;
		const float thres_intra = rand_location_intra[index] * cumsum_intra;
		cumsum_intra = 0;
		for (int i = khstart_intra; i < khend_intra; ++i) {
			for (int j = kwstart_intra; j < kwend_intra; ++j) {
				cumsum_intra += kernel_intra[i * size_intra + j];
				if (cumsum_intra >= thres_intra) {
					fake_index_cover = (((2 * n + 0) * K
							+ (dct_h + i - siz_intra) * 8 + dct_w + j
							- siz_intra) * H + fake_index_h_inter) * W
							+ fake_index_w_inter;
					fake_index_stego = (((2 * n + 1) * K
							+ (dct_h + i - siz_intra) * 8 + dct_w + j
							- siz_intra) * H + fake_index_h_inter) * W
							+ fake_index_w_inter;
					break;
				}
			}
		}


		if (rand_operation[index] > 0.5) {
			out[fake_index_stego] = in[fake_index_cover] + diff;
		} else {
			out[fake_index_stego] = in[fake_index_cover] - diff;
		}

	}
}

template<typename Dtype>
void FakeEmbedJpegLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	if (this->phase_ == TRAIN) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = bottom[0]->mutable_gpu_data();
		const int count = bottom[0]->count() / 2;

		Blob<Dtype> rand_map_location_inter(num_ / 2, channels_, height_,
				width_);
		caffe_gpu_rng_uniform(rand_map_location_inter.count(), Dtype(0),
				Dtype(1), rand_map_location_inter.mutable_gpu_data());

		Blob<Dtype> rand_map_operation(num_ / 2, channels_, height_,
				width_);
		caffe_gpu_rng_uniform(rand_map_operation.count(), Dtype(0),
				Dtype(1), rand_map_operation.mutable_gpu_data());

		Blob<Dtype> temp(num_, channels_, height_, width_);

		for (int i = 0; i < num_; i += 2) {
			caffe_copy(count * 2 / num_, bottom_data + bottom[0]->offset(i),
					temp.mutable_gpu_data() + temp.offset(i));
			caffe_copy(count * 2 / num_, bottom_data + bottom[0]->offset(i),
					temp.mutable_gpu_data() + temp.offset(i + 1));
		}

		if (inter_only_) {
			FakeEmbedJpegInterOnly<Dtype> <<<CAFFE_GET_BLOCKS(count),
					CAFFE_CUDA_NUM_THREADS>>>(count, channels_, height_, width_,
					gaussian_kernel_inter_.gpu_data(), size_inter_,
					rand_map_location_inter.gpu_data(),
					rand_map_operation.gpu_data(), bottom_data,
					temp.mutable_gpu_data());
			CUDA_POST_KERNEL_CHECK;
		} else {
			Blob<Dtype> rand_map_location_intra(num_ / 2, channels_, height_,
					width_);
			caffe_gpu_rng_uniform(rand_map_location_intra.count(), Dtype(0),
					Dtype(1), rand_map_location_intra.mutable_gpu_data());

			FakeEmbedJpegInterIntra<Dtype> <<<CAFFE_GET_BLOCKS(count),
					CAFFE_CUDA_NUM_THREADS>>>(count, channels_, height_, width_,
					gaussian_kernel_inter_.gpu_data(),
					gaussian_kernel_intra_.gpu_data(), size_inter_, size_intra_,
					rand_map_location_inter.gpu_data(),
					rand_map_location_intra.gpu_data(),
					rand_map_operation.gpu_data(), bottom_data,
					temp.mutable_gpu_data());
			CUDA_POST_KERNEL_CHECK;
		}
		caffe_copy(temp.count(), temp.gpu_data(), top_data);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(FakeEmbedJpegLayer);

}  // namespace caffe
