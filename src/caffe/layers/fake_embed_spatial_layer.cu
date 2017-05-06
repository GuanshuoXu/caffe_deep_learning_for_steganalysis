#include "caffe/layers/fake_embed_spatial_layer.hpp"

namespace caffe {

template<typename Dtype>
__global__ void FakeEmbedSpatial(const int n, const int K, const int H,
		const int W, const Dtype* kernel, const int size,
		const Dtype* rand_location, const Dtype* rand_operation,
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

		const int siz = (size - 1) / 2;
		const int hstart = max(h - siz, 0);
		const int hend = min(h + siz, H);
		const int wstart = max(w - siz, 0);
		const int wend = min(w + siz, W);
		const int khstart = hstart - h + siz;
		const int khend = hend - h + siz;
		const int kwstart = wstart - w + siz;
		const int kwend = wend - w + siz;

		Dtype cumsum = 0.;
		for (int i = khstart; i < khend; ++i) {
			for (int j = kwstart; j < kwend; ++j) {
				cumsum += kernel[i * size + j];
			}
		}

		int fake_index_cover = 0;
		int fake_index_stego = 0;
		const float thres = rand_location[index] * cumsum;
		cumsum = 0;
		for (int i = khstart; i < khend; ++i) {
			for (int j = kwstart; j < kwend; ++j) {
				cumsum += kernel[i * size + j];
				if (cumsum >= thres) {
					fake_index_cover = (((2 * n + 0) * K + k) * H + h + i - siz)
							* W + w + j - siz;
					fake_index_stego = (((2 * n + 1) * K + k) * H + h + i - siz)
							* W + w + j - siz;
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
void FakeEmbedSpatialLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	if (this->phase_ == TRAIN) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = bottom[0]->mutable_gpu_data();
		const int count = bottom[0]->count() / 2;

		Blob<Dtype> rand_map_location(num_ / 2, channels_, height_, width_);
		caffe_gpu_rng_uniform(rand_map_location.count(), Dtype(0), Dtype(1),
				rand_map_location.mutable_gpu_data());

		Blob<Dtype> rand_map_operation(num_ / 2, channels_, height_, width_);
		caffe_gpu_rng_uniform(rand_map_operation.count(), Dtype(0), Dtype(1),
				rand_map_operation.mutable_gpu_data());

		Blob<Dtype> temp(num_, channels_, height_, width_);

		for (int i = 0; i < num_; i += 2) {
			caffe_copy(count * 2 / num_, bottom_data + bottom[0]->offset(i),
					temp.mutable_gpu_data() + temp.offset(i));
			caffe_copy(count * 2 / num_, bottom_data + bottom[0]->offset(i),
					temp.mutable_gpu_data() + temp.offset(i + 1));
		}

		FakeEmbedSpatial<Dtype> <<<CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS>>>(count, channels_, height_, width_,
				gaussian_kernel_.gpu_data(), size_,
				rand_map_location.gpu_data(), rand_map_operation.gpu_data(),
				bottom_data, temp.mutable_gpu_data());
		caffe_copy(temp.count(), temp.gpu_data(), top_data);

		CUDA_POST_KERNEL_CHECK;
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(FakeEmbedSpatialLayer);

}  // namespace caffe
