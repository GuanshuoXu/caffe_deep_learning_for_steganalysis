// This is the input layer for jpeg steganalysis.
// This layer can perform per-epoch data shuffling.
// This layer makes sure cover and stego form pairs within each mini-batch.
// This layer only accepts '.jpg' data.
// This layer adopts "jpeglib" to read .jpg files and outputs BDCT coefficients.
// JPEG applies 8x8 DCT to each pixel block of the original image with stride 8. So the output of this layer is 64 downsized (by 8 in height and width) BDCT sub-images.
// This layer makes sure random mirroring and rotation is synchronized for each cover-stego pair.
// Mirrorings and rotations are done in the BDCT domain instead of in the spatial domain in order to reserve the possibility of cross-domain (DCT and spatial) research for jpeg steganalysis in the future.

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_steganalysis_jpeg_dct_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
ImageDataSteganalysisJpegDctLayer<Dtype>::~ImageDataSteganalysisJpegDctLayer<Dtype>() {
	this->StopInternalThread();
}

template<typename Dtype>
void ImageDataSteganalysisJpegDctLayer<Dtype>::PerformTransform(
		Blob<Dtype>* transformed_blob) {

	string filename_cover = cover_dir_ + lines_[lines_id_] + image_fmt_;
	JpegReader jpeg_reader_cover;
	string filename_stego = stego_dir_ + lines_[lines_id_] + image_fmt_;
	JpegReader jpeg_reader_stego;

	if (rand_mirror_rotate_) {
		int a1, a2, a3;
		caffe_rng_bernoulli(1, Dtype(0.5), &a1);
		caffe_rng_bernoulli(1, Dtype(0.5), &a2);
		caffe_rng_bernoulli(1, Dtype(0.5), &a3);
		int augment = 0;
		augment |= a1 << 2;
		augment |= a2 << 1;
		augment |= a3 << 0;
		CHECK_EQ(jpeg_reader_cover.JpegRead(filename_cover.c_str(), augment), 1)<< "Failed to read JPEG coefficients.";
		CHECK_EQ(jpeg_reader_stego.JpegRead(filename_stego.c_str(), augment), 1)<< "Failed to read JPEG coefficients.";
	} else {
		CHECK_EQ(jpeg_reader_cover.JpegRead(filename_cover.c_str()), 1)<< "Failed to read JPEG coefficients.";
		CHECK_EQ(jpeg_reader_stego.JpegRead(filename_stego.c_str()), 1)<< "Failed to read JPEG coefficients.";
	}

	const int16_t* ucoef_cover = jpeg_reader_cover.GetQuantizedCoefficients();
	const int16_t* ucoef_stego = jpeg_reader_stego.GetQuantizedCoefficients();

	const int channels = transformed_blob->channels();
	const int height = transformed_blob->height();
	const int width = transformed_blob->width();
	Dtype* transformed_data = transformed_blob->mutable_cpu_data();
	int top_index;
	int offset = height * width * channels;
	int h, w, c;
	for (int i = 0; i < offset; i++) {
		c = i % 64;
		h = (i / 64) / width;
		w = (i / 64) % width;
		top_index = (c * height + h) * width + w;
		transformed_data[top_index] = (Dtype)(ucoef_cover[i]);
		transformed_data[top_index + offset] = (Dtype)(ucoef_stego[i]);
	}

}

template<typename Dtype>
void ImageDataSteganalysisJpegDctLayer<Dtype>::DataLayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	image_fmt_ = ".jpg";

	cover_dir_ =
			this->layer_param_.image_data_steganalysis_jpeg_dct_param().cover_dir();
	if (cover_dir_.at(cover_dir_.length()-1) != '/') {
		cover_dir_.at(cover_dir_.length()-1) = '/';
	}
	stego_dir_ =
			this->layer_param_.image_data_steganalysis_jpeg_dct_param().stego_dir();
	if (stego_dir_.at(stego_dir_.length() - 1) != '/') {
		stego_dir_.at(stego_dir_.length() - 1) = '/';
	}

	// random rotation and mirroring for jpeg dct coefficients
	rand_mirror_rotate_ =
			this->layer_param_.image_data_steganalysis_jpeg_dct_param().rand_mirror_rotate();
	// not allowed for testing
	if (this->phase_ == TEST) {
		rand_mirror_rotate_ = false;
	}

	// Read the file with filenames and labels
	const string& source =
			this->layer_param_.image_data_steganalysis_jpeg_dct_param().source();
	LOG(INFO)<< "Opening file " << source;
	std::ifstream infile(source.c_str());
	string image_id;
	while (infile >> image_id) {
		lines_.push_back(image_id);
	}

	// randomly shuffle data
	shuffle_ = this->layer_param_.image_data_steganalysis_jpeg_dct_param().shuffle();
	if (shuffle_) {
		LOG(INFO)<< "Shuffling data";
		const unsigned int prefetch_rng_seed = caffe_rng_rand();
		prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
		ShuffleImages();
	}
	LOG(INFO)<< "A total of " << 2*lines_.size() << " images.";

	lines_id_ = 0;
	// read an image, and use the size of it to initialize the top blob.
	string filename = cover_dir_ + lines_[lines_id_] + image_fmt_;
	JpegReader jpeg_reader;
	CHECK_EQ(jpeg_reader.JpegRead(filename.c_str()), 1)<< "Failed to read JPEG coefficients.";
	top_shape_.push_back(1*2); // to hold both cover and stego
	top_shape_.push_back(64); // 64 DCT modes
	top_shape_.push_back(jpeg_reader.GetImageHeight()/8); // mode-wise storage
	top_shape_.push_back(jpeg_reader.GetImageWidth()/8);
	if (jpeg_reader.GetImageHeight()%8!=0 || jpeg_reader.GetImageWidth()%8!=0) {LOG(ERROR)<< "Image size must be multiple of 8";}
	// temp
	this->transformed_data_.Reshape(top_shape_);

	// Reshape prefetch_data and top[0] according to the batch_size.
	batch_size_ =
			this->layer_param_.image_data_steganalysis_jpeg_dct_param().batch_size();
	CHECK_GT(batch_size_, 0)<< "Positive batch size required";
	CHECK_EQ(batch_size_%2, 0)<< "Even batch size required";
	top_shape_[0] = batch_size_;
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].data_.Reshape(top_shape_);
	}
	top[0]->Reshape(top_shape_);

	// label
	vector<int> label_shape(1, batch_size_);
	top[1]->Reshape(label_shape);
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].label_.Reshape(label_shape);
	}
}

template<typename Dtype>
void ImageDataSteganalysisJpegDctLayer<Dtype>::ShuffleImages() {
	caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template<typename Dtype>
void ImageDataSteganalysisJpegDctLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

	CHECK(batch->data_.count());
	CHECK(this->transformed_data_.count());

	Dtype* prefetch_data = batch->data_.mutable_cpu_data();
	Dtype* prefetch_label = batch->label_.mutable_cpu_data();

	const int lines_size = lines_.size();
	for (int item_id = 0; item_id < batch_size_/2; ++item_id) {
		// get a blob
		CHECK_GT(lines_size, lines_id_);
		int offset = batch->data_.offset(item_id*2);
		this->transformed_data_.set_cpu_data(prefetch_data + offset);
		PerformTransform(&(this->transformed_data_));
		prefetch_label[item_id*2] = 0;
		prefetch_label[item_id*2+1] = 1;
		lines_id_++;
		if (lines_id_ >= lines_size) {
			// We have reached the end. Restart from the first.
			DLOG(INFO)<< "Restarting data prefetching from start.";
			lines_id_ = 0;
			if (shuffle_) {
				ShuffleImages();
			}
		}
	}

}

INSTANTIATE_CLASS(ImageDataSteganalysisJpegDctLayer);
REGISTER_LAYER_CLASS(ImageDataSteganalysisJpegDct);

}  // namespace caffe

