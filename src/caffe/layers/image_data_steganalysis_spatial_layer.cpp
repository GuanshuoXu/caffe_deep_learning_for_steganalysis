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
#include "caffe/layers/image_data_steganalysis_spatial_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
ImageDataSteganalysisSpatialLayer<Dtype>::~ImageDataSteganalysisSpatialLayer<Dtype>() {
	this->StopInternalThread();
}

template<typename Dtype>
void ImageDataSteganalysisSpatialLayer<Dtype>::PerformTransform(
		Blob<Dtype>* transformed_blob) {
	string filename_cover = cover_dir_ + lines_[lines_id_] + image_fmt_;
	cv::Mat cv_img_cover = cv::imread(filename_cover, CV_LOAD_IMAGE_GRAYSCALE);
	if (!cv_img_cover.data) {
		LOG(ERROR)<< "Could not open or find file " << filename_cover;
	}
	string filename_stego = stego_dir_ + lines_[lines_id_] + image_fmt_;
	cv::Mat cv_img_stego = cv::imread(filename_stego, CV_LOAD_IMAGE_GRAYSCALE);
	if (!cv_img_stego.data) {
		LOG(ERROR)<< "Could not open or find file " << filename_stego;
	}
	Dtype do_mirror, do_rotate;
	int idx_mirror = 0; int idx_rotate = 0;
	if (this->phase_ == TEST) {
		idx_mirror = this->layer_param_.image_data_steganalysis_spatial_param().mirror_rotate() / 4;
		idx_rotate = this->layer_param_.image_data_steganalysis_spatial_param().mirror_rotate() % 4;
	} else {
		if (rand_mirror_) {
			caffe_rng_uniform(1, (Dtype) 0, (Dtype) 1, &do_mirror);
		}

		if (rand_rotate_) {
			caffe_rng_uniform(1, (Dtype) 0, (Dtype) 1, &do_rotate);
		}
	}
	const int channels = transformed_blob->channels();
	const int height = transformed_blob->height();
	const int width = transformed_blob->width();

	Dtype* transformed_data = transformed_blob->mutable_cpu_data();
	int top_index;
	int offset = height * width * channels;
	for (int h = 0; h < height; ++h) {
		const uchar* ptr_cover = cv_img_cover.ptr<uchar>(h);
		const uchar* ptr_stego = cv_img_stego.ptr<uchar>(h);
		int img_index_cover = 0;
		int img_index_stego = 0;
		for (int w = 0; w < width; ++w) {
			for (int c = 0; c < channels; ++c) {
				int h_idx = h;
				int w_idx = w;
				if (this->phase_ == TEST) {
					if (idx_mirror) {
						w_idx = width - 1 - w;
					}
					if (idx_rotate == 1) {
						int temp = w_idx;
						w_idx = height - 1 - h_idx;
						h_idx = temp;
					} else if (idx_rotate == 2) {
						w_idx = width - 1 - w_idx;
						h_idx = height - 1 - h_idx;
					} else if (idx_rotate == 3) {
						int temp = h_idx;
						h_idx = width - 1 - w_idx;
						w_idx = temp;
					}
				} else {
					if (rand_mirror_ && do_mirror >= 0.5) {
						w_idx = width - 1 - w;
					}
					if (rand_rotate_ && do_rotate >= 0.25 && do_rotate < 0.5) {
						int temp = w_idx;
						w_idx = height - 1 - h_idx;
						h_idx = temp;
					} else if (rand_rotate_ && do_rotate >= 0.5
							&& do_rotate < 0.75) {
						w_idx = width - 1 - w_idx;
						h_idx = height - 1 - h_idx;
					} else if (rand_rotate_ && do_rotate >= 0.75) {
						int temp = h_idx;
						h_idx = width - 1 - w_idx;
						w_idx = temp;
					}
				}
				top_index = (c * height + h_idx) * width + w_idx;

				Dtype pixel_cover = static_cast<Dtype>(ptr_cover[img_index_cover++]);
				Dtype pixel_stego = static_cast<Dtype>(ptr_stego[img_index_stego++]);
				transformed_data[top_index] = pixel_cover;
				transformed_data[top_index + offset] = pixel_stego;
			}
		}
	}

}

template<typename Dtype>
void ImageDataSteganalysisSpatialLayer<Dtype>::DataLayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	cover_dir_ =
			this->layer_param_.image_data_steganalysis_spatial_param().cover_dir();
	if (cover_dir_.at(cover_dir_.length()-1) != '/') {
		cover_dir_.at(cover_dir_.length()-1) = '/';
	}
	stego_dir_ =
			this->layer_param_.image_data_steganalysis_spatial_param().stego_dir();
	if (stego_dir_.at(stego_dir_.length() - 1) != '/') {
		stego_dir_.at(stego_dir_.length() - 1) = '/';
	}
	image_fmt_ =
			this->layer_param_.image_data_steganalysis_spatial_param().image_fmt();
	if (image_fmt_[0] != '.') {
		image_fmt_.insert(0,1,'.');
	}

	rand_mirror_ =
			this->layer_param_.image_data_steganalysis_spatial_param().rand_mirror();
	rand_rotate_ =
			this->layer_param_.image_data_steganalysis_spatial_param().rand_rotate();

	// Read the file with filenames and labels
	const string& source =
			this->layer_param_.image_data_steganalysis_spatial_param().source();
	LOG(INFO)<< "Opening file " << source;
	std::ifstream infile(source.c_str());
	string image_id;
	while (infile >> image_id) {
		lines_.push_back(image_id);
	}

	// randomly shuffle data
	shuffle_ = this->layer_param_.image_data_steganalysis_spatial_param().shuffle();
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
	cv::Mat cv_img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (!cv_img.data) {
		LOG(ERROR)<< "Could not open or find file " << filename;
	}
	top_shape_.push_back(1*2); // to hold both cover and stego
	top_shape_.push_back(1);
	top_shape_.push_back(cv_img.rows);
	top_shape_.push_back(cv_img.cols);

	// temp
	this->transformed_data_.Reshape(top_shape_);

	// Reshape prefetch_data and top[0] according to the batch_size.
	batch_size_ =
			this->layer_param_.image_data_steganalysis_spatial_param().batch_size();
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

	CHECK_GE(this->layer_param_.image_data_steganalysis_spatial_param().mirror_rotate(),0);
	CHECK_LE(this->layer_param_.image_data_steganalysis_spatial_param().mirror_rotate(),7);
}

template<typename Dtype>
void ImageDataSteganalysisSpatialLayer<Dtype>::ShuffleImages() {
	caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template<typename Dtype>
void ImageDataSteganalysisSpatialLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

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

INSTANTIATE_CLASS(ImageDataSteganalysisSpatialLayer);
REGISTER_LAYER_CLASS(ImageDataSteganalysisSpatial);

}  // namespace caffe

