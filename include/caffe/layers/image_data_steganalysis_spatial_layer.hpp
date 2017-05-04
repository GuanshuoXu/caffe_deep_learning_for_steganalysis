#ifndef CAFFE_IMAGE_DATA_STEGANALYSIS_SPATIAL_LAYER_HPP_
#define CAFFE_IMAGE_DATA_STEGANALYSIS_SPATIAL_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <opencv2/core/core.hpp>

namespace caffe {

template<typename Dtype>
class ImageDataSteganalysisSpatialLayer: public BasePrefetchingDataLayer<Dtype> {
public:
	explicit ImageDataSteganalysisSpatialLayer(const LayerParameter& param) :
			BasePrefetchingDataLayer<Dtype>(param),
			lines_id_(0),
			shuffle_(true),
			batch_size_(0),
			rand_mirror_(true),
			rand_rotate_(true) {
	}
	virtual ~ImageDataSteganalysisSpatialLayer();
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const {
		return "ImageDataSteganalysisSpatial";
	}
	virtual inline int ExactNumBottomBlobs() const {
		return 0;
	}
	virtual inline int ExactNumTopBlobs() const {
		return 2;
	}

protected:
	shared_ptr<Caffe::RNG> prefetch_rng_;
	virtual void ShuffleImages();
	virtual void load_batch(Batch<Dtype>* batch);
	virtual void PerformTransform(Blob<Dtype>* transformed_blob);

	vector<std::string> lines_;
	int lines_id_;
	bool shuffle_; // random permute for each epoch
	int batch_size_;
	std::string cover_dir_;
	std::string stego_dir_;
	std::string image_fmt_;
	std::vector<int> top_shape_;
	bool rand_mirror_;
	bool rand_rotate_;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_STEGANALYSIS_SPATIAL_HPP_
