// This is the input layer for jpeg steganalysis.
// This layer can perform per-epoch data shuffling.
// This layer makes sure cover and stego form pairs within each mini-batch.
// This layer only accepts '.jpg' data.
// This layer adopts "jpeglib" to read .jpg files and outputs BDCT coefficients.
// JPEG applies 8x8 DCT to each pixel block of the original image with stride 8. So the output of this layer is 64 downsized (by 8 in height and width) BDCT sub-images.
// This layer makes sure random mirroring and rotation is synchronized for each cover-stego pair.
// Mirrorings and rotations are done in the BDCT domain instead of in the spatial domain in order to reserve the possibility of cross-domain (DCT and spatial) research for jpeg steganalysis in the future.

#ifndef CAFFE_IMAGE_DATA_STEGANALYSIS_JPEG_DCT_LAYER_HPP_
#define CAFFE_IMAGE_DATA_STEGANALYSIS_JPEG_DCT_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <opencv2/core/core.hpp>

#include "caffe/util/jpeg_reader.hpp"

namespace caffe {

template<typename Dtype>
class ImageDataSteganalysisJpegDctLayer: public BasePrefetchingDataLayer<Dtype> {
public:
	explicit ImageDataSteganalysisJpegDctLayer(const LayerParameter& param) :
			BasePrefetchingDataLayer<Dtype>(param),
			lines_id_(0),
			shuffle_(true),
			batch_size_(0),
			rand_mirror_rotate_(false) {
	}
	virtual ~ImageDataSteganalysisJpegDctLayer();
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const {
		return "ImageDataSteganalysisJpegDct";
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
	bool rand_mirror_rotate_;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_STEGANALYSIS_JPEG_DCT_HPP_

