// This is a modified version of jpeg reader to fit the need of our research.
// The original code can be obtained from: https://www.numberduck.com/Blog/?nPostId=2
// The modification also referenced https://github.com/victorvde/jpeg2png
// The Transpose(), HMirror(), and VMirror() functions are mainly inspired by "transupp.c" in the IJG libjpeg package


#include "caffe/util/jpeg_reader.hpp"
#include <stdlib.h>
#include <cstring>
#include <iostream>

JpegReader::JpegReader() {
	coef_ = NULL;
}

JpegReader::~JpegReader() {
	Cleanup();
}

const bool JpegReader::JpegRead(const char* szFileName, const int augment_index) {
	Cleanup();

	jpeg_decompress_struct cinfo;
	ErrorManager errorManager;

	FILE* pFile = fopen(szFileName, "rb");

	// set our custom error handler
	cinfo.err = jpeg_std_error(&errorManager.defaultErrorManager);
	errorManager.defaultErrorManager.error_exit = ErrorExit;
	errorManager.defaultErrorManager.output_message = OutputMessage;
	if (setjmp(errorManager.jumpBuffer)) {
		// We jump here on errorz
		Cleanup();
		jpeg_destroy_decompress(&cinfo);
		fclose(pFile);
		return false;
	}
	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, pFile);
	jpeg_read_header(&cinfo, TRUE);

	coef_ = new Coef();
	coef_->w = int(cinfo.image_width);
	coef_->h = int(cinfo.image_height);
	coef_->data = new int16_t[coef_->h * coef_->w];

	JQUANT_TBL *quant_tbl_ptr =
			cinfo.quant_tbl_ptrs[cinfo.comp_info[0].quant_tbl_no];
	memcpy(coef_->quant_table, quant_tbl_ptr->quantval, sizeof(uint16_t) * 64);

	jvirt_barray_ptr *coefs = jpeg_read_coefficients(&cinfo);
	int16_t *data = coef_->data;
	for (int y = 0; y < coef_->h / 8; y++) {
		JBLOCKARRAY b = cinfo.mem->access_virt_barray((j_common_ptr) &cinfo,
				coefs[0], y, 1, FALSE);
		for (int x = 0; x < coef_->w / 8; x++) {
			memcpy(data, b[0][x], 64 * sizeof(int16_t));
			data += 64;
		}
	}

	jpeg_destroy_decompress(&cinfo);
	fclose(pFile);

	if (augment_index >= 0) {
		int16_t *temp = new int16_t[coef_->h * coef_->w];
		data = coef_->data;
		switch (augment_index) {
		case 0:            // no change
			break;
		case 1:            // transpose + h_mirror
			Transpose(data, temp, coef_->h / 8, coef_->w / 8);
			HMirror(temp, data, coef_->h / 8, coef_->w / 8);
			break;
		case 2:            // h_mirror + v_mirror
			HMirror(data, temp, coef_->h / 8, coef_->w / 8);
			VMirror(temp, data, coef_->h / 8, coef_->w / 8);
			break;
		case 3:            // transpose + v_mirror
			Transpose(data, temp, coef_->h / 8, coef_->w / 8);
			VMirror(temp, data, coef_->h / 8, coef_->w / 8);
			break;
		case 4:            // h_mirror
			HMirror(data, temp, coef_->h / 8, coef_->w / 8);
			memcpy(data, temp, coef_->h * coef_->w * sizeof(int16_t));
			break;
		case 5:            // transpose + h_mirror + v_mirror
			Transpose(data, temp, coef_->h / 8, coef_->w / 8);
			HMirror(temp, data, coef_->h / 8, coef_->w / 8);
			VMirror(data, temp, coef_->h / 8, coef_->w / 8);
			memcpy(data, temp, coef_->h * coef_->w * sizeof(int16_t));
			break;
		case 6:            // v_mirror
			VMirror(data, temp, coef_->h / 8, coef_->w / 8);
			memcpy(data, temp, coef_->h * coef_->w * sizeof(int16_t));
			break;
		case 7:            // transpose
			Transpose(data, temp, coef_->h / 8, coef_->w / 8);
			memcpy(data, temp, coef_->h * coef_->w * sizeof(int16_t));
			break;
		default:            //
			std::cout << "Error\n";
			return false;
			break;
		}
		delete[] temp;
	}

	return true;
}

void JpegReader::Transpose(const int16_t* in, int16_t* out, const int BLK_H,
		const int BLK_W) {
	/* Transposing pixels within a block just requires transposing the
	 * DCT coefficients.*/
	for (int blk_h = 0; blk_h < BLK_H; blk_h++) {
		for (int blk_w = 0; blk_w < BLK_W; blk_w++) {
			for (int dct_h = 0; dct_h < 8; dct_h++) {
				for (int dct_w = 0; dct_w < 8; dct_w++) {
					out[(blk_w * BLK_H + blk_h) * 64 + dct_w * 8 + dct_h] =
							in[(blk_h * BLK_W + blk_w) * 64 + dct_h * 8 + dct_w];
				}
			}
		}
	}
}

void JpegReader::HMirror(const int16_t* in, int16_t* out, const int BLK_H,
		const int BLK_W) {
	/* Horizontal mirroring of DCT blocks is accomplished by swapping
	 * pairs of blocks.  Within a DCT block, we perform horizontal
	 * mirroring by changing the signs of odd-numbered columns.
	 */
	int new_blk_w = 0;
	for (int blk_h = 0; blk_h < BLK_H; blk_h++) {
		for (int blk_w = 0; blk_w < BLK_W; blk_w++) {
			new_blk_w = BLK_W - blk_w - 1;
			for (int dct_h = 0; dct_h < 8; dct_h++) {
				for (int dct_w = 0; dct_w < 8; dct_w += 2) {
					out[(blk_h * BLK_W + new_blk_w) * 64 + dct_h * 8 + dct_w] =
							in[(blk_h * BLK_W + blk_w) * 64 + dct_h * 8 + dct_w];
					dct_w++;
					out[(blk_h * BLK_W + new_blk_w) * 64 + dct_h * 8 + dct_w] =
							-in[(blk_h * BLK_W + blk_w) * 64 + dct_h * 8 + dct_w];
					dct_w--;
				}
			}
		}
	}
}

void JpegReader::VMirror(const int16_t* in, int16_t* out, const int BLK_H,
		const int BLK_W) {
	/* Within a DCT block, vertical mirroring is done by changing the signs
	 * of odd-numbered rows.
	 */
	int new_blk_h = 0;
	for (int blk_h = 0; blk_h < BLK_H; blk_h++) {
		new_blk_h = BLK_H - blk_h - 1;
		for (int blk_w = 0; blk_w < BLK_W; blk_w++) {
			for (int dct_h = 0; dct_h < 8; dct_h += 2) {
				for (int dct_w = 0; dct_w < 8; dct_w++) {
					out[(new_blk_h * BLK_W + blk_w) * 64 + dct_h * 8 + dct_w] =
							in[(blk_h * BLK_W + blk_w) * 64 + dct_h * 8 + dct_w];
					dct_h++;
					out[(new_blk_h * BLK_W + blk_w) * 64 + dct_h * 8 + dct_w] =
							-in[(blk_h * BLK_W + blk_w) * 64 + dct_h * 8 + dct_w];
					dct_h--;
				}
			}
		}
	}
}

const int16_t* JpegReader::GetQuantizedCoefficients() {
	return coef_->data;
}

const int JpegReader::GetImageHeight() {
	return coef_->h;
}

const int JpegReader::GetImageWidth() {
	return coef_->w;
}

void JpegReader::Cleanup() {
	if (coef_) {
		delete[] coef_->data;
		delete coef_;
		coef_ = NULL;
	}
}

void JpegReader::ErrorExit(j_common_ptr cinfo) {
	// cinfo->err is actually a pointer to my_error_mgr.defaultErrorManager, since pub
	// is the first element of my_error_mgr we can do a sneaky cast
	ErrorManager* pErrorManager = (ErrorManager*) cinfo->err;
	(*cinfo->err->output_message)(cinfo); // print error message (actually disabled below)
	longjmp(pErrorManager->jumpBuffer, 1);
}

void JpegReader :: OutputMessage(j_common_ptr cinfo)
{
	// disable error messages
	/*char buffer[JMSG_LENGTH_MAX];
	(*cinfo->err->format_message) (cinfo, buffer);
	fprintf(stderr, "%s\n", buffer);*/
}
