// This is a modified version of jpeg reader to fit the need of our research.
// The original code can be obtained from: https://www.numberduck.com/Blog/?nPostId=2
// The modification also referenced https://github.com/victorvde/jpeg2png
// The Transpose(), HMirror(), and VMirror() functions are mainly inspired by "transupp.c" in the IJG libjpeg package

#ifndef JPEG_READER_H
#define JPEG_READER_H
#include <stdio.h>
#include "jpeglib.h"
#include <setjmp.h>
#include <stdint.h>

class JpegReader {
public:
	struct Coef {
		int h;
		int w;
		// DCT coefficients
		int16_t *data;
		// quantization table
		uint16_t quant_table[64];
	};

	JpegReader();
	~JpegReader();

	const bool JpegRead(const char* szFileName, const int augment_index = -1);
	const int16_t* GetQuantizedCoefficients();
	const int GetImageHeight();
	const int GetImageWidth();

private:
	Coef* coef_;
	void Cleanup();

	// work in BDCT domain
	void Transpose(const int16_t* in, int16_t* out, const int BLK_H, const int BLK_W);
	void HMirror(const int16_t* in, int16_t* out, const int BLK_H, const int BLK_W);
	void VMirror(const int16_t* in, int16_t* out, const int BLK_H, const int BLK_W);

	struct ErrorManager {
		jpeg_error_mgr defaultErrorManager;
		jmp_buf jumpBuffer;
	};

	static void ErrorExit(j_common_ptr cinfo);
	static void OutputMessage(j_common_ptr cinfo);
};
#endif
