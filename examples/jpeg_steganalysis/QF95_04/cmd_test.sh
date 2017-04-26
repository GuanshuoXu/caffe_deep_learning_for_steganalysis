#!/usr/bin/env sh

/home/forensics/caffe-research/IH17/steganalysis-jpeg-ih17/build/tools/caffe test -model CNN_test.prototxt -weights inference90000.caffemodel -gpu 1 -iterations 200 -prob prob90000.txt
/home/forensics/caffe-research/IH17/steganalysis-jpeg-ih17/build/tools/caffe test -model CNN_test.prototxt -weights inference85000.caffemodel -gpu 1 -iterations 200 -prob prob85000.txt
/home/forensics/caffe-research/IH17/steganalysis-jpeg-ih17/build/tools/caffe test -model CNN_test.prototxt -weights inference80000.caffemodel -gpu 1 -iterations 200 -prob prob80000.txt
