#ifndef SOFT354_CUDA_CANNY_H
#define SOFT354_CUDA_CANNY_H

#include <vector>
#include "Matrix2D.h"

namespace Canny {
    std::vector<unsigned char> generateGrayscaleImage(std::vector<unsigned char> originalImage, unsigned int pixelWidth, unsigned int pixelHeight);

    Matrix2D<float> generateGaussianKernel();
}

#endif //SOFT354_CUDA_CANNY_H
