#ifndef SOFT354_CUDA_CANNY_H
#define SOFT354_CUDA_CANNY_H

#include <vector>
#include "Matrix2D.h"

namespace Canny {
    Matrix2D<float> generateGrayscaleImage(std::vector<unsigned char> originalImage, unsigned int width, unsigned int height);

    Matrix2D<float> generateGaussianKernel();
}

#endif //SOFT354_CUDA_CANNY_H
