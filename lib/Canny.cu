#include <cstdlib>
#include <vector>
#include <cuda.h>
#include "Canny.h"

namespace Canny {
    Matrix2D<float> generateGrayscaleImage(std::vector<unsigned char> originalImage, unsigned int width, unsigned int height) {
        auto expectedIntensityElements = (float*) malloc(sizeof(float));
        expectedIntensityElements[0] = 20;
        return Matrix2D<float>(expectedIntensityElements, 1, 1);
    }

    Matrix2D<float> generateGaussianKernel() {

    }
}
