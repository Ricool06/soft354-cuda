#include <cstdlib>
#include <vector>
#include <cuda.h>
#include "HandleError.h"
#include "Canny.h"

namespace Canny {
    namespace {
        __device__
        unsigned char calculateAverageIntensityOfPixel(unsigned char *originalImage, size_t firstByteIndex) {
            // Implicit cast from float to byte
            return ((originalImage[firstByteIndex]
                     + originalImage[firstByteIndex + 1]
                     + originalImage[firstByteIndex + 2]) / 3) // Mean of RGB
                     * ((originalImage[firstByteIndex + 3]) / 255.0f); // Opacity
        }

        __global__
        void generateGrayscaleImageKernel(unsigned char *resultImage, unsigned char *originalImage, size_t pixelCount) {
            size_t pixelId = blockIdx.x * blockDim.x + threadIdx.x;

            if (pixelId < pixelCount) {
                size_t byteIndex = pixelId * 4;
                unsigned char intensity = calculateAverageIntensityOfPixel(originalImage, byteIndex);
                resultImage[byteIndex] = intensity;
                resultImage[byteIndex + 1] = intensity;
                resultImage[byteIndex + 2] = intensity;
                resultImage[byteIndex + 3] = 255;
            }
        }
    }

    /**
     *
     * @param originalImage The pixels of the input image as a vector of bytes, with every consecutive 4 representing a single RGBA pixel
     * @param pixelWidth The width in pixels of the image
     * @param pixelHeight The height in pixels of the image
     * @return Matrix2D representing 0-255 intensity of each pixel in the original image
     */
    std::vector<unsigned char> generateGrayscaleImage(std::vector<unsigned char> originalImage, unsigned int pixelWidth, unsigned int pixelHeight) {
        unsigned char *dOriginalImage, *dResultImage;
        size_t pixelCount = pixelWidth * pixelHeight;
        size_t imageSizeInBytes = pixelCount * 4 * sizeof(unsigned char); // 4 * size of a byte * image dimensions

        checkCudaCall(cudaMalloc(&dOriginalImage, imageSizeInBytes));
        checkCudaCall(cudaMalloc(&dResultImage, imageSizeInBytes));

        checkCudaCall(cudaMemcpy(dOriginalImage, originalImage.data(), imageSizeInBytes, cudaMemcpyHostToDevice));

        size_t blockSize = 1024;
        size_t blockCount = (pixelCount + blockSize - 1) / blockSize;

        generateGrayscaleImageKernel<<<blockCount, blockSize>>>(dResultImage, dOriginalImage, pixelCount);
        auto resultImage = (unsigned char*) malloc(imageSizeInBytes);
        cudaDeviceSynchronize();

        checkCudaCall(cudaMemcpy(resultImage, dResultImage, imageSizeInBytes, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        checkCudaCall(cudaFree(dOriginalImage));
        checkCudaCall(cudaFree(dResultImage));

        std::vector<unsigned char> result (resultImage, resultImage + imageSizeInBytes);
        return result;
    }

    Matrix2D<float> generateGaussianKernel() {

    }
}
