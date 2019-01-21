#include <cstdlib>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <cuda.h>
#include "HandleError.h"
#include "Canny.h"

namespace Canny {
    namespace {
        __device__
        unsigned char calculateAverageIntensityOfPixel(const unsigned char *originalImage, const size_t firstByteIndex) {
            // Implicit cast from float to byte
            return ((originalImage[firstByteIndex]
                     + originalImage[firstByteIndex + 1]
                     + originalImage[firstByteIndex + 2]) / 3) // Mean of RGB
                   * ((originalImage[firstByteIndex + 3]) / 255.0f); // Opacity
        }

        __device__
        size_t clampIndex(const size_t index, const size_t inclusiveMinimum, const size_t exclusiveMaximum) {
            return max(inclusiveMinimum, min(index, exclusiveMaximum - 1));
        }

        __device__
        unsigned char clampByte(const unsigned char byte) {
            return max(0, min(byte, 255));
        }

        __global__
        void generateGrayscaleImageKernel(unsigned char *resultImage, const unsigned char *originalImage, const size_t pixelCount) {
            size_t pixelId = blockIdx.x * blockDim.x + threadIdx.x;

            if (pixelId < pixelCount) {
                size_t byteIndex = pixelId * 4;
                unsigned char intensity = calculateAverageIntensityOfPixel(originalImage, byteIndex);
                resultImage[byteIndex] = intensity;
                resultImage[byteIndex + 1] = intensity;
                resultImage[byteIndex + 2] = intensity;
                resultImage[byteIndex + 3] = originalImage[byteIndex + 3];
            }
        }

        __global__
        void generateGaussianBlurredImageKernel(unsigned char *resultImage, const unsigned char *originalImage, const unsigned int pixelWidth, const unsigned int pixelHeight, const Matrix2D<float> gaussianKernel) {
            size_t centrePixelX = blockIdx.x * blockDim.x + threadIdx.x;
            size_t centrePixelY = blockIdx.y * blockDim.y + threadIdx.y;

            if (centrePixelX < pixelWidth && centrePixelY < pixelHeight) {
                // Store new values as float vars to reduce global memory access, but also to reduce colour loss from
                // too frequently casting floats to unsigned chars
                float newRed = 0.0f, newGreen = 0.0f, newBlue = 0.0f, newAlpha = 0.0f;
                size_t index = centrePixelY * 4 * pixelWidth + (centrePixelX * 4);

                for (size_t kernelY = 0; kernelY < gaussianKernel.height; ++kernelY) {
                    // It is conventional in an even dimensioned Gaussian kernel to place the extra element before the 0
                    // i.e. {-2, -1, 0 -1} is conventional
                    // Therefore, integer division is okay as it effectively floors the real number
                    size_t pixelOffsetY = kernelY - (gaussianKernel.height / 2);
                    size_t blendPixelY = clampIndex(centrePixelY + pixelOffsetY, 0, pixelHeight);

                    for (size_t kernelX = 0; kernelX < gaussianKernel.width; ++kernelX) {
                        size_t pixelOffsetX = kernelX - (gaussianKernel.width / 2);
                        size_t blendPixelX = clampIndex(centrePixelX + pixelOffsetX, 0, pixelWidth);
                        size_t blendPixelIndex = blendPixelY * 4 * pixelWidth + (blendPixelX * 4);

                        size_t kernelIndex = (kernelY * gaussianKernel.width) + kernelX;

                        newRed += originalImage[blendPixelIndex] * gaussianKernel.elements[kernelIndex];
                        newGreen += originalImage[blendPixelIndex + 1] * gaussianKernel.elements[kernelIndex];
                        newBlue += originalImage[blendPixelIndex + 2] * gaussianKernel.elements[kernelIndex];
                        newAlpha += originalImage[blendPixelIndex + 3] * gaussianKernel.elements[kernelIndex];
                    }
                }

                resultImage[index] = static_cast<unsigned char>(newRed);
                resultImage[index + 1] = static_cast<unsigned char>(newGreen);
                resultImage[index + 2] = static_cast<unsigned char>(newBlue);
                resultImage[index + 3] = static_cast<unsigned char>(newAlpha);
            }
        }

        __global__
        void generateGaussianFilterKernel(Matrix2D<float> gaussianKernel, const float standardDeviation) {
            size_t column = blockIdx.x * blockDim.x + threadIdx.x;
            size_t row = blockIdx.y * blockDim.y + threadIdx.y;

            __shared__ float sum;

            if (column < gaussianKernel.width && row < gaussianKernel.height) {
                size_t index = row * gaussianKernel.width + column;
                size_t x = column - (gaussianKernel.width / 2);
                size_t y = row - (gaussianKernel.height / 2);

                float sumOfX2AndY2 = (x * x) + (y * y);
                float twoSigmaSquared = 2.0f * standardDeviation * standardDeviation;
                gaussianKernel.elements[index] = static_cast<float>(exp(-sumOfX2AndY2 / twoSigmaSquared) / (M_PI * twoSigmaSquared));

                atomicAdd(&sum, gaussianKernel.elements[index]);
                __syncthreads();

                // Normalize Gaussian kernel (meaning all elements add up to 1)
                gaussianKernel.elements[index] /= sum;

                // Wait until all threads have normalized their Gaussian kernel element,
                // then zero the sum so subsequent runs of this kernel can do so safely.
                __syncthreads();
                sum = 0.0f;
            }
        }
    }

    /**
     * Generates a greyscale version of the given image.
     * @param originalImage The pixels of the input image as a vector of bytes, with every consecutive 4 representing a single RGBA pixel
     * @param pixelWidth The width in pixels of the image
     * @param pixelHeight The height in pixels of the image
     * @return vector<unsigned char> byte vector representing a grayscale, opaque version of the original image
     */
    std::vector<unsigned char> generateGreyscaleImage(const std::vector<unsigned char> &originalImage, unsigned int pixelWidth, unsigned int pixelHeight) {
        unsigned char *dOriginalImage, *dResultImage;
        size_t pixelCount = pixelWidth * pixelHeight;
        size_t imageSizeInBytes = pixelCount * 4 * sizeof(unsigned char); // 4 * size of a byte * image dimensions

        checkCudaCall(cudaMalloc(&dOriginalImage, imageSizeInBytes));
        checkCudaCall(cudaMalloc(&dResultImage, imageSizeInBytes));

        checkCudaCall(cudaMemcpy(dOriginalImage, originalImage.data(), imageSizeInBytes, cudaMemcpyHostToDevice));

        size_t blockSize = 1024;
        size_t blockCount = (pixelCount + blockSize - 1) / blockSize;

        generateGrayscaleImageKernel<<<blockCount, blockSize>>>(dResultImage, dOriginalImage, pixelCount);
        auto resultImage = (unsigned char *) malloc(imageSizeInBytes);

        checkCudaCall(cudaMemcpy(resultImage, dResultImage, imageSizeInBytes, cudaMemcpyDeviceToHost));

        checkCudaCall(cudaFree(dOriginalImage));
        checkCudaCall(cudaFree(dResultImage));

        std::vector<unsigned char> result(resultImage, resultImage + (pixelCount * 4));
        free(resultImage);
        return result;
    }

    /**
     * Applies a Gaussian blur filter to the given image
     * @param originalImage The image to blur
     * @param pixelWidth The width of the image
     * @param pixelHeight The height of the image
     * @param standardDeviation The standard deviation of the Gaussian distribution that comprises the filter kernel
     * @return vector<unsigned char> byte vector representing the Gaussian blurred image
     */
    std::vector<unsigned char> generateGaussianBlurredImage(const std::vector<unsigned char> &originalImage, unsigned int pixelWidth, unsigned int pixelHeight, float standardDeviation) {
        // Generate Gaussian kernel
        // TODO: Tidy mallocs & cudaDeviceSynchronizes and repeated 5s etc
        // TODO: Try different sizes of Gaussian kernel
        float *gaussianKernelElements = (float *)malloc(5 * 5 * sizeof(float));
        Matrix2D<float> gaussianKernel(gaussianKernelElements, 5, 5);
        Matrix2D<float> dGaussianKernel(nullptr, gaussianKernel.width, gaussianKernel.height);
        size_t gaussianKernelSizeInBytes = gaussianKernel.width * gaussianKernel.height * sizeof(float);

        checkCudaCall(cudaMalloc(&dGaussianKernel.elements, gaussianKernelSizeInBytes));

        // TODO: different sizes of Gaussian kernel might require more complex summing to normalize ()
        dim3 filterKernelBlockDim(dGaussianKernel.width, dGaussianKernel.height);
        dim3 filterKernelGridDim(1, 1);
        generateGaussianFilterKernel<<<filterKernelGridDim, filterKernelBlockDim>>>(dGaussianKernel, standardDeviation);

        // Apply filter
        unsigned char *dOriginalImage, *dResultImage;
        size_t pixelCount = pixelWidth * pixelHeight;
        size_t imageSizeInBytes = pixelCount * 4 * sizeof(unsigned char); // image dimensions * 4 * size of a byte

        checkCudaCall(cudaMalloc(&dOriginalImage, imageSizeInBytes));
        checkCudaCall(cudaMalloc(&dResultImage, imageSizeInBytes));

        // cudaDeviceSynchronize for kernel generation not needed because cudaMemcpy implicitly calls cudaDeviceSynchronize
        checkCudaCall(cudaMemcpy(dOriginalImage, originalImage.data(), imageSizeInBytes, cudaMemcpyHostToDevice));

        // TODO: set thread params correctly
        dim3 convolutionBlockDim(32, 32);
        dim3 convolutionGridDim((pixelWidth / convolutionBlockDim.x) + 1, (pixelHeight / convolutionBlockDim.y) + 1);

        generateGaussianBlurredImageKernel<<<convolutionGridDim, convolutionBlockDim>>>(dResultImage, dOriginalImage, pixelWidth, pixelHeight, dGaussianKernel);
        auto resultImage = (unsigned char *) malloc(imageSizeInBytes);

        checkCudaCall(cudaMemcpy(resultImage, dResultImage, imageSizeInBytes, cudaMemcpyDeviceToHost));

        checkCudaCall(cudaFree(dGaussianKernel.elements));
        checkCudaCall(cudaFree(dOriginalImage));
        checkCudaCall(cudaFree(dResultImage));

        std::vector<unsigned char> result(resultImage, resultImage + (pixelCount * 4));

        // Freeing resultImage is fine as initializing result as std::vector copies all elements from original
        free(resultImage);
        free(gaussianKernelElements);

        return result;
    }

}
