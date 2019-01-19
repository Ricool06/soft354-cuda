#include <gtest/gtest.h>
#include "Matrix2D.h"
#include "Matrix2D.h"
#include "Canny.h"
#include "Canny.h"

TEST(GrayscaleFilter, SinglePixelImage) {
    unsigned int width = 1, height = 1;

    // Pixel with RGB mean of 100, at 20% opacity...
    std::vector<unsigned char> pixel = {50, 100, 150, 51};
    // Therefore, resulting grayscale intensity should be 20:
    auto expectedIntensityElements = (float*) malloc(sizeof(float));
    expectedIntensityElements[0] = 20.0;
    Matrix2D<float> expectedIntensityMatrix = Matrix2D(expectedIntensityElements, 1, 1);

    Matrix2D<float> actualIntensityMatrix = Canny::generateGrayscaleImage(pixel, width, height);

    EXPECT_EQ(actualIntensityMatrix.width, expectedIntensityMatrix.width);
    EXPECT_EQ(actualIntensityMatrix.height, expectedIntensityMatrix.height);
    EXPECT_FLOAT_EQ(actualIntensityMatrix.elements[0], expectedIntensityMatrix.elements[0]);

    free(expectedIntensityElements);
}
TEST(GrayscaleFilter, MultiPixelImage) {
    unsigned int width = 2, height = 2;

    // 2x2 image of pixels with RGB mean of 100, at 20%, 40%, 60%, 80% opacity...
    std::vector<unsigned char> pixels = {50, 100, 150, 51,
                                         40, 100, 160, 102,
                                         30, 100, 170, 153,
                                         20, 90, 190, 204};

    // Therefore, resulting grayscale intensity matrix should be 20, 40, 60, 80:
    auto expectedIntensityElements = (float*) malloc(sizeof(float) * 4);
    expectedIntensityElements[0] = 20.0;
    expectedIntensityElements[1] = 40.0;
    expectedIntensityElements[2] = 60.0;
    expectedIntensityElements[3] = 80.0;
    Matrix2D<float> expectedIntensityMatrix = Matrix2D(expectedIntensityElements, 2, 2);

    Matrix2D<float> actualIntensityMatrix = Canny::generateGrayscaleImage(pixels, width, height);

    EXPECT_EQ(actualIntensityMatrix.width, expectedIntensityMatrix.width);
    EXPECT_EQ(actualIntensityMatrix.height, expectedIntensityMatrix.height);
    EXPECT_FLOAT_EQ(actualIntensityMatrix.elements[0], expectedIntensityMatrix.elements[0]);
    EXPECT_FLOAT_EQ(actualIntensityMatrix.elements[1], expectedIntensityMatrix.elements[1]);
    EXPECT_FLOAT_EQ(actualIntensityMatrix.elements[2], expectedIntensityMatrix.elements[2]);
    EXPECT_FLOAT_EQ(actualIntensityMatrix.elements[3], expectedIntensityMatrix.elements[3]);
}
