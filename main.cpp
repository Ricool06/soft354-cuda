#include <iostream>
#include "Canny.h"
#include <lodepng.h>

int main() {
    std::cout << "Hello, World!" << std::endl;

    // Load images
    std::vector<unsigned char> flowers, tiger;
    unsigned tigerWidth, tigerHeight, flowersWidth, flowersHeight;
    unsigned error = lodepng::decode(flowers, flowersWidth, flowersHeight, "img/flowers.png");
    if(error) std::cout << "decoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    error = lodepng::decode(tiger, tigerWidth, tigerHeight, "img/tiger.png");
    if(error) std::cout << "decoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    std::vector<unsigned char> greyscaleTiger = Canny::generateGreyscaleImage(tiger, tigerWidth, tigerHeight);
    error = lodepng::encode("img/gs_tiger.png", greyscaleTiger, tigerWidth, tigerHeight);
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    std::vector<unsigned char> gaussianBlurredGreyscaleTiger = Canny::generateGaussianBlurredImage(greyscaleTiger, tigerWidth, tigerHeight, 4.0);
    error = lodepng::encode("img/gb_gs_tiger.png", gaussianBlurredGreyscaleTiger, tigerWidth, tigerHeight);
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    std::vector<unsigned char> gaussianBlurredTiger = Canny::generateGaussianBlurredImage(tiger, tigerWidth, tigerHeight, 4.0);
    error = lodepng::encode("img/gb_tiger.png", gaussianBlurredTiger, tigerWidth, tigerHeight);
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    std::vector<unsigned char> gaussianBlurredFlowers = Canny::generateGaussianBlurredImage(flowers, flowersWidth, flowersHeight, 4.0);
    error = lodepng::encode("img/gb_flowers.png", gaussianBlurredFlowers, flowersWidth, flowersHeight);
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    return 0;
}