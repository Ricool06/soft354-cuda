#include <iostream>
#include "Canny.h"
#include <lodepng.h>

int main() {
    std::cout << "Hello, World!" << std::endl;

    std::vector<unsigned char> image;
    unsigned width, height;
    unsigned error = lodepng::decode(image, width, height, "img/tiger.png");
    if(error) std::cout << "decoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    std::vector<unsigned char> output = Canny::generateGrayscaleImage(image, width, height);

    error = lodepng::encode("img/gs_tiger.png", output, width, height);
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    return 0;
}