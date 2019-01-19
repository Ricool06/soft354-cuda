add_library(${LIBRARY} Canny.cu Canny.h Matrix2D.cu Matrix2D.h)
target_link_libraries(${LIBRARY} ${LODE_PNG_LIB})
