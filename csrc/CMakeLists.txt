set(
    tensor_libs_file
    value.cc
)

add_library(${DEEPTENSOR_LIBS} STATIC ${tensor_libs_file})
