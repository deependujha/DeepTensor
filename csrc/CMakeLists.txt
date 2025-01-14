set(
    tensor_libs_file
    value.cc
    tensor.cc
    loss.cc
)

add_library(${DEEPTENSOR_LIBS} STATIC ${tensor_libs_file})
