set(
    tensor_libs_file
    value.cc
    jsoncpp.cc
)

add_library(${DEEPTENSOR_LIBS} STATIC ${tensor_libs_file})
