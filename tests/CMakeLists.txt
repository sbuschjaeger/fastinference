cmake_minimum_required(VERSION 3.0)
project(TestCode) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-unused-variable -Ofast")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")  
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ")

if (NOT CMAKE_CROSSCOMPILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
endif()

# This is the main CMakeList.txt we will use for all compiling. For cross-compiling we have different toolchain files
# available which can be included if necessary. Otherweise we use the MODELNAME variable to specifiy 
# the appropriate model name from the fastinference code generation. For example, a RidgeClassifier is compiled with
#           cmake . -DMODEL_NAME=RidgeClassifier
# 
# Since treelite uses a slightly different function signature (e.g. it works on Arrays of entrys not arrays of floats)
# we also supply the option to compile for treelite. This can get messy in the future if there are more systems to support
# but so far having two different options seems okay
# SET(MODELNAME "" CACHE STRING "Name of the model / classifier. Usually found in the corresponding JSON file.")

SET(FEATURE_TYPE "" CACHE STRING "Data type of input features, e.g. int or double")
SET(LABEL_TYPE "" CACHE STRING "Data type of labels, e.g. int or double")


# if (CMAKE_CROSSCOMPILING)
#     # For some reason cmake keeps adding the "-rdynamic" options, so we forcefully remove it here
#     set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "--specs=nosys.specs")
#     set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "--specs=nosys.specs")
# endif()

file(GLOB SRC_CPP
    "*.cpp"
)

add_executable(testCode main.cpp ${SRC_CPP})

target_compile_definitions(testCode PRIVATE -Dpredict=predict_${MODELNAME} -DFEATURE_TYPE=${FEATURE_TYPE} -DLABEL_TYPE=${LABEL_TYPE})
