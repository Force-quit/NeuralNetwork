cmake_minimum_required(VERSION 3.8)
project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TRAIN_SOURCES
    src/trainBPN.cpp
    src/NeuralNetwork.cpp
    src/NeuralNetworkTrainer.cpp
    src/DataReader.cpp
    src/Matrix.cpp
    src/ActivationFunctions.cpp
    src/StopWatcher.h
    src/StopWatcher.cpp
)

add_executable(trainBPN ${TRAIN_SOURCES})

file(COPY resources/config.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY resources/mnist-ubyte DESTINATION ${CMAKE_CURRENT_BINARY_DIR})