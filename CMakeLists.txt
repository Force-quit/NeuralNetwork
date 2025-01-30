cmake_minimum_required(VERSION 3.29)
project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TRAIN_SOURCES
    src/main.cpp
    src/NeuralNetwork.h
    src/NeuralNetwork.cpp
    src/NeuralNetworkTrainer.h
    src/NeuralNetworkTrainer.cpp
    src/DataReader.h
    src/DataReader.cpp
    src/ConfigFileParser.h
    src/Matrix.h
    src/Matrix.cpp
    src/ActivationFunctions.h
    src/ActivationFunctions.cpp
    src/StopWatcher.h
    src/StopWatcher.cpp
    src/vectorstream.h
)

add_executable(trainBPN ${TRAIN_SOURCES})

file(COPY resources/config.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY resources/mnist-ubyte DESTINATION ${CMAKE_CURRENT_BINARY_DIR})