
# NeuralNetwork

Back Propagation Neural Network (BPN) training.

 - This code is a fork from [Xavier Provençal's NeuralNetwork fork from Bobby Anguelov's NeuralNetwork](https://github.com/xprov/NeuralNetwork).

The creation and training of this BPN uses the <a href="http://yann.lecun.com/exdb/mnist/">MNIST</a> database of hand written digits.

## Compilation

### Compile under Gnu/Linux

#### Using `cmake` (recommended)

 - Using Synaptic Manager, install the required compilation tools

	```
	sudo apt update
	sudo apt install g++ make cmake
	```

 -  __Compilation__

	```
	mkdir build
	cd build
	cmake ..
	make
	```

 - __Debug VS Release Mode__

	Since training of neural networks requires a lot of computation, one might find useful to compile a debug and a release versions (the release version should run significantly faster):
	```
	mkdir debug && cd debug && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j
	cd ..
	mkdir release && cd release && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
	```

 - __Run__

	From the `build` folder
	```
	./trainNN -h
	./evalNN -h
	```


### Compile under Windows

See [Visual Studio's guide for CMake projects](https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio)

# Visualization and testing of a BPN

Go on [M. Provençal's website](https://xprov.org/gui/fcnn.html)
