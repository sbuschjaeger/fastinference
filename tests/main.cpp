#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>

#include "testing.h"
#include "model.h"

namespace FAST_INFERENCE {}
using namespace FAST_INFERENCE;

auto benchmark(unsigned int repeat = 20) {
    //double output[NUM_CLASSES] = {0};
    double * output = new double[NUM_CLASSES];

	unsigned int matches = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int k = 0; k < repeat; ++k) {
    	matches = 0;
	    for (unsigned int i = 0; i < NUM_EXAMPLES; ++i) {
	        std::fill(output, output+NUM_CLASSES, 0);
	        unsigned int label = Y[i];

	        // Note: To make this code more universially applicable we define predict to be the correct function
	        //       which is given in the command line argument. For example, a RidgeClassifier is compiled with
	        //          cmake . -DMODEL_NAME=RidgeClassifier
	        //       where in the cmake file we define
	        //          SET(MODELNAME "" CACHE STRING "Name of the model / classifier. Usually found in the corresponding JSON file.")
	        //          target_compile_definitions(testCode PRIVATE -Dpredict=predict_${MODELNAME})
			//int const * const x = &X[i*NUM_FEATURES];
			FEATURE_TYPE const * const x = &X[i*NUM_FEATURES];
			predict(x, output);

	        double max = output[0];
	        unsigned int argmax = 0;
	        for (unsigned int j = 1; j < NUM_CLASSES; j++) {
	            if (output[j] > max) {
	                max = output[j];
	                argmax = j;
	            }
	        }

			if (argmax == label) {
				++matches;
			}
	    }
    }

	delete[] output;
    auto end = std::chrono::high_resolution_clock::now();   
    auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / (NUM_EXAMPLES * repeat);
    float accuracy = static_cast<float>(matches) / NUM_EXAMPLES * 100.f;
    return std::make_pair(accuracy, runtime);
}

int main (int argc, char *argv[]) {
	unsigned int repeat = 2;
	if (argc > 1) {
		repeat = std::stoi(argv[1]);
	}

    std::cout << "RUNNING BENCHMARK WITH " << repeat << " REPETITIONS" << std::endl;
    auto results = benchmark(repeat);
    
    std::cout << "Accuracy: " << results.first << " %" << std::endl;
    std::cout << "Latency: " << results.second << " [ms/elem]" << std::endl;
	#ifdef REF_ACCURACY
		float difference = results.first - REF_ACCURACY;
		std::cout << "Reference Accuracy: " << REF_ACCURACY << " %" << std::endl;
		std::cout << "Difference: " << difference << std::endl;
	    
        std::cout << results.first << "," << REF_ACCURACY << "," << difference << "," << results.second << std::endl;
	#else
        std::cout << results.first << "," << "," << "," << results.second << std::endl;
    #endif

    return 0;
}