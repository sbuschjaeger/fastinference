#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <assert.h>

#include "model.h"

namespace FAST_INFERENCE {}
using namespace FAST_INFERENCE;

auto read_csv(std::string &path) {
	std::vector<std::vector<FEATURE_TYPE>> X;
	std::vector<unsigned int> Y;

	std::ifstream file(path);
	// if (!file_exists(path)) {
	// 	throw std::runtime_error("File not found " + path);
	// }
	std::string header;
	std::getline(file, header);

	unsigned int label_pos = 0;
	std::stringstream ss(header);
	std::string entry;
	while (std::getline(ss, entry, ',')) {
		if (entry == "label") {
			break;
		} else {
			label_pos++;
		}
	}
	std::cout << "label_pos: " << label_pos << std::endl;

	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {
			if (line.size() > 0) {
				std::stringstream ss(line);
				entry = "";

				unsigned int i = 0;
				std::vector<FEATURE_TYPE> x;
				while (std::getline(ss, entry, ',')) {
					if (i == label_pos) {
						Y.push_back(static_cast<unsigned int>(std::stoi(entry)));
					} else {
						x.push_back(static_cast<FEATURE_TYPE>(std::stof(entry)));
					}
					++i;
				}
				X.push_back(x);
			}
		}
		file.close();
	}
	return std::make_tuple(X,Y);
}

auto benchmark(std::vector<std::vector<FEATURE_TYPE>> &X, std::vector<unsigned int> &Y, unsigned int n_classes, unsigned int repeat) {
    //double output[n_classes] = {0};
	double * output = new double[n_classes];
    unsigned int n_features = X[0].size();

	unsigned int matches = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int k = 0; k < repeat; ++k) {
    	matches = 0;
	    for (unsigned int i = 0; i < X.size(); ++i) {
	        std::fill(output, output+n_classes, 0);
	        unsigned int label = Y[i];

	        // Note: To make this code more universially applicable we define predict to be the correct function
	        //       which is given in the command line argument. For example, a RidgeClassifier is compiled with
	        //          cmake . -DMODEL_NAME=RidgeClassifier
	        //       where in the cmake file we define
	        //          SET(MODELNAME "" CACHE STRING "Name of the model / classifier. Usually found in the corresponding JSON file.")
	        //          target_compile_definitions(testCode PRIVATE -Dpredict=predict_${MODELNAME})
			//int const * const x = &X[i*NUM_FEATURES];
			FEATURE_TYPE const * const x = &X[i][0];
			predict(x, output);

	        double max = output[0];
	        unsigned int argmax = 0;
	        for (unsigned int j = 1; j < n_classes; j++) {
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
    auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / (X.size() * repeat);
    float accuracy = static_cast<float>(matches) / X.size() * 100.f;
    return std::make_pair(accuracy, runtime);
}

int main (int argc, char *argv[]) {
	if (argc <= 3) {
		std::cout << "Please provide three arguments: path n_classes n_repetitions" << std::endl;
	}
	std::string path = std::string(argv[1]);
	unsigned int n_classes = std::stoi(argv[2]);
	unsigned int repeat = std::stoi(argv[3]);

	auto data = read_csv(path);

	assert(std::get<0>(data).size() > 0);
	assert(std::get<0>(data).size() == std::get<1>(data).size());

    std::cout << "RUNNING BENCHMARK WITH " << repeat << " REPETITIONS" << std::endl;
    auto results = benchmark(std::get<0>(data), std::get<1>(data), n_classes, repeat);
    
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