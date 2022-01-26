.. _extending-label:

Extending fastinference
=======================

One of the main design goals of Fastinference is to allow to easily add new types of optimizations and implementations while benefiting from existing optimizations and implementations out of the box. The central object in Fastinference is the model.  

Adding a new type of implementation
-----------------------------------

Adding a new implementation for a given object is easy. Simply provide a :code:`implement.py` file which contains a function :code:`to_implementation` which receives 

- model: The model to be implemented. This is a deepcopy of the original model's object so you can perform changes on this object if required. 
- out_path: The folder in which the source code for this model should be stored
- out_name: The filename under which the models implementation should be stored
- weight: The weight of this model in case it is part of an ensemble. The prediction should be scaled by this weight.

.. code-block:: python

    def to_implementation(model, out_path, out_name, weight = 1.0, **kwargs):
        # Generate the new implementation here

Fastinference will search for existing implementations under :code:`implementations/my/new/implementation/imlement.py` which can then be loaded via :code:`--implementation my.new.implementation`. Per convention we currently store implementations under :code:`implementations/{model}/{language}/{implementation}`. You can pass any additional argument using :code:`kwargs` and fastinference will try to lazily pass any command-line arguments to you function. Don't forget to document your implementation. Just adapt :code:`docs/implementations.rst` to include your new implementation and the docstring of your :code:`to_implementation` will be include in the docs.

**A note for ensembles**: For the :code:`cpp` implementations we currently assume the following signature. Here, the predictions should be **added** into the :code:`pred` array and not copied, because the implementation of the ensemble will call each base-learners implementation on the **same** array.  

.. code-block:: c++

    void predict_{{model.name}}({{ feature_type }} const * const x, {{ label_type }} * pred) {
        // the actual code
    }

**Important:** Currently all implementations utilize the template engine jinja (https://jinja.palletsprojects.com/en/3.0.x/), but there is no requirement to use jinja for new types of implementations. We originally intended to provide all implementations via jinja (e.g. also for other languages), but although jinja is very powerful it would sometimes be very difficult to provide certain types of implementations. Hence, we decided to simply use python code to generate the necessary implementations without any formal depenence on jinja. Nevertheless, we recommend to use jinja whenever possible. For any C-type language (e.g. C, Java etc.) we recommend to simply copy the entire implementation folder of each model and then to adapt the jinja templates wherever necessary. 

Adding a new type of optimization
---------------------------------

Adding a new optimization for a given object is easy. Simply provid a function :code:`optimize` which receives   the model to be optimized and returns the optimized model: 

.. code-block:: python

    def optimize(model, **kwargs):
        # Perform some optimizations on model the new implementation here

        return model

Fastinference will search for existing optimizations under :code:`optimizations/my/new/optimization.py` which can then be loaded via :code:`--optimize my.new.optimization`. Per convention we currently store optimizations under :code:`{optimizers}/{model}/`. You can pass any additional argument using :code:`kwargs` and fastinference will try to lazily pass any command-line arguments to you function. Don't forget to document your implementation. Just adapt :code:`docs/{model}.rst` to include your new optimization and the docstring of your :code:`optimize` will be include in the docs.


Adding a new type of model
--------------------------

Adding a new model to fastinference is slightly more work. First, you need to implement `fastinference.models.Model`. To do so, you will have to implement the :code:`predict_proba` method which executes the given model on a batch of data and the :code:`to_dict` method which return a dictionary representation of the model. Last, you also might need to supply a new model category such as :code:`{linear, tree, ensemble, discriminant, neuralnet}`:

.. code-block:: python

    class MyModel(Model):
        def __init__(self, classes, n_features, category, accuracy = None, name = "Model"):
            super().__init__(classes, n_features, "A-new-category", accuracy, name)
            pass
        
        def predict_proba(self,X):
            pass
            
        def to_dict(self):
            model_dict = super().to_dict()
            # Add some stuff to model_dict
            return model_dict


Once the model is implemented you need to provide methods for loading and storing. The main entry points for loading and storing in fastinference 

- :code:`Loader.model_from_file` for loading a new model from a file
- :code:`Loader.model_to_json` for storing a new model into a JSON file

In order to load the model you will have to adapt :code:`Loader.model_from_file`. If your model does not really fit into a JSON format or comes with its own format (e.g. as for neural networks and the ONNX format) then you can ignore :code:`Loader.model_to_json`. However, we try to keep these loading / storing functions as consistent as possible so try to provide both if possible.


Testing your implementation / optimization
------------------------------------------

Training a model, generating the code and finally compiling it can be a cumbersome endeavor if you want to debug / test your implementation. We offer some scripts which help during development

- :code:`environment.yml`: A anaconda environment file which we use during development.
- :code:`tests/generate_data.py`: A script to generate some random test and training data.
- :code:`tests/train_{linear,discriminant,tree,mlp,cnn}.py`: A script to train the respective classifier or an ensemble of those. 
- :code:`tests/convert_data.py`: A script to convert the test data into a static header file for the c++ implementations.
- :code:`tests/main.cpp`: The main.cpp file when testing c++ implementations.
- :code:`tests/CMakeLists.txt`: The CMakeLists when testing c++ implementations.

A complete example of the entire workflow can be found in `run_tests.sh <https://github.com/sbuschjaeger/fastinference/blob/main/tests/run_test.sh>`_ and we try to maintain a CI/CD pipeline under `tests.yml <https://github.com/sbuschjaeger/fastinference/blob/main/.github/workflows/tests.yml>`_. Please check this file for the latest test configurations.