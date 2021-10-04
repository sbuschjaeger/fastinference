.. Fastinference documentation master file, created by
   sphinx-quickstart on Wed Sep 22 11:09:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Fastinference
=============

.. |ls8| image:: ls8-logo-shadow.svg
  :width: 25

.. |sfb| image:: sfb-logo.svg
  :width: 20

.. toctree::
   :maxdepth: 3
   :hidden:
   
   self
   models
   implementations
   extending

Fastinference is a machine learning model optimizer and model compiler that generates the optimal implementation for your model and hardware architecture:

**In Fastinference the user comes first.** We believe that the user know best what implementation and what type of optimizations should be performed. Hence, we generate *readable* code so that the user can adapt and change the implementation if necessary. 

**In Fastinference optimizations and implementations can be freely combined.** Fastinference distinguishes between optimizations for specific models which are independent from the implementation and specific types of implementations. Consider for example a simple decision tree, then the pruning of the model does not affect its implementation and vice-versa. 

**Fastinference can be easily extended.** You can easily add your own implementation while benefiting from all optimizations performed on the model and vice-versa. 


How to install
==============

You can install this package via pip from git 

.. code-block::

   pip install git+https://github.com/sbuschjaeger/fastinference.git

If you have trouble with dependencies you can try setting up a conda environment which I use for development:

.. code-block::

   git clone git@github.com:sbuschjaeger/fastinference.git
   cd fastinference
   conda env create -f environment.yml
   conda activate fi

Please note that this environment also contains some larger packages such as PyTorch so the installation may take some time. 

How to use fastinference
========================

Using fastinference from the command linear
-------------------------------------------

If you have stored your model on disk (e.g. as an :code:`json` file) then you can generate the code directly from the CLI via:

.. code-block:: bash

   python3 fastinference/main.py --model /my/nice/model.json --feature_type float --out_path /my/nice/model --out_name "model" --implementation my.newest.implementation --optimize my.newest.optimization

This call will load the model stored in :code:`/my/nice/model.json`, performs the optimizations implemented in :code:`my.newest.optimization` and then finally generates the implementation according to :code:`my.newest.implementation` where the data type of features is :code:`float`. Any additional arguments passed to :code:`main.py` will be passed to the :code:`my.newest.optimization` and :code:`my.newest.implementation` respectively so you can just pass anything you require. Note that for ensembles you can additionally pass :code:`baseimplementation` and :code:`baseoptimize` to specify optimizations on the base learners as well as their respective implementations.

For Linear, Discriminant, Tree, Ensemble models we currently support :code:`.json` files which have previously been written via :code:`Loader.model_to_json`. For Neural Networks we use :code:`onnx` files which e.g. have been written via :code:`torch.onnx.export` or :code:`tf2onnx`. Reading onnx files can be tricky sometimes so please check out Neural Network for caveats. 

Using fastinference in your python program
------------------------------------------

Simply import :code:`fastinference.Loader`, load your model and you are ready to go:

.. code-block:: python

    import fastinference.Loader

    loaded_model = fastinference.Loader.model_from_file("/my/nice/model.json")
    loaded_model.optimize("my.newest.optimization", None)
    loaded_model.implement("/my/nice/model", "model", "my.newest.implementation")

Again for ensembles you can pass additional :code:`base_optimizers` and :code:`base_args` arguments to the call of :code:`optimize` for the optimization of base learners in the ensemble. For :code:`scikit-learn` models you can also :code:`Loader.model_from_sklearn` to load the model. For Deep Learning approaches you will always have to store the model as an ONNX file first. 

A complete example
==================

A complete example which trains a Random Forest on artificial data, performs some optimizations on the trees and finally generates some c++ code would look like the following: 

.. code-block:: bash

  # Define some constants
  OUTPATH="/tmp/fastinference"
  MODELNAME="RandomForestClassifier"
  FEATURE_TYPE="int"

  # Generate some artificial data with 5 classes, 20 features and 10000 data points. 
  python3 tests/data/generate_data.py --out $OUTPATH --nclasses 5 --nfeatures 20 --difficulty 0.5 --nexamples 10000

  # Train a RF with 25 trees on the generated data
  python3 tests/train_$TYPE.py --training $OUTPATH/training.csv --testing $OUTPATH/testing.csv --out $OUTPATH --name $MODELNAME  --nestimators 25 

  # Perform the actual optimization + code generation
  python3 fastinference/main.py --model $OUTPATH/$MODELNAME.json --feature_type $FEATURE_TYPE --out_path $OUTPATH --out_name "model" --implementation cpp --baseimplementation cpp.ifelse --baseoptimize swap 

  # Prepare the C++ files for compilation
  python3 ./tests/data/convert_data.py --file $OUTPATH/testing.csv --out $OUTPATH/testing.h --dtype $FEATURE_TYPE --ltype "unsigned int"
  cp ./tests/main.cpp $OUTPATH
  cp ./tests/CMakeLists.txt $OUTPATH
  cd $OUTPATH

  # Compile the code
  cmake . -DMODELNAME=$MODELNAME -DFEATURE_TYPE=$FEATURE_TYPE
  make

  # Run the code
  ./testCode

There is a CI/CD pipeline running which tests the current code and uses :code:`tests/run_tests.sh` to orchestrate the various scripts. In doubt please have a look at these files.

Acknowledgements
================
The software is written and maintained by `Sebastian Buschjäger <https://sbuschjaeger.github.io/>`_ as part of his work at the `Chair for Artificial Intelligence <https://www-ai.cs.tu-dortmund.de>`_ |ls8| at the TU Dortmund University and the `Collaborative Research Center 876 <https://sfb876.tu-dortmund.de>`_ |sfb|. If you have any question feel free to contact me under sebastian.buschjaeger@tu-dortmund.de.

Special thanks goes to `Maik Schmidt <maik.schmidt@tu-dortmund.de>`_ and `Andreas Buehner <andreas.buehner@tu-dortmund.de>`_ who provided parts of this implementation during their time at the TU Dortmund University. 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
