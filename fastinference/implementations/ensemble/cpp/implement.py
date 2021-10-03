import os
import numpy as np

from jinja2 import Environment, FileSystemLoader

from fastinference.Util import dynamic_import

def to_implementation(model, out_path, out_name, weight = 1.0, namespace = "FAST_INFERENCE", feature_type = "double", label_type = "double", **kwargs ):
    """Generates a C++ implementation of the given Ensemble model. This implementation simply calls the respective implementations of the base learners. You can use this implementation by simply passing :code:`"cpp"` to implement. To choose the implementation of the base-learners pass an additional option to implement:

    .. code-block:: python

        loaded_model = fastinference.Loader.model_from_file("/my/nice/model.json")
        loaded_model.implement("/some/nice/place/", "mymodel", "cpp", "implementation.of.base.learners")

    Args:
        model (Ensemble): The Ensemble model to be implemented
        out_path (str): The folder in which the :code:`*.cpp` and :code:`*.h` files are stored.
        out_name (str): The filenames.
        weight (float, optional): The weight of this model inside an ensemble. The weight is ignored if it is 1.0, otherwise the prediction is scaled by the respective weight.. Defaults to 1.0.
        namespace (str, optional): The namespace under which this model will be generated. Defaults to "FAST_INFERENCE".
        feature_type (str, optional): The data types of the input features. Defaults to "double".
        label_type (str, optional): The data types of the label. Defaults to "double".
    """

    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )

    implementation = env.get_template('base.j2').render(
        model = model,
        feature_type = feature_type,
        namespace = namespace,
        label_type = label_type,
        code_static = "",
        weight = weight
    )

    header = env.get_template('header.j2').render(
        model = model,
        feature_type = feature_type,
        namespace = namespace,
        label_type = label_type,
        weight = weight
    )

    with open(os.path.join(out_path, "{}.{}".format(out_name,"cpp") ), 'w') as out_file:
        out_file.write(implementation)

    with open(os.path.join(out_path, "{}.{}".format(out_name,"h")), 'w') as out_file:
        out_file.write(header)