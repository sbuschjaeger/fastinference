import os
import numpy as np

from jinja2 import Environment, FileSystemLoader

from fastinference.Util import dynamic_import

def to_implementation(
    model, 
    out_path, 
    out_name, 
    weight = 1.0, 
    base_implementation = "",
    feature_type = "double",
    namespace = "FAST_INFERENCE",
    label_type = "double",
    internal_type = "double",
    **kwargs
):

    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )

    implementation = env.get_template('base.j2').render(
        model = model,
        feature_type = feature_type,
        namespace = namespace,
        label_type = label_type,
        internal_type = internal_type,
        code_static = "",
        weight = weight
    )

    header = env.get_template('header.j2').render(
        model = model,
        feature_type = feature_type,
        namespace = namespace,
        label_type = label_type,
        internal_type = internal_type,
        weight = weight
    )

    with open(os.path.join(out_path, "{}.{}".format(out_name,"cpp") ), 'w') as out_file:
        out_file.write(implementation)

    with open(os.path.join(out_path, "{}.{}".format(out_name,"h")), 'w') as out_file:
        out_file.write(header)