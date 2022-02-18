from setuptools import find_packages, setup

setup(name='fastinference',
      version='0.1',
      description='Fastinference is a machine learning model compiler specifically targeted to small, embedded systems  and FPGAs.',
      long_description='Fastinference is a machine learning model compiler specifically targeted to small, embedded systems and FPGAs.',
      url='https://github.com/sbuschjaeger/fastinference',
      author=u'Sebastian Buschjäger',
      author_email='{sebastian.buschjaeger}@tu-dortmund.de',
      license='MIT',
      #packages=['fastinference'],
      #scripts=["fastinference/fastinference.py"],
      zip_safe=False,
      packages=find_packages(),#include=['fastinference', 'fastinference.models', 'fastinference.models.nn'])
      include_package_data = True,
      package_data = {'': ['*.j2'],},
      install_requires = [
      "jinja2",
      "onnx",
      "onnxruntime",
      "xgboost",
      "PyPruning @ git+https://git@github.com/sbuschjaeger/PyPruning.git"
    ]
)