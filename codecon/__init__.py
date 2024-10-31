# codecon/__init__.py
# codecon/__init__.py

try:
    import pandas
    import numpy
    import sklearn
    import transformers
    import torch
except ImportError as e:
    missing_package = str(e).split()[-1]
    raise ImportError(
        f"Missing required library: {missing_package}. "
        "Please run 'pip install -r requirements.txt' or 'pip install codecon' to install it."
    )

from codecon.cl_nlp_train import cl_nlp_train
from codecon.cl_nlp_pred import cl_nlp_pred
from codecon.cl_nlp_findtrain import cl_nlp_findtrain
from codecon.tp_nlp import tp_nlp
from codecon.gai_nlp import gai_nlp