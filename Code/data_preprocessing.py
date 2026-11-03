# Alias module — re-exports everything from 01_data_preprocessing.py
# Needed because 02_model_training.py imports 'data_preprocessing'
import importlib, sys

_mod = importlib.import_module('01_data_preprocessing'.replace('-', '_'))
sys.modules[__name__].__dict__.update(
    {k: v for k, v in _mod.__dict__.items() if not k.startswith('__')}
)

# Direct import fallback
from importlib.util import spec_from_file_location, module_from_spec
import os as _os

_path = _os.path.join(_os.path.dirname(__file__), '01_data_preprocessing.py')
_spec = spec_from_file_location('data_preprocessing_impl', _path)
_impl = module_from_spec(_spec)
_spec.loader.exec_module(_impl)

preprocess = _impl.preprocess
load_data  = _impl.load_data
