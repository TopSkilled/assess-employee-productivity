# Alias module — re-exports run_training from 02_model_training.py
# Needed because 03_model_evaluation.py imports 'model_training'
from importlib.util import spec_from_file_location, module_from_spec
import os as _os

_path = _os.path.join(_os.path.dirname(__file__), '02_model_training.py')
_spec = spec_from_file_location('model_training_impl', _path)
_impl = module_from_spec(_spec)
_spec.loader.exec_module(_impl)

run_training = _impl.run_training
