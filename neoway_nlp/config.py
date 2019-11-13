from os import path

import neoway_nlp

base_path = path.dirname(path.dirname(neoway_nlp.__file__))
workspace_path = path.join(base_path, 'workspace')
data_path = path.join(workspace_path, 'data')
models_path = path.join(workspace_path, 'models')
