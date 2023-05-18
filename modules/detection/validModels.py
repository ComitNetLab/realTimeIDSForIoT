from enum import Enum
from NB15ReconModel import recon_model
from NB15FuzzerModel import fuzzer_model
from typing import Callable

'''
Enum class for valid models
add to the list to verify the model name, and remember to name the model file
like the model name in the enum class
'''
class ValidModels(Enum):
    NB15_RECON_MODEL = 'NB15-recon-model'
    NB15_FUZZER_MODEL = 'NB15-fuzzer-model'

'''
Map the model name to the related module function
The functions should receive the payload and the model as parameters
'''
def get_model_function(model) -> Callable[[str, any], None]:
    if model == ValidModels.NB15_RECON_MODEL.value:
        return recon_model
    if model == ValidModels.NB15_FUZZER_MODEL.value:
        return fuzzer_model
    else:
        raise ValueError('Invalid model name, check valid models in ./modules/detection/validModels.py')
    
