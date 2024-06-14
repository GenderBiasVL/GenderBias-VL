
from .vqa_dataset import *

from .occ_dataset import *

dataset_dict = {
    # VQA 
    'ScienceQA': ScienceQADataset,
    ### img bias
    'OccBaseAskPerson': OccDataset,
    'OccCfAskPerson': OccDataset,
    'OccBaseAskPersonSwapOption': OccDataset,
    'OccCfAskPersonSwapOption': OccDataset,

    ### img and text bias 
    'OccBaseAskGender': OccDataset,
    'OccCfAskGender': OccDataset,
    'OccBaseAskGenderSwapOption': OccDataset,
    'OccCfAskGenderSwapOption': OccDataset,

    ### text bias
    'OccTextBaseAskGender': OccDataset,
    'OccTextCfAskGender': OccDataset,
    'OccTextBaseAskGenderSwapOption': OccDataset,
    'OccTextCfAskGenderSwapOption': OccDataset,
}
