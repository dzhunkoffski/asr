import logging
from pathlib import Path
import json

import torchaudio
from datasets import load_dataset
import re
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)

class JSONDataset(BaseDataset):
    def __init__(self, index_path, *args, **kwargs):
        with open(index_path, 'r') as fd:
            index = json.load(fd)
        super().__init__(index, *args, **kwargs)