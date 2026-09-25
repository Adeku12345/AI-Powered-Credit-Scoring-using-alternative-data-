import os
import sys
from backend.core.exceptions import CustomException
from backend.core.logger import logging  
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifact',"data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config    