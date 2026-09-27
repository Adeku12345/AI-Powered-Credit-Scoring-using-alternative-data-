import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegression
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from backend.core.exceptions import CustomException
from backend.core.logger import logging

from backend.core.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array, preprocessor_path)
        try:
            logging.info("split training and test input data")
            x_train,y_train,x_test, y_test=(
                train_array[:,:-1],
                tra
            )
            
        
        
        except:
            pass    
            