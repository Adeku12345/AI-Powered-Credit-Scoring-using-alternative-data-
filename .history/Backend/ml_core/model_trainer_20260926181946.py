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