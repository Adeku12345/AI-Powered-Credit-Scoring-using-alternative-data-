import sys

from backend.core.exceptions import CustomException
from backend.core.logger import logging

from backend.ml_core.data_ingestion import DataIngestion
from backend.ml_core.data_transformation import DataTransformation
from backend.ml_core.train_pipeline import train_pipeline


def run_training_pipeline():
    try:
        logging.info("===== Training pipeline started =====")

        # 1. Ingestion: read source CSV, split into train/test, save to artifacts/
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # 2. Transformation: impute/scale/encode features, save preprocessor.pkl
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        # 3. Training: fit + tune candidate models, save the best one as model.pkl
        Train = ModelTrainer()
        r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"===== Training pipeline completed. Best model R2 score: {r2} =====")
        print(f"Training complete. Best model R2 score on test set: {r2:.4f}")

        return r2

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
