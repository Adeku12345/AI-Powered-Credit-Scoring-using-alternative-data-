import sys

from backend.core.exceptions import CustomException
from backend.core.logger import logging

from backend.ml_core.data_ingestion import DataIngestion
from backend.ml_core.data_transformation import DataTransformation
from backend.ml_core.model_trainer import ModelTrainer


def run_training_pipeline():
    try:
        logging.info("===== Training pipeline started =====")

        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        model_trainer = ModelTrainer()
        r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"===== Training pipeline completed. Best model R2 score: {r2} =====")
        print(f"Training complete. Best model R2 score on test set: {r2:.4f}")

        return r2

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()