import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")
    log_file_path: str = os.path.join(os.getcwd(), 'logs', f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Entered the data ingestion method or component")

            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Saved raw data to CSV')

            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info('Saved training data to CSV')

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Saved test data to CSV')

            with open(self.ingestion_config.log_file_path, 'a') as file:
                timestamp = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
                file.write(f"[{timestamp}] Ingestion of the data is completed\n")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
    


