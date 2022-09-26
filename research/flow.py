import os
from dataclasses import dataclass
from pathlib import Path
import json
from urllib.parse import urlparse

import tensorflow as tf
import mlflow

from deepClassifier.constants import *
from deepClassifier.utils import read_yaml, create_directories, save_json

secret_file_path = os.path.join(
    os.getcwd(),
    'research',
    'secret.json'
)

with open(file=secret_file_path, mode='r') as file:
    secrets = json.load(file)

os.environ['MLFLOW_TRACKING_USERNAME'] = secrets['MLFLOW_TRACKING_USERNAME']
os.environ['MLFLOW_TRACKING_PASSWORD'] = secrets['MLFLOW_TRACKING_PASSWORD']

@dataclass(frozen=True)
class EvaluationConfig:
    trained_mdeol_path: Path
    training_data: Path
    params_batch_size: int
    params_image_size: list
    all_params: dict


class ConfigurationManager:
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_evaluation_config(self) -> EvaluationConfig:
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, 'PetImages')
        evaluation_config = EvaluationConfig(
            trained_mdeol_path=self.config.training.trained_model_path,
            training_data=training_data,
            params_batch_size=self.params.BATCH_SIZE,
            params_image_size=self.params.IMAGE_SIZE,
            all_params=self.params
        )

        return evaluation_config


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def load_model(self):
        self.model = tf.keras.models.load_model(
            self.config.trained_mdeol_path
        )

    def _evaluation_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation='bilinear'
        )

        evaluation_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.evaluation_generator = evaluation_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset='validation',
            shuffle=False,
            **dataflow_kwargs
        )

    def evaluate(self):
        self.load_model()
        self._evaluation_generator()
        self.score = self.model.evaluate(self.evaluation_generator)
        save_json(
            path=Path('score.json'),
            data={
                'loss': self.score[0],
                'accuracy': self.score[1]
            }
        )
    
    def log_into_mlflow(self):
        print('Start of logging...')
        mlflow.set_registry_uri(secrets['MLFLOW_TRACKING_URI'])
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(tracking_url_type_store)
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # if tracking_url_type_store != "file":
            #     mlflow.tensorflow.log_model(tf_saved_model_dir='models', registered_model_name='VGG16Model')
            # else:
            #     mlflow.tensorflow.log_model(tf_saved_model_dir='models', registered_model_name='VGG16Model')
        
        print('Start of logging...')


try:
    config = ConfigurationManager()
    evaluation_config = config.get_evaluation_config()
    evaluation = Evaluation(evaluation_config)
    evaluation.load_model()
    evaluation.evaluate()
    evaluation.log_into_mlflow()
except Exception as e:
    raise e
