from pathlib import Path

import tensorflow as tf

from deepClassifier.config.configuration import EvaluationConfig
from deepClassifier.utils import save_json

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