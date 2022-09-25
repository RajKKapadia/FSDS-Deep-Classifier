from deepClassifier.config import ConfigurationManager
from deepClassifier.components import Evaluation
from deepClassifier import logger

STAGE_NAME = "Evaluation of the model"

def main():
    config = ConfigurationManager()
    evaluation_config = config.get_evaluation_config()
    evaluation = Evaluation(evaluation_config)
    evaluation.load_model()
    evaluation.evaluate()
    

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e