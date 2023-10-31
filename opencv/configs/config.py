import yaml
from pydantic import BaseModel
import os

try:
    from .. import lr
except ImportError:
    from opencv import lr

logger = lr.setup_logger()


class Config(BaseModel):
    name: str
    cache_folder: str
    debug_mode: bool

    def __init__(self, config_name: str = 'default.yaml'):
        config_path = os.path.join(os.path.dirname(__file__), config_name)

        logger.info(f"Loading {config_name} config")
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        config_data['cache_folder'] = os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ),
            config_data['cache_folder'])
        super().__init__(**config_data)
        self.print_self()

    def print_self(self):
        for attribute, value in self.model_dump().items():
            logger.info(f"Config -> Attribute: {attribute}, Value: {value}")


if __name__ == '__main__':
    # Instantiate the Config class by passing the config file name
    config = Config('default.yaml')

    # Access the class attributes
    print(config.name)  # MyApplication
    print(config.cache_folder)  # my_api_key
    print(config.debug_mode)  # True
