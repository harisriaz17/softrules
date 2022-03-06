import yaml
import sys

from src.argparser import get_softrules_argparser
from typing import Union, List

class Config:
    def __init__(self, config):
        self.config = config

    """
    The reason behind accessing the config dictionary through this method is to discourage
     accessing it directly wherever needed in code. This is done because accessing it like:
     >>> config_object.config.get("param_name", default_value)
    can hide errors. "param_name" might not be in the config (or might be mispelled)
    """
    def get(self, param):
        if param in self.config:
            return self.config[param]
        else:
            error_str = f"The parameter {param} is not in the config, which contains the following keys: {list(self.config.keys())}.\nThe full config is: {self.config}"
            raise ValueError(error_str)

    """

    """
    @staticmethod
    def get_config(paths: Union[str, List[str]] = "config/default_config.yaml"):
        if isinstance(paths, List):
            config = {}
            for path in paths:
                with open(path) as f:
                    new_config = yaml.load(f, Loader=yaml.FullLoader)
                    config.update(new_config)
        else:
            with open(paths) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

        return Config(config)

    @staticmethod
    def parse_args_and_get_config(args):
        parser = get_softrules_argparser()
        args   = parser.parse_args(args)
        args   = vars(args)

        config = {}
        for path in args['path']:
            with open(path) as f:
                new_config = yaml.load(f, Loader=yaml.FullLoader)
                config.update(new_config)
        
        config.update(args)

        return Config(config)


# python -m src.config
if __name__ == "__main__":
    c = Config.parse_args_and_get_config(sys.argv[1:])
    print(c.config)