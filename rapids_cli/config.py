import pathlib
import yaml


_CURRENT_DIR = pathlib.Path(__file__).parent


with open(_CURRENT_DIR / 'config.yml', 'r') as file: 
    config = yaml.safe_load(file)
