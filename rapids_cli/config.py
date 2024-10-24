import pathlib
import yaml


_ROOT_DIR = pathlib.Path(__file__).parent


with open(_ROOT_DIR / "config.yml", "r") as file:
    config = yaml.safe_load(file)
