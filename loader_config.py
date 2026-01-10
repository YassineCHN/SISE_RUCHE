import yaml
import os


def load_config(env="dev"):
    with open(f"config/{env}.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


ENV = os.getenv("ENV", "dev")
config = load_config(ENV)
