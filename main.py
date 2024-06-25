import argparse
import torch
import yaml
from task_utils.classify_task import Classify


def main(config_path: str) -> None:
    with open(config_path) as file:
        config = yaml.safe_load(file)

    trainer = Classify(config)
    trainer.train()
    trainer.test()


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--config_path', type=str)
    arguments = argument_parser.parse_args()
    main(arguments.config_path)
    
#old pipe, too many bugs gg