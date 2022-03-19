import os
import argparse
from configs_and_pipelines.configs_manager import Configs
from configs_and_pipelines.pipelines import build_datasets, train_models, generate


def build_train_generate(configs_path: str, new_configs_path: str = None):
    new_configs_path = configs_path if new_configs_path is None else new_configs_path
    configs = Configs.from_json(configs_path)
    configs = build_datasets(configs=configs,
                             fit_language_masker=True,
                             fast_dev=configs.fast_dev)
    configs.to_json(new_configs_path)
    configs = train_models(configs,
                           train_generator_classifier=True,
                           train_generator=True,
                           fast_dev=configs.fast_dev)
    configs.to_json(new_configs_path)
    generate(configs,
             splits=['validation'],
             fast_dev=configs.fast_dev)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser('DoCoGen Training and Generation')
    parser.add_argument('--configs_path', '-c', required=True,
                        help='path to configuration json file')
    args = parser.parse_args()
    build_train_generate(args.configs_path)
