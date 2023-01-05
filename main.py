import argparse 
import wandb
from audience.data import get_datapipeline
from audience.model import get_model
from audience.utils.config import Config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--wandb_name', type=str, default=None)
    args = parser.parse_args()
    return args

def initalize(config):
    # sets up the data pipeline and loads the model
    pipeline = get_datapipeline(config.experiment.data_pipeline)(config.experiment.dataset_dir)

    # intialize the model, and take its tokenizer
    if len(config.experiment.model_names) == 1:
        model = get_model(config.experiment.model_names[0])()
        tokenizer = model.tokenizer
    else:
        # get an array of models
        models = [get_model(name) for name in config.experiment.model_names]
        tokenizers = [model.tokenizer for model in models]

    # create the factories

    # TODO: fix this!
    tokenizer_func = lambda x: tokenizer(x, return_tensors='pt')

    precondition_factory, update_factory = pipeline.create_factories(tokenizer_func, pipeline.precondition_factory, pipeline.update_factory)
    print("Meow")

    return models, pipeline, precondition_factory, update_factory

if __name__ == '__main__':
    args = parse_args()
    # load config
    config = Config.load_yaml(args.config)

    # set up wandb
    #wandb.init(project='audience', name=args.wandb_name)
    # set config 
    #wandb.config.update(config.to_dict())
    initalize(config)
