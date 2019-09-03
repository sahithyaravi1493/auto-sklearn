from autosklearn.pipeline.classification import SimpleClassificationPipeline
import pprint
import sklearn.pipeline
"""
The goal of this script is to randomly sample a pipeline from the config space of auto-sklearn
and run it.
"""
def get_params(stage_str, vector_config):
    stage = vector_config[stage_str + ':__choice__']
    params = {}
    for key, value in vector_config.items():
        if stage in key:
            key = key.replace(stage_str+ ':' + stage + ':', '')
            params.update({key: value})
    print (stage, params)
    return stage, params


# 1. Create a configuration space object
cs = SimpleClassificationPipeline().get_hyperparameter_search_space()

# 2. Sample a configuration/ pipeline from configuration space
config = cs.sample_configuration()
vector_config = config.get_dictionary()
pprint.pprint(vector_config)

# 3. Run a single configuration

classifier, classifier_params = get_params('classifier', vector_config)
preprocessor, preprocessor_params = get_params('preprocessor', vector_config)
rescaler, rescaler_params = get_params('rescaling', vector_config)
categorical_encoder, encoder_params = get_params('categorical_encoding', vector_config)
balancing_strategy =  vector_config['balancing:strategy']
imputation_strategy = vector_config['imputation:strategy']

