from autosklearn.pipeline.classification import SimpleClassificationPipeline
import pprint
from autosklearn.classification import AutoSklearnClassifier
from scriptsauto.update_metadata_util import load_task
from autosklearn.evaluation import ExecuteTaFuncWithQueue
from autosklearn.metrics import r2, balanced_accuracy
import logging
from smac.stats.stats import Stats
from smac.scenario.scenario import Scenario
from test.test_evaluation.evaluation_util import get_multiclass_classification_datamanager
import os
"""
The goal of this script is to randomly sample a pipeline from the config space of auto-sklearn
and run it.
"""

# 1. Create a configuration space object
cs = SimpleClassificationPipeline().get_hyperparameter_search_space()

# 2. Sample a configuration/ pipeline from configuration space
config = cs.sample_configuration()
vector_config = config.get_dictionary()
pprint.pprint(vector_config)

# 3. Run a single configuration
task_id = 5
X_train, y_train, X_test, y_test, cat = load_task(task_id)
seed = 30
tmp_dir = '~/autosklearn_files'
automl_arguments = {
    'time_left_for_this_task': 30,
    'per_run_time_limit': 5,
    'initial_configurations_via_metalearning': 0,
    'ensemble_size': 0,
    'ensemble_nbest': 0,
    'seed': seed,
    'ml_memory_limit': 3072,
    'resampling_strategy': 'holdout',
    'resampling_strategy_arguments': {'folds': 10},
    'delete_tmp_folder_after_terminate': False,
    'tmp_folder': tmp_dir,
    'disable_evaluator_output': False,
}
automl = AutoSklearnClassifier(**automl_arguments)
automl.fit(X_train, y_train, dataset_name=str(task_id), metric=balanced_accuracy,
           feat_type=cat)
data = automl._automl[0]._backend.load_datamanager()



# Data manager can't be replaced with save_datamanager, it has to be deleted
# first
os.remove(automl._automl[0]._backend._get_datamanager_pickle_filename())
data.data['X_test'] = X_test
data.data['Y_test'] = y_test
automl._automl[0]._backend.save_datamanager(data)
stats = Stats(
    Scenario({
        'cutoff_time': 5 * 2,
        'run_obj': 'quality',
    })
)
stats.start_timing()
# To avoid the output "first run crashed"...
stats.ta_runs += 1
ta = ExecuteTaFuncWithQueue(backend= automl._automl[0]._backend,
                            logger = logging.getLogger('Testing:)'),
                            stats=stats,
                                    autosklearn_seed=seed,
                                    resampling_strategy='test',
                                    memory_limit=3072*2,
                                    disable_file_output=True,
                                    all_scoring_functions=True,
                                    metric=balanced_accuracy)
# status, cost, runtime, additional_run_info = ta.start(
#     config=config, instance=None, cutoff=15)
status, cost, runtime, additional_run_info = ta.run(
                config=config, cutoff=15, instance=None)



# Print test accuracy and statistics.
#y_pred = automl.predict(X_test)
