import pprint
import numpy as np
import logging
import tempfile
import os

import autosklearn
import autosklearn.evaluation
import autosklearn.data.xy_data_manager
import autosklearn.util.backend
import autosklearn.constants
import autosklearn.util.pipeline
import autosklearn.metrics
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.classification import AutoSklearnClassifier
from scriptsauto.update_metadata_util import load_task
from autosklearn.evaluation import ExecuteTaFuncWithQueue
from autosklearn.metrics import r2, balanced_accuracy


from smac.stats.stats import Stats
from smac.scenario.scenario import Scenario

tmp_dir = '~/autosklearn_tmp'
out_dir = '~/autosklearn_out'

###############################################################################################################
"""
The goal of this script is to randomly sample a pipeline from the config space of auto-sklearn
and run this randomly sampled pipeline.
"""


# 1. Create a configuration space object
cs = SimpleClassificationPipeline().get_hyperparameter_search_space()

# 2. Sample a configuration/ pipeline from configuration space
config = cs.sample_configuration()
vector_config = config.get_dictionary()
pprint.pprint("Sampled pipeline vector is: ", vector_config)


# 3. Run a single configuration
# Create data manager
task_id = 6
X_train, y_train, X_test, y_test, cat = load_task(task_id)
variable_types = ['categorical' if c else 'numerical'
                  for c in cat]
num_classes = len(np.unique(y_train))
if num_classes == 2:
    task_type = autosklearn.constants.BINARY_CLASSIFICATION
elif num_classes > 2:
    task_type = autosklearn.constants.MULTICLASS_CLASSIFICATION
data_manager = autosklearn.data.xy_data_manager.XYDataManager(
    X=X_train, y=y_train, task=task_type, dataset_name=str(task_id),X_test=X_test,
y_test=y_test, feat_type=variable_types)

# Create backend
backend = autosklearn.util.backend.create(
    temporary_directory=tmp_dir,
    output_directory=out_dir,
    delete_tmp_folder_after_terminate=False,
    )
backend.save_datamanager(data_manager)
logger = logging.getLogger('sample')
stats = Stats(
    Scenario({
        'cutoff_time': 5 * 2,
        'run_obj': 'quality',
    })
)


# Create and run evaluator
evaluator = autosklearn.evaluation.ExecuteTaFuncWithQueue(
    backend=backend,
    autosklearn_seed=30,
    resampling_strategy='holdout',
    logger=logger,
    memory_limit=3072*2,
    metric=balanced_accuracy,
    stats=stats,
    disable_file_output = False,
    all_scoring_functions=True,
    )

print("running evaluator")
status, cost, runtime, additional_run_info = evaluator.run(
    config=config, cutoff=1800)

print("status: ", status, "cost: ", cost, "runtime: ", runtime, "run_info :", additional_run_info)