import os
import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "mlflow-demo"

client = MlflowClient()

# Retrieve Experiment information
EXPERIMENT_ID = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

# Retrieve Runs information (parameter 'depth', metric 'accuracy')
ALL_RUNS_INFO = client.search_runs(experiment_ids=EXPERIMENT_ID, order_by=["metrics.accuracy DESC"]).to_list()
best_run = ALL_RUNS_INFO[0]
best_model_path = best_run.info.artifact_uri
best_model = mlflow.sklearn.load_model(best_model_path+"/classifier")
print(best_run.data.metrics['accuracy'])

# Delete runs (DO NOT USE UNLESS CERTAIN)
for runs in ALL_RUNS_INFO:
    client.delete_run(runs.info.run_id)

# Delete experiment (DO NOT USE UNLESS CERTAIN)
client.delete_experiment(EXPERIMENT_ID)