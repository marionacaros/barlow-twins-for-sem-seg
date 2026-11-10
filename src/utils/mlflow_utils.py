"""
Optional MLflow tracking for the training and test entry points.

Tracking is off unless a tracking URI is given with --mlflow_uri or the MLFLOW_TRACKING_URI environment variable,
so the scripts (and the tests) run without mlflow installed. All the stages of one dataset log to the experiment
`barlow_twins_<dataset>` by default; test metrics are added to the fine-tuning run that produced the checkpoint.
"""
import logging
import math
import os

# the MLflow 3 UI shows experiments without this tag as GenAI experiments (traces, no metric charts)
EXPERIMENT_KIND_TAG = 'mlflow.experimentKind'
EXPERIMENT_KIND = 'custom_model_development'


def mark_ml_experiment(client, experiment_id):
    """
    Tag an experiment as a model-training experiment, so the MLflow UI shows its runs and metric charts.

    :param client: mlflow.MlflowClient
    :param experiment_id: str
    """
    if EXPERIMENT_KIND_TAG not in client.get_experiment(experiment_id).tags:
        client.set_experiment_tag(experiment_id, EXPERIMENT_KIND_TAG, EXPERIMENT_KIND)


def add_mlflow_args(parser):
    """
    Add --mlflow_uri and --mlflow_experiment to an argparse parser.

    :param parser: argparse.ArgumentParser
    """
    parser.add_argument('--mlflow_uri', type=str, default=os.environ.get('MLFLOW_TRACKING_URI'),
                        help='MLflow tracking URI, e.g. http://localhost:5000 (default: $MLFLOW_TRACKING_URI; '
                             'tracking is off when neither is set)')
    parser.add_argument('--mlflow_experiment', type=str, default=None,
                        help='MLflow experiment name (default: barlow_twins_<dataset>)')


def experiment_name(experiment, dataset):
    """
    :param experiment: str or None, name given on the command line
    :param dataset: str, dataset key
    :return: str
    """
    return experiment or f'barlow_twins_{dataset}'


def segmentation_metrics(metrics, class_names, prefix):
    """
    Flatten a SegmentationMetrics into MLflow metrics.

    :param metrics: SegmentationMetrics
    :param class_names: list of str
    :param prefix: str, e.g. 'test'
    :return: dict {name: float}
    """
    out = {f'{prefix}_mIoU': metrics.mean_iou(),
           f'{prefix}_accuracy': metrics.accuracy(),
           f'{prefix}_mIoU_per_sample': metrics.sample_mean_iou()}
    out.update({f'{prefix}_iou_{c}': v for c, v in zip(class_names, metrics.iou_per_class())})
    return out


class MLflowRun:
    """
    Context manager around one MLflow run; every method is a no-op when no tracking URI is given.

    Example:
        with MLflowRun(uri, 'barlow_twins_dales', run_name='finetune', params=vars(args)) as run:
            run.log_metrics({'val_loss': 0.3}, step=epoch)
    """

    def __init__(self, uri, experiment, run_name=None, run_id=None, params=None, tags=None):
        """
        :param uri: str or None, tracking URI (None disables tracking)
        :param experiment: str, experiment name (used when a new run is started)
        :param run_name: str or None
        :param run_id: str or None, resume this run instead of starting a new one
        :param params: dict or None, logged once as run parameters
        :param tags: dict or None
        """
        self.enabled = bool(uri)
        self.uri, self.experiment, self.run_name = uri, experiment, run_name
        self.resume_id, self.params, self.tags = run_id, params or {}, tags or {}
        self._mlflow = None
        self.run_id = None

    def __enter__(self):
        if not self.enabled:
            return self
        import mlflow
        self._mlflow = mlflow
        mlflow.set_tracking_uri(self.uri)
        run = None
        if self.resume_id:
            try:
                run = mlflow.start_run(run_id=self.resume_id)
            except Exception as e:  # e.g. the run was logged to another tracking server
                logging.warning(f'Cannot resume MLflow run {self.resume_id} ({e}); starting a new run')
        if run is None:
            mlflow.set_experiment(self.experiment)
            run = mlflow.start_run(run_name=self.run_name)
        self.run_id = run.info.run_id
        mark_ml_experiment(mlflow.MlflowClient(), run.info.experiment_id)
        if self.params:
            mlflow.log_params({k: str(v)[:6000] for k, v in self.params.items()})
        if self.tags:
            mlflow.set_tags(self.tags)
        logging.info(f'MLflow run {self.run_id} ({self.uri})')
        return self

    def log_metrics(self, metrics, step=None):
        """
        Log a dict of scalar metrics; non-finite values (e.g. IoU of an absent class) are skipped.

        :param metrics: dict {name: float}
        :param step: int or None
        """
        if not self.enabled:
            return
        finite = {k: float(v) for k, v in metrics.items() if v is not None and math.isfinite(float(v))}
        if finite:
            self._mlflow.log_metrics(finite, step=step)

    def set_tags(self, tags):
        """
        :param tags: dict {name: str}
        """
        if self.enabled:
            self._mlflow.set_tags(tags)

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            self._mlflow.end_run(status='FAILED' if exc_type else 'FINISHED')
        return False


def lightning_loggers(output_dir, uri, experiment, run_name, params):
    """
    Loggers for a Lightning Trainer: TensorBoard under `output_dir` (the Trainer default) plus MLflow when `uri` is
    given.

    :param output_dir: str, Lightning root folder
    :param uri: str or None, MLflow tracking URI
    :param experiment: str, MLflow experiment name
    :param run_name: str, MLflow run name
    :param params: dict, logged as run parameters
    :return: list of Lightning loggers
    """
    from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

    loggers = [TensorBoardLogger(output_dir)]
    if uri:
        mlflow_logger = MLFlowLogger(experiment_name=experiment, run_name=run_name, tracking_uri=uri,
                                     log_model=False)
        mark_ml_experiment(mlflow_logger.experiment, mlflow_logger.experiment_id)
        mlflow_logger.log_hyperparams({k: str(v)[:6000] for k, v in params.items()})
        logging.info(f'MLflow run {mlflow_logger.run_id} ({uri})')
        loggers.append(mlflow_logger)
    return loggers
