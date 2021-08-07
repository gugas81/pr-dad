from os import path
from pathlib import Path


class PATHS:
    PROJECT_ROOT = path.join(*Path(path.dirname(path.abspath(__file__))).parts[: -1])
    LOG = path.join(PROJECT_ROOT, 'logs')
    DATASETS = '/data/datasets'
    DATASETS_S3 = 's3://phase-retrieval/datasets'
    CLEARML_BUCKET = 's3://data-clearml'
