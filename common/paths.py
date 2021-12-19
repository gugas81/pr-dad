from os import path
from pathlib import Path


class PATHS:
    PROJECT_ROOT = path.join(*Path(path.dirname(path.abspath(__file__))).parts[: -1])
    LOG = path.join(PROJECT_ROOT, 'logs')
    PROJECT_PR_NAME = 'phase-retrieval'
    DATASETS = '/data/datasets'
    DATASETS_S3 = f's3://{PROJECT_PR_NAME}/datasets'
    CLEARML_BUCKET = 's3://data-clearml'
    S3_CML_PROJECT = path.join(CLEARML_BUCKET, PROJECT_PR_NAME)
    MODEL_CACHE_LOCAL = '/data/cache'
