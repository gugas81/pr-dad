from os import path
from pathlib import Path


class PATHS:
    PROJECT_ROOT = path.join(Path.parts(path.dirname(path.abspath(__file__)))[: -2])
    LOG = path.join(PROJECT_ROOT, 'logs')
    DATASETS = '/data/datasets'
    CLEARML_BUCKET = 's3://data-clearml'
