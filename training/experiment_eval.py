import fire
import os

from typing import List, Optional, Callable, Union

from common import S3FileSystem

from training.phase_retrival_evaluator import Evaluator

S3_CML_PATH = 's3://data-clearml/phase-retrieval'
S3_OUT_PATH = 's3://phase-retrieval/eval'


def experiment_eval(name: str, out_path: Optional[str] = None):
    s3 = S3FileSystem()
    experiment_paths = s3.glob(os.path.join(S3_CML_PATH, name) + '*')
    assert len(experiment_paths) == 1, f'not valid experiment_path: {experiment_paths}'
    experiment_url = s3.s3_url(experiment_paths[0])
    assert s3.exists(experiment_url)
    model_url = os.path.join(experiment_url, 'models', 'phase-retrieval-gan-model.pt')
    assert s3.isfile(model_url)

    out_path = os.path.join(S3_OUT_PATH, name) if out_path is None else out_path

    Evaluator(model_type=model_url).benchmark_evaluation(save_out_url=out_path)


if __name__ == '__main__':
    fire.Fire(experiment_eval)
