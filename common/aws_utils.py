from typing import Callable, Optional, Iterable, List, Sequence, Any
import tempfile
from pathlib import Path
from multiprocessing.pool import ThreadPool
import os
import s3fs


class S3FileSystem(s3fs.S3FileSystem):
    HOST = 's3'
    URL_PREFIX = 's3://'
    S3_CML_PATH = 's3://data-clearml/phase-retrieval'

    def __init__(self, *, key=None, secret=None):
        super().__init__(key=key,
                         secret=secret,
                         default_fill_cache=False,
                         use_listings_cache=False)

    @classmethod
    def is_s3_url(cls, url: str) -> bool:
        return url.startswith(cls.URL_PREFIX)

    @classmethod
    def s3_url(cls, path: str) -> str:
        return cls.URL_PREFIX + path

    def load_object(self, url: str, loader: Callable) -> Any:
        assert self.isfile(url), f'not file:{url}'

        suffix = Path(os.path.basename(url)).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix) as local_temp_file:
            self.download(url, local_temp_file.name)
            obj = loader(local_temp_file.name)

        return obj

    def save_object(self, url: str, saver: Callable, obj: Optional[object] = None) -> None:

        suffix = Path(os.path.basename(url)).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix) as local_temp_file:
            if obj is None:
                saver(local_temp_file.name)
            else:
                saver(obj, local_temp_file.name)
            self.upload(local_temp_file.name, url)

        if not self.isfile(url):
            raise RuntimeError(f'Cannot save an object in s3_path:{url}')



