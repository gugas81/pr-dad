from typing import Callable, Optional, Iterable, List, Sequence, Any
import tempfile
from pathlib import Path
from multiprocessing.pool import ThreadPool
import os
import s3fs
from urllib.parse import urlparse

class S3FileSystem(s3fs.S3FileSystem):
    HOST = 's3'
    URL_PREFIX = 's3://'

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


class CloudPath(ABC):

    SEP = '/'
    URL_PREFIX = 'schema://'

    @classmethod
    def with_prefix(cls, url: str) -> bool:
        return url.startswith(cls.URL_PREFIX)

    @classmethod
    def parts(cls, key: str) -> List[str]:
        return key.split(cls.SEP)

    @classmethod
    def join(cls, *keys: str) -> str:
        return cls.SEP.join(keys)

    @classmethod
    def join_url_local(cls, url_path: str, local_path: str) -> str:
        out_bucket_name, out_key = cls.parse_url(url_path)
        out_key = cls.join(out_key, local_path)
        out_url = cls.make_url(out_bucket_name, out_key)
        return out_url

    @classmethod
    def relative_path(cls, key: str, base_key: str) -> str:
        len_path_base = len(cls.parts(base_key))
        rel_path_parts = cls.parts(key)[len_path_base:]
        rel_path = cls.join(*rel_path_parts)
        return rel_path

    @classmethod
    def os_path(cls, key: str) -> str:
        path_parts = cls.parts(key)
        return path.join(*path_parts)

    @classmethod
    def basename(cls, key: str) -> str:
        return cls.parts(key)[-1]

    @classmethod
    def dir_name(cls, key: str) -> str:
        if key[-1] == cls.SEP:
            dir_name = key[:-1]
        else:
            path_parts = cls.parts(key)
            dir_name = cls.join(*path_parts[:-1]) if len(path_parts) > 1 else path_parts[0]
        return dir_name

    @classmethod
    def split_head(cls, key: str) -> (str, str):
        split = key.split(cls.SEP, maxsplit=1)
        head = split[0]
        tail = split[1] if len(split) > 1 else ''
        return head, tail

    @classmethod
    def from_os_path_to_url(cls, os_path: str) -> str:
        return cls.join(*Path(os_path).parts)

    @classmethod
    def parse_url(cls, url: str) -> (str, str):
        """
        Split a URL to bucket name and key
        :param url: URL to parse
        :return: bucket, key
        """
        assert isinstance(url, str), f'url={url} must be a string'
        assert url.startswith(cls.URL_PREFIX), f'url must start with prefix {cls.URL_PREFIX}, url={url}'

        bucket_name = urlparse(url).hostname

        key = url[len(cls.URL_PREFIX) + len(bucket_name) + 1:]
        return bucket_name, key

    @classmethod
    def make_url(cls, bucket: str, key: str = None) -> str:
        """
        Make a URL from bucket name and key
        """
        if not cls.with_prefix(bucket):
            url_path = cls.URL_PREFIX + bucket
        if key is not None:
            url_path = url_path + cls.SEP + key
        return url_path

    @classmethod
    def norm_url(cls, url: str):
        norm_url = url
        if norm_url.startswith(cls.URL_PREFIX):
            norm_url = norm_url[len(cls.URL_PREFIX):]

        norm_url = norm_url.replace('\\', '/')
        norm_url = norm_url.replace('//', '/')

        if norm_url.endswith('/'):
            norm_url = norm_url[:-1]

        norm_url = f'{cls.URL_PREFIX}{norm_url}'

        return norm_url

class S3Path(CloudPath):

    SEP = '/'
    HOST = 's3'
    URL_PREFIX = 's3://'
