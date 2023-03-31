import os
import sys
# 3rd party:
import boto3
import time
import subprocess
import logging
import inspect

def get_project_root_path(module: object) -> str:
    """
    Returns the path to the root directory of the current project, assuming the project
    is structured as [project_root/sources_root], where sources_root the directory which contains all source files.
    :return: Returns the path to the root directory of the current project.
    """
    project_src_path = os.path.dirname(os.path.realpath(module.__file__))
    project_root_path = os.path.realpath(os.path.join(project_src_path, '..'))
    return project_root_path

class AwsBatchJobsMonitor(object):
    _logger = logging.getLogger(__file__)
    _batch_client = None
    _ecr_client = None
    _region = 'us-west-2'
    _access_key = None
    _secret_key = None

    @staticmethod
    def delete_job_queue(queue_name):
        AwsBatchJobsMonitor._set_job_queue(queue_name, 'DISABLED')
        client = AwsBatchJobsMonitor.get_batch_client()
        response = client.delete_job_queue(jobQueue=queue_name)
        AwsBatchJobsMonitor._logger.info(f'job que {queue_name} deleted: \n{response}')
        return response

    @staticmethod
    def kill_all_running_jobs(queue_name):
        all_running_jobs = AwsBatchJobsMonitor.list_running_jobs(queue_name)
        for job in all_running_jobs:
            AwsBatchJobsMonitor.kill_job(job['jobId'])
        msg = f'terminated {len(all_running_jobs)} jobs on que: {queue_name}'
        AwsBatchJobsMonitor._logger.info(msg)
        return msg

    @staticmethod
    def kill_job(job_id, reason='because'):
        client = AwsBatchJobsMonitor.get_batch_client()
        client.terminate_job(jobId=job_id, reason=reason)
        AwsBatchJobsMonitor._logger.info(f'terminated job: {job_id}')
        return

    @staticmethod
    def list_running_jobs(queue_name):
        client = AwsBatchJobsMonitor.get_batch_client()
        jobs = client.list_jobs(jobQueue=queue_name)
        AwsBatchJobsMonitor._logger.info(f'found {len(jobs)} running jobs on queue {queue_name}')
        return jobs['jobSummaryList']

    @staticmethod
    def _set_job_queue(queue_name, set_state_trgt):
        client = AwsBatchJobsMonitor.get_batch_client()
        client.update_job_queue(jobQueue=queue_name, state=set_state_trgt)
        time.sleep(1)
        max_wait = 20
        waited = 0.
        queue_ready = False
        while not queue_ready and waited <= max_wait:
            time.sleep(1)
            waited += 1
            queue_ready = client.describe_job_queues(jobQueues=[queue_name])['jobQueues'][0]['state'] == set_state_trgt
        AwsBatchJobsMonitor._logger.info(f'waited {waited} seconds to {set_state_trgt} job que')
        if not queue_ready:
            AwsBatchJobsMonitor._logger.warning(f'waited {waited} seconds to {set_state_trgt} job que')
            raise TimeoutError('AWS_NOT_RESPONDING')
        return

    @classmethod
    def get_batch_client(cls):
        if cls._batch_client is None:
            client = boto3.client('batch',
                                  region_name=cls._region,
                                  aws_access_key_id=cls._access_key,
                                  aws_secret_access_key=cls._secret_key)
            cls._batch_client = client
        return cls._batch_client

    @classmethod
    def get_available_docker_images(cls, repo_name: str):
        ecr = cls.get_ecr_client()
        try:
            response = ecr.describe_images(repositoryName=repo_name)
            images = [
                f'{x["repositoryName"]}:{tag}'
                for x in response['imageDetails']
                for tag in x.get('imageTags', [])
            ]
            return images
        except ecr.exceptions.RepositoryNotFoundException:
            raise ValueError(f'Could not find Docker repository "{repo_name}" in ECR.')

    @classmethod
    def get_all_jobs_definitions(cls):
        batch = cls.get_batch_client()
        return batch.describe_job_definitions()['jobDefinitions']

    @classmethod
    def get_ecr_client(cls):
        if cls._ecr_client is None:
            cls._ecr_client = boto3.client('ecr',
                                           region_name=cls._region,
                                           aws_access_key_id=cls._access_key,
                                           aws_secret_access_key=cls._secret_key)
        return cls._ecr_client



class JobCommon(object):
    DEFAULT_DOCKER_IMAGE_NAME = ''
    DEFAULT_INSTANCE_TYPE = 'p3-spot'
    DEFAULT_REGION = 'us-west-2'

    SUPPORTED_EC2_INSTANCE_TYPES = {'p2', 'p2-spot', 'p3', 'p3-spot', 'p3-spot-big'}
    INSTANCE_TYPE_TO_VCPUS_LIMIT = {
        'p2': 4,
        'p2-spot': 4,
        'p3': 8,
        'p3-spot': 8,
        'p3-spot-big': 8
    }
    INSTANCE_TYPE_TO_MEMORY_LIMIT = {
        'p2': 60_000,
        'p2-spot': 60_000,
        'p3': 60_000,
        'p3-spot': 60_000,
        'p3-spot-big': 60_000
    }
    INSTANCE_TYPE_TO_JOB_QUEUE_NAMES = {
        'p2': 'batch-jq-p2',
        'p2-spot': 'batch-jq-p2-spot',
        'p3': 'batch-jq-p3',
        'p3-spot': 'batch-jq-p3-spot',
        'p3-spot-big': 'batch-jq-p3-spot-big'
    }

    ECR_BASE_URL = None
    NUMBER_OF_RETIRES_FOR_JOB = 5
    DEFAULT_BUCKET = None
    RES_GPU = [{'type': 'GPU', 'value': '1'}]


class CommandContainer(object):

    def __init__(self, ecr_tag, entry_function, arguments):
        assert hasattr(entry_function, '__call__'), f'entry function {entry_function} should be a function'
        assert isinstance(arguments, dict)
        assert isinstance(ecr_tag, str)
        assert not entry_function.__module__.startswith('__main__'), 'entry should not be in the same file as submitter'
        self._validate_function_arguments(entry_function, arguments)

        self.ecr_tag = ecr_tag
        self.arguments = arguments
        self.entry_function = entry_function
        relative_path = entry_function.__module__
        self.relative_path = os.path.normpath(relative_path.replace(r'.', os.path.sep) + '.py')
        self.func_name = entry_function.__name__

    @staticmethod
    def _validate_function_arguments(func, args_dict):
        sig = inspect.signature(func)
        func_args = [sig.parameters[p].name for p in sig.parameters]

        unexpected_args = set(args_dict.keys()) - set(func_args)
        assert not unexpected_args, f'`arguments` include arguments which are ' \
                                    f'not expected by `entry_func`: {unexpected_args}'

        defaultless_func_params = [sig.parameters[p].name for p in sig.parameters
                                   if sig.parameters[p].default == inspect.Parameter.empty]

        missing_params = set(defaultless_func_params) - set(args_dict.keys())
        assert not missing_params, f'`arguments` are missing some mandatory ' \
                                   f'arguments which are expected by `entry_func`: {missing_params}'

    def to_command(self):
        command = ['python', self.relative_path, self.func_name]
        for key, value in self.arguments.items():
            command.extend([f'--{key}', str(value)])
        return command


class AwsBatchJobSubmitter(object):
    def __init__(self, module):
        self._logger = logging.getLogger('JobSubmitter')
        self._batch = AwsBatchJobsMonitor.get_batch_client()
        self._module = module

    def _get_job_definition(self, ecr_tag, instance_type):
        ecr_tag_name = ecr_tag.replace(':', '-')
        job_definition_name = f'batch-{ecr_tag_name}-{instance_type}'
        job_definition_revision = 0

        # fetch the entire list of job definitions.
        job_definitions = AwsBatchJobsMonitor.get_all_jobs_definitions()

        # find the last revision of the relevant job definition.
        for job_definition in job_definitions:
            same_job_def_name = job_definition['jobDefinitionName'] == job_definition_name
            if same_job_def_name and job_definition_revision < job_definition['revision']:
                job_definition_revision = job_definition['revision']

        # The job definition does not already exists. Create a new one.
        if job_definition_revision == 0:
            job_definition_revision = self._create_new_job_def(ecr_tag, instance_type, job_definition_name)

        job_definition = f'{job_definition_name}:{job_definition_revision}'
        return job_definition

    def _create_new_job_def(self, ecr_tag, instance_type, job_definition_name):
        self._logger.debug(f'A job definition named {job_definition_name} does not exist. Creating a new one')
        new_job_definition = self._batch.register_job_definition(
            jobDefinitionName=job_definition_name,
            type='container',
            containerProperties={
                'image': JobCommon.ECR_BASE_URL + ecr_tag,
                'vcpus': JobCommon.INSTANCE_TYPE_TO_VCPUS_LIMIT[instance_type],
                'memory': JobCommon.INSTANCE_TYPE_TO_MEMORY_LIMIT[instance_type],
                'resourceRequirements': JobCommon.RES_GPU
            },

        )
        job_definition_revision = new_job_definition['revision']
        return job_definition_revision

    def submit(self,
               command_container: CommandContainer,
               job_name: str,
               instance_type: str = JobCommon.DEFAULT_INSTANCE_TYPE,
               local_run: bool = False,
               retries: int = JobCommon.NUMBER_OF_RETIRES_FOR_JOB) -> None:
        """
        Submit job to AWS BATCH
        :param command_container: command which job have to run on container(docker)
        :param job_name: name of job for submit
        :param instance_type: name of instance type from JobCommon.SUPPORTED_EC2_INSTANCE_TYPES
        :param local_run: is run on local machine without submit to AWS BATCH
        :param retries: how time to retry to submit job
        :return:
        """

        assert isinstance(command_container, CommandContainer)
        assert instance_type in JobCommon.SUPPORTED_EC2_INSTANCE_TYPES, \
            f'instance_type{instance_type} must be from SUPPORTED_EC2_INSTANCE_TYPES: ' \
            f'{JobCommon.SUPPORTED_EC2_INSTANCE_TYPES}'

        ecr_tag = command_container.ecr_tag
        self._verify_docker_image_exists(ecr_tag)
        command = command_container.to_command()

        if local_run:
            self._local_run(command)
        else:
            job_definition = self._get_job_definition(ecr_tag, instance_type)
            command_str = ' '.join(command)
            self._logger.info(f'[{ecr_tag}] Submitted {job_name}...\nCommand: {command_str}')
            self._batch.submit_job(
                jobName=job_name,
                jobQueue=JobCommon.INSTANCE_TYPE_TO_JOB_QUEUE_NAMES[instance_type],
                jobDefinition=job_definition,
                containerOverrides={'command': command},
                retryStrategy={'attempts': retries}
            )

    def _local_run(self, command):
        proj_root = get_project_root_path(self._module)
        command[1] = os.path.join(proj_root, command[1])
        command_str = ' '.join(command)
        self._logger.info(f'(local_run) Submit Command: \n{command_str}')
        command[0] = sys.executable
        process = subprocess.Popen(command)
        process.communicate()

    @staticmethod
    def _create_job_command(command_container):
        assert isinstance(command_container, CommandContainer)
        command = command_container.to_command()
        return command

    @staticmethod
    def _verify_docker_image_exists(image_name):
        repo_name, tag_name = image_name.split(':')
        image_names = AwsBatchJobsMonitor.get_available_docker_images(repo_name)
        if image_name not in image_names:
            print('Could not find docker image {} in AWS ECR.'.format(image_name))
            raise ValueError('No Docker Image Found')

def example_function_relative_path():
    import common as common_module
    # get the relative path of this file within the common project.
    project_root_dir = get_project_root_path(common_module)
    rel_path = os.path.realpath(__file__)[len(project_root_dir):].strip('/')
    return rel_path


def example_function(stuff):
    msg = f'worker code doing {stuff}'
    print(msg)
    return 'success'

def example():
    import common as common_module
    arguments = {'stuff': 'important_stuff!'}
    command_container = CommandContainer(JobCommon.DEFAULT_DOCKER_IMAGE_NAME, example_function, arguments)
    job_submitter = AwsBatchJobSubmitter(module=common_module)
    job_submitter.submit(command_container=command_container, job_name='my_job',
                         instance_type=JobCommon.DEFAULT_INSTANCE_TYPE, local_run=True)


if __name__ == '__main__':
    example()
