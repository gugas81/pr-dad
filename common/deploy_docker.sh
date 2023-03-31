#!/usr/bin/env bash

# Usage:
#    $ sh ./deploy_worker.sh [<image_tag_name>]
# Example
#    $ sh ./deploy_worker.sh
#    $ sh ./deploy_worker.sh develop

set -e  # stop on error.

if [[ -z "$1" ]]; then
	echo "A tag name for your Docker image: "
	read TAG_NAME
else
	TAG_NAME=$1
fi

echo 'Docker image tag name:' ${TAG_NAME}

echo 'Building Docker image...'
GIT_HEAD_HASH=`git log -1 --format=%h`
CWD=`dirname "$0"`
docker build --build-arg GIT_HEAD_HASH=${GIT_HEAD_HASH} -t anonym -f ${CWD}/../Dockerfile ${CWD}/..

echo 'Tagging Docker image:'
docker tag anonym:latest XXXX.dkr.ecr.us-west-2.amazonaws.com/anonym:${TAG_NAME}

echo 'Logging in to AWS ECR:'
aws ecr get-login-password --region us-west-2 | docker login \
  --username AWS \
  --password-stdin \
  xxxx.dkr.ecr.us-west-2.amazonaws.com

echo 'Pushing Docker image to AWS ECR:' ${TAG_NAME}
docker push xxxxx.dkr.ecr.us-west-2.amazonaws.com/anonym:${TAG_NAME}
