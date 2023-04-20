#!/usr/bin/env bash

# Fail on error and unset variables.
set -eu -o pipefail

CWD=$(readlink -e "$(dirname "$0")")
cd "${CWD}/.." || exit $?
source ./docker/common.sh

DEVICE=0

docker run \
    -it --rm \
    --name "template" \
    --runtime=nvidia \
    --gpus all \
    --privileged \
    --shm-size 8g \
    -v "${CWD}/..":/workspace \
    -v "/mnt/data":/data \
    -e CUDA_VISIBLE_DEVICES="$DEVICE" \
    ${IMAGE_TAG} \
    "$@" || exit $?
