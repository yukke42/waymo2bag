#!/bin/bash

set -e

[[ ! -d "waymo-od" ]] && git clone https://github.com/waymo-research/waymo-open-dataset waymo-od
cat build.sh.in > waymo-od/pip_pkg_scripts/build.sh
cd waymo-od
git checkout a476ab

mkdir -p /tmp/pip_pkg_build
docker build --tag=open_dataset_pip -f pip_pkg_scripts/build.Dockerfile .
docker run --mount type=bind,source=/tmp/pip_pkg_build,target=/tmp/pip_pkg_build \
  -e "GITHUB_BRANCH=master" -e "PYTHON_VERSION=2" -e "PYTHON_MINOR_VERSION=7" \
  -e "PIP_MANYLINUX2010=1" -e "TF_VERSION=2.0.0" open_dataset_pip
