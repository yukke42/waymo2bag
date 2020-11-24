FROM ros:melodic-ros-base

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive \
        apt-get -y install --no-install-recommends \
            ros-melodic-cv-bridge \
            ros-melodic-vision-opencv \
            ros-melodic-tf \
            python-pip python-matplotlib \
    && rm -rf /var/lib/apt/lists/*

COPY . /waymo2bag
COPY pip_pkg_build/waymo_open_dataset_tf_2_0_0-1.2.0-cp27-cp27mu-manylinux2010_x86_64.whl /tmp
RUN pip install -e /waymo2bag
RUN pip install --upgrade pip \
  && pip install /tmp/waymo_open_dataset_tf_2_0_0-1.2.0-cp27-cp27mu-manylinux2010_x86_64.whl

WORKDIR /data

#dataENTRYPOINT ["/waymo2bag/docker_entrypoint.sh"]
