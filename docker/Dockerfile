# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA=12.2.2
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu20.04
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA

# Use bash to support string substitution.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet \
        build-essential \
        cmake \
        cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
        git \
        hmmer \
        kalign \
        tzdata \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove --yes \
    && apt-get clean

# Compile HHsuite from source.
RUN git clone --branch v3.3.0 --single-branch https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
    && mkdir /tmp/hh-suite/build \
    && pushd /tmp/hh-suite/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make --jobs 4 && make install \
    && ln -s /opt/hhsuite/bin/* /usr/bin \
    && popd \
    && rm -rf /tmp/hh-suite

# Install Miniconda package manager.
RUN wget -q -P /tmp \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh

# Install Conda packages.
ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"
RUN conda install --quiet --yes conda==24.11.1 pip python=3.11 \
    && conda install --quiet --yes --channel nvidia cuda=${CUDA} \
    && conda install --quiet --yes --channel conda-forge openmm=8.0.0 pdbfixer \
    && conda clean --all --force-pkgs-dirs --yes

COPY . /app/alphafold
RUN wget -q -P /app/alphafold/alphafold/common/ \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Install pip packages.
RUN pip3 install --upgrade pip --no-cache-dir \
    && pip3 install -r /app/alphafold/requirements.txt --no-cache-dir \
    && pip3 install --upgrade --no-cache-dir \
      jax==0.4.26 \
      jaxlib==0.4.26+cuda12.cudnn89 \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Add SETUID bit to the ldconfig binary so that non-root users can run it.
RUN chmod u+s /sbin/ldconfig.real

# Currently needed to avoid undefined_symbol error.
RUN ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 /opt/conda/lib/libffi.so.7

# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for
# details.
# ENTRYPOINT does not support easily running multiple commands, so instead we
# write a shell script to wrap them up.
# WORKDIR /app/alphafold
# RUN echo $'#!/bin/bash\n\
# ldconfig\n\
# python /app/alphafold/run_alphafold.py "$@"' > /app/run_alphafold.sh \
#   && chmod +x /app/run_alphafold.sh
# ENTRYPOINT ["/app/run_alphafold.sh"]

# =============================================================================
# Install Phenix dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet \
        apache2-dev \
        libpcre2-dev \
        libssl-dev \
        locales \
        ssh \
        tcsh \
    && locale-gen en_US.UTF-8 en_GB.UTF-8 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove --yes \
    && apt-get clean

RUN conda install -y -c conda-forge \
      curl \
      flask \
      future \
      hdf5=1.14 \
      libacl \
      libboost-python-devel=1.86 \
      libboost-devel=1.86 \
      libtiff \
      matplotlib \
      mrcfile \
      msgpack-cxx \
      psutil \
      pybind11 \
      scons \
      setuptools \
      pyzmq

# pass in user and pass for apache server
ARG USER=apache_server
ARG PASS=password-for-phenix-rest-server
ARG ROOT_PASS=password-for-phenix-rest-server
ARG HTTPD_VERSION=2.4.63
ARG MOD_WSGI_VERSION=5.0.2

# Copy just directories required to set up environment
# Get Phenix from /app/external_phenix which will mount an
#    external version of phenix (/net/cci-gpu-00/raid1/scratch1/terwill/PHENIX)
#
#  Requires environmental variable specifying the apacher_server user
#

# Put versions of flaskapp.wsgi and httpd.conf that refer to this
# specific configuration at:
# /usr/local/apache2/htdocs/FlaskApp/
#
#  Create a path so that an external Phenix installation can be used:
#     Make a directory for phenix that is internal: /app/external_phenix
#     Then create a dummy full path that is identical to
#     the actual full path for external phenix, and link it
#     to /app/external_phenix, where we are going to mount this external
#     directory in the run docker step. Now a lookup inside the container of the
#     full path goes to /app/external_phenix and if we source
#     /app/external_phenix/build/setpaths.csh it will work correctly

RUN mkdir -p /app/apache_scripts

COPY ./apache_scripts /app/apache_scripts

RUN cd /app/apache_scripts \
    && mkdir -p /usr/local/apache2/conf \
    && mkdir -p /usr/local/apache2/htdocs/FlaskApp \
    && cp flaskapp.wsgi /usr/local/apache2/htdocs/FlaskApp/flaskapp.wsgi \
    && cp httpd.conf /usr/local/apache2/conf/httpd.conf \
    && cp httpd-ssl.conf /usr/local/apache2/conf/httpd-ssl.conf \
    && useradd -m -s /bin/bash $USER && echo "${USER}:${PASS}" | chpasswd  \
    && mkdir -p /home/${USER} \
    && echo ". /app/phenix/build/setpaths.sh" >> /home/${USER}/.bashrc

# Create self-signed certificate
RUN cd /usr/local/apache2/conf \
    && /app/apache_scripts/make_certificate.sh

# Compile httpd from source
RUN wget -q -P /usr/local/src/ \
  https://dlcdn.apache.org/httpd/httpd-${HTTPD_VERSION}.tar.bz2
RUN cd /usr/local/src \
    && tar -xf httpd-${HTTPD_VERSION}.tar.bz2 \
    && cd httpd-${HTTPD_VERSION} \
    && ./configure --with-ssl=/opt/conda \
    && make -j 8 \
    && make install

# Compile mod_wsgi from source
RUN wget -q -P /usr/local/src/ \
  https://github.com/GrahamDumpleton/mod_wsgi/archive/refs/tags/${MOD_WSGI_VERSION}.tar.gz
RUN cd /usr/local/src \
    && tar -xf ${MOD_WSGI_VERSION}.tar.gz \
    && cd mod_wsgi-${MOD_WSGI_VERSION} \
    && ./configure --prefix=/usr/local --with-python=/opt/conda/bin/python \
    && make -j 8 \
    && make install \
    && cp /usr/lib/apache2/modules/mod_wsgi.so /usr/local/apache2/modules/mod_wsgi.so

# Clean up conda packages
RUN conda clean -y --all

# Enable ssh
# https://web.archive.org/web/20150215051915/https://docs.docker.com/examples/running_ssh_service/
RUN mkdir /var/run/sshd \
    && echo 'root:${ROOT_PASS}' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd


WORKDIR /app
RUN ldconfig

EXPOSE 22 8000
CMD ["/usr/sbin/sshd", "-D"]

# =============================================================================
