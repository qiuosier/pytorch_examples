FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime AS base

# Create working directory
ENV HOME /etc/datascience
RUN mkdir -p ${HOME}
WORKDIR ${HOME}

# Create sync directory and expose as environment variable
RUN mkdir /opt/ml
ENV OCI__SYNC_DIR=/opt/ml

RUN apt-get update && \
    apt-get install -y curl

#NVIDIA Nsight Systems 2022.1.1
RUN  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         apt-transport-https \
         ca-certificates \
         gnupg && \
     rm -rf /var/lib/apt/lists/*
RUN curl -L https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/nvidia.pub | apt-key add - && \
     echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/ /" >> /etc/apt/sources.list.d/nsight.list && \
     apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         nsight-systems-2022.1.1 && \
     rm -rf /var/lib/apt/lists/*

RUN curl -Lo /tmp/oci-cli-installer.sh https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.sh && \
    chmod +x /tmp/*-installer.sh && \
    bash -c "/tmp/oci-cli-installer.sh --accept-all-defaults" && \
    rm -rf /tmp/*-installer.sh /tmp/oci_cli_install_*

# Setup
ARG ARTIFACT_DIR=oci_dist_training_artifacts/pytorch/v1_metrics
ARG CONDA_ENV_FILE=environments.yaml

# Copy artifacts
COPY ${ARTIFACT_DIR} ${HOME}
# Install dependencies
RUN /opt/conda/bin/conda env update -n base --file ${CONDA_ENV_FILE}

# Copy user code
ARG CODE_DIR=.
RUN mkdir -p ${HOME}/code
COPY ${CODE_DIR} ${HOME}/code

RUN chmod u+x ${HOME}/run.sh
ENTRYPOINT ["/etc/datascience/run.sh"]
