FROM dustynv/l4t-pytorch:r36.2.0 AS l4t_pytorch
ARG DEBIAN_FRONTEND noninteractive

# Install cuRobo
RUN apt update \
    && apt install -y git-lfs \
    && git clone https://github.com/NVlabs/curobo.git \
    && cd curobo \
    && git-lfs pull * && git-lfs pull .

# RUN pip install --upgrade setuptools setuptools_scm
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://pypi.org/simple"
RUN pip3 install pybind11 scikit_build_core

ARG PLACE_NET_CUDA_ARCH
ENV PIP_TRUSTED_HOST="https://pypi.ngc.nvidia.com"
ENV PIP_INDEX_URL="https://pypi.ngc.nvidia.com"
RUN cd curobo \
    && export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;${PLACE_NET_CUDA_ARCH}" \
    && export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST%;}" \
    && export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST#;}" \
    && pip install -e . --no-build-isolation

