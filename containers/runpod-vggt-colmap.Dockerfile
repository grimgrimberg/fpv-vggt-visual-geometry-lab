# syntax=docker/dockerfile:1.7

# Template for the expanded inference pod image.
#
# Production rule:
# - Build this once.
# - Push it to a registry.
# - Run the job by image digest, not by a mutable tag.
# - Set VGGT_REF to a commit SHA, not "main", for the paid production run.
#
# The default base is intentionally a build argument so it can be replaced by a
# known-good RunPod/PyTorch image if your pod provider has one preloaded.
ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG VGGT_REF=main
ARG DUST3R_REF=main
ARG MAST3R_REF=main
ARG OPENMVS_REF=develop
ARG INSTALL_OPENMVS=false
ARG INSTALL_SHOWCASE_TOOLS=false
ARG INSTALL_RESEARCH_METHODS=false
ARG COMPILE_RESEARCH_CUDA_KERNELS=false

ENV VGGT_REPO_DIR=/workspace/vggt
ENV DUST3R_REPO_DIR=/workspace/dust3r
ENV MAST3R_REPO_DIR=/workspace/mast3r
ENV OPENMVS_REPO_DIR=/workspace/openMVS
ENV PYTHONPATH=/workspace/vggt:/workspace/dust3r:/workspace/mast3r:${PYTHONPATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    colmap \
    ffmpeg \
    git \
    libboost-iostreams-dev \
    libboost-program-options-dev \
    libboost-serialization-dev \
    libboost-system-dev \
    libcgal-dev \
    libceres-dev \
    libeigen3-dev \
    libglew-dev \
    libgl1 \
    libglib2.0-0 \
    libglfw3-dev \
    libjpeg-dev \
    libopencv-dev \
    libpng-dev \
    libsqlite3-dev \
    libtiff-dev \
    ninja-build \
    unzip \
    wget \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# OpenMVS is the recommended dense-mesh alternative to ODM for this project.
# Keep it in the pinned image rather than compiling it during an expensive run.
RUN if [ "${INSTALL_OPENMVS}" = "true" ]; then \
      git clone --recursive https://github.com/cdcseacave/openMVS.git "${OPENMVS_REPO_DIR}" \
      && cd "${OPENMVS_REPO_DIR}" \
      && git checkout "${OPENMVS_REF}" \
      && git submodule update --init --recursive \
      && git rev-parse HEAD > /opt_openmvs_commit.txt \
      && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DOpenMVS_USE_CUDA=ON \
      && cmake --build build --parallel "$(nproc)" \
      && cmake --install build \
      && ldconfig \
      && InterfaceCOLMAP -h >/dev/null \
      && DensifyPointCloud -h >/dev/null \
      && ReconstructMesh -h >/dev/null; \
    fi

RUN git clone https://github.com/facebookresearch/vggt.git /workspace/vggt \
    && cd /workspace/vggt \
    && git checkout "${VGGT_REF}" \
    && git rev-parse HEAD > /opt_vggt_commit.txt \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir -r requirements_demo.txt \
    && python -m pip install --no-cache-dir pycolmap trimesh evo


# DUSt3R and MASt3R are optional research baselines. Their official setup
# uses recursive clones and requirements files; enabling this build arg makes
# `research_methods_ready` true when imports succeed. Keep refs pinned for paid
# production runs.
RUN if [ "${INSTALL_RESEARCH_METHODS}" = "true" ]; then \
      git clone --recursive https://github.com/naver/dust3r.git "${DUST3R_REPO_DIR}" \
      && cd "${DUST3R_REPO_DIR}" \
      && git checkout "${DUST3R_REF}" \
      && git submodule update --init --recursive \
      && git rev-parse HEAD > /opt_dust3r_commit.txt \
      && python -m pip install --no-cache-dir -r requirements.txt \
      && python -m pip install --no-cache-dir -r requirements_optional.txt \
      && git clone --recursive https://github.com/naver/mast3r.git "${MAST3R_REPO_DIR}" \
      && cd "${MAST3R_REPO_DIR}" \
      && git checkout "${MAST3R_REF}" \
      && git submodule update --init --recursive \
      && git rev-parse HEAD > /opt_mast3r_commit.txt \
      && python -m pip install --no-cache-dir -r requirements.txt \
      && python -m pip install --no-cache-dir -r dust3r/requirements.txt \
      && python -m pip install --no-cache-dir -r dust3r/requirements_optional.txt \
      && python -m pip install --no-cache-dir cython; \
    fi

RUN if [ "${INSTALL_RESEARCH_METHODS}" = "true" ] && [ "${COMPILE_RESEARCH_CUDA_KERNELS}" = "true" ]; then \
      cd "${DUST3R_REPO_DIR}/croco/models/curope" \
      && python setup.py build_ext --inplace \
      && cd "${MAST3R_REPO_DIR}/dust3r/croco/models/curope" \
      && python setup.py build_ext --inplace; \
    fi

# Showcase tools are optional and can make image builds slower or more fragile.
# Enable only for a separate visual-showcase image or after the VGGT+COLMAP
# package is already proven.
RUN if [ "${INSTALL_SHOWCASE_TOOLS}" = "true" ]; then \
      python -m pip install --no-cache-dir nerfstudio gsplat; \
    fi

WORKDIR /workspace

CMD ["bash"]
