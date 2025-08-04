# syntax = docker/dockerfile:1

# BASE_IMAGE is the image that will be extended by this Dockerfile.
# It is assumed to a well configured Python installation.
# The reminaing packages will be installed with pip.
ARG BASE_IMAGE
ARG GIT_IMAGE

########################################################################
# Install apt packages.

FROM ${BASE_IMAGE} AS runtime-apt-pkgs

# A directory to record all the dependency files used at multiple stages.
# This is useful for a later inspection or debugging.
ENV DEPENDENCIES_DIR=/opt/template-dependencies
RUN mkdir ${DEPENDENCIES_DIR}
COPY apt.txt ${DEPENDENCIES_DIR}/apt.txt

# Enable caching for `apt` packages in Docker.
# https://docs.docker.com/engine/reference/builder/#run---mounttypecache
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > \
    /etc/apt/apt.conf.d/keep-cache

ARG DEBIAN_FRONTEND=noninteractive

# sed is only used as a hack to remove comments from the file apt.txt.
#RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
#    --mount=type=cache,target=/var/lib/apt,sharing=private \
#    apt update && \
#    sed -e 's/#.*//g' -e 's/\r//g' ${DEPENDENCIES_DIR}/apt.txt | \
#    xargs -t apt-get install -y --no-install-recommends && \
#    rm -rf /var/lib/apt/lists/*

# Podman: Comment the above and use this instead with podman as it doesn't support sharing mount modes.
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt update && \
    sed -e 's/#.*//g' -e 's/\r//g' ${DEPENDENCIES_DIR}/apt.txt | \
    xargs -t apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

########################################################################
# Install dependencies.

FROM runtime-apt-pkgs AS runtime-deps

# RUN echo "BULDING FOR SAGE MATH"
# RUN echo "Etc/UTC" > /etc/timezone && \
#     apt-get update && apt-get upgrade -y && apt-get install -y --fix-missing --no-install-recommends \
#     sagemath

# Install pip packages.
ENV PIP_CACHE_DIR=/root/.cache/pip
RUN pip freeze > ${DEPENDENCIES_DIR}/requirements-freeze-before-pip-install.txt
RUN pip list --format freeze > ${DEPENDENCIES_DIR}/requirements-list-before-pip-install.txt
COPY requirements.txt ${DEPENDENCIES_DIR}/requirements.txt
RUN --mount=type=cache,target=${PIP_CACHE_DIR} \
    pip install -r ${DEPENDENCIES_DIR}/requirements.txt
# Podman: Comment the above and use this instead with podman as it doesn't support sharing mount modes.
#RUN --mount=type=cache,target=${PIP_CACHE_DIR} \
#    pip install -r ${DEPENDENCIES_DIR}/requirements.txt
RUN pip freeze > ${DEPENDENCIES_DIR}/requirements-freeze-after-pip-install.txt
RUN pip list --format freeze > ${DEPENDENCIES_DIR}/requirements-list-after-pip-install.txt


# For reproducible requirements use the following after getting the requirements-freeze.txt file from the first build.
#COPY requirements-freeze.txt ${DEPENDENCIES_DIR}/requirements-freeze.txt
#RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=private \
#    pip install --no-deps -r ${DEPENDENCIES_DIR}/requirements-freeze.txt
# For podman
#RUN --mount=type=cache,target=${PIP_CACHE_DIR} \
#    pip install --no-deps -r ${DEPENDENCIES_DIR}/requirements-freeze.txt

########################################################################
#ADDED LATER

RUN apt update && apt install -y openssh-client openssh-server sagemath

# Install SageMath from source, following https://askubuntu.com/questions/1516582/how-to-install-sagemath-in-ubuntu-in-24-04
# and https://sagemanifolds.obspm.fr/install_ubuntu.html

#RUN apt-get update && apt-get install -y automake bc binutils bzip2 ca-certificates cliquer cmake curl ecl eclib-tools fflas-ffpack flintqs g++ gengetopt gfan gfortran git glpk-utils gmp-ecm lcalc libatomic-ops-dev libboost-dev libbraiding-dev libbz2-dev libcdd-dev libcdd-tools libcliquer-dev libcurl4-openssl-dev libec-dev libecm-dev libffi-dev libflint-dev libfreetype-dev libgc-dev libgd-dev libgf2x-dev libgiac-dev libgivaro-dev libglpk-dev libgmp-dev libgsl-dev libhomfly-dev libiml-dev liblfunction-dev liblrcalc-dev liblzma-dev libm4rie-dev libmpc-dev libmpfi-dev libmpfr-dev libncurses-dev libntl-dev libopenblas-dev libpari-dev libpcre3-dev libplanarity-dev libppl-dev libprimesieve-dev libpython3-dev libqhull-dev libreadline-dev librw-dev libsingular4-dev libsqlite3-dev libssl-dev libsuitesparse-dev libsymmetrica2-dev zlib1g-dev libzmq3-dev libzn-poly-dev m4 make nauty openssl palp pari-doc pari-elldata pari-galdata pari-galpol pari-gp2c pari-seadata patch perl pkg-config planarity ppl-dev python3-setuptools python3-venv r-base-dev r-cran-lattice singular sqlite3 sympow tachyon tar tox xcas xz-utils

#RUN git clone --branch 10.4 https://github.com/sagemath/sage.git sage
#RUN cd sage && make configure
#RUN cd sage && ./configure --enable-build-as-root
#RUN cd sage && MAKE="make -j128" make

########################################################################

#
# Optional optimizations.
# Hack to enable Intel MKL optimizations on AMD CPUs.
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
ARG FAKEINTEL_PATH=/opt/fakeintel/libfakeintel.so
ENV FAKEINTEL_PATH=${FAKEINTEL_PATH}
# https://danieldk.eu/Posts/2020-08-31-MKL-Zen.html
# Build.
RUN echo 'int mkl_serv_intel_cpu_true() {return 1;}' > /tmp/fakeintel.c && \
    mkdir -p /opt/fakeintel && \
    gcc -shared -fPIC -o ${FAKEINTEL_PATH} /tmp/fakeintel.c
# Enable.
ENV LD_PRELOAD=${FAKEINTEL_PATH}:${LD_PRELOAD}

########################################################################
# Here you can install other software


########################################################################
# Download Z-Shell enhancements.

FROM ${GIT_IMAGE} AS get-pure

ARG PURE_URL=https://github.com/sindresorhus/pure.git
ARG ZSHA_URL=https://github.com/zsh-users/zsh-autosuggestions.git
ARG ZSHS_URL=https://github.com/zsh-users/zsh-syntax-highlighting.git

RUN git clone --depth 1 "${PURE_URL}" /opt/zsh/pure
RUN git clone --depth 1 "${ZSHA_URL}" /opt/zsh/zsh-autosuggestions
RUN git clone --depth 1 "${ZSHS_URL}" /opt/zsh/zsh-syntax-highlighting

########################################################################
# This stage is the final user-agnostic (generic) stage.
# This layer can be distributed so that subsequent users

FROM runtime-deps AS runtime-generic

ARG PROJECT_NAME
ARG PACKAGE_NAME
ENV PROJECT_NAME=${PROJECT_NAME}
ENV PACKAGE_NAME=${PACKAGE_NAME}
ENV HYDRA_FULL_ERROR=1

# A final record of the dependencies from pip freeze.
RUN pip freeze > ${DEPENDENCIES_DIR}/requirements-freeze-final.txt
RUN pip list --format freeze > ${DEPENDENCIES_DIR}/requirements-list-final.txt

# Shell configuration.
ENV ZSH_ENHANCE_DIR=/etc/zsh/enhance
ARG PURE_PATH=${ZSH_ENHANCE_DIR}/pure
ARG ZSHA_PATH=${ZSH_ENHANCE_DIR}/zsh-autosuggestions
ARG ZSHS_PATH=${ZSH_ENHANCE_DIR}/zsh-syntax-highlighting
COPY --from=get-pure /opt/zsh/pure ${PURE_PATH}
COPY --from=get-pure /opt/zsh/zsh-autosuggestions ${ZSHA_PATH}
COPY --from=get-pure /opt/zsh/zsh-syntax-highlighting ${ZSHS_PATH}
RUN {   echo "fpath+=${PURE_PATH}"; \
        echo "autoload -Uz promptinit; promptinit"; \
        echo "prompt pure"; \
        echo "source ${ZSHA_PATH}/zsh-autosuggestions.zsh"; \
        echo "source ${ZSHS_PATH}/zsh-syntax-highlighting.zsh"; \
        echo "alias ls='ls --color=auto'"; \
        echo "alias ll='ls -lh'"; \
        echo "alias update-env-file='source \${PROJECT_ROOT_AT}/installation/docker-amd64-cuda/update-env-file.sh'"; \
    } >> /etc/zsh/zshrc


# Entrypoints.
# Don't overwrite the entrypoint, it is installing the project
# and testing that you correctly mounted the project code.
# It also performs some other important setup depending on the deployment platform.
ARG BASE_ENTRYPOINT
ARG BASE_ENTRYPOINT_EXECS
ENV BASE_ENTRYPOINT=${BASE_ENTRYPOINT}
ENV BASE_ENTRYPOINT_EXECS=${BASE_ENTRYPOINT_EXECS}
ENV ENTRYPOINTS_ROOT=/opt/template-entrypoints
COPY entrypoints ${ENTRYPOINTS_ROOT}

RUN pip install opencv-python-headless==4.5.4.58

ENTRYPOINT ["/opt/template-entrypoints/pre-entrypoint.sh"]
CMD ["/bin/zsh"]

