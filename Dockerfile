FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN apt-get update && apt-get -y install cmake

# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

WORKDIR /binary-tensor
COPY src/ ./src/
COPY test/ ./test/
COPY CMakeLists.txt ./
COPY Config.cmake.in ./
WORKDIR /binary-tensor

WORKDIR /binary-tensor/build

RUN cmake ..
RUN make install

WORKDIR /binary-tensor
