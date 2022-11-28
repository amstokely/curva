FROM nvidia/cuda:11.6.0-devel-ubuntu20.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR .
RUN apt-get -y update && apt-get install -y
RUN apt-get install wget -y
RUN apt-get install build-essential -y
RUN apt-get -y install g++ git -y
RUN mkdir /curva_cuda-11.6.0_python_3.9_ubuntu-20.04
WORKDIR ./curva_cuda-11.6.0_python_3.9_ubuntu-20.04
RUN mkdir -p ./miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -O /curva_cuda-11.6.0_python_3.9_ubuntu-20.04/miniconda3/miniconda.sh
RUN bash /curva_cuda-11.6.0_python_3.9_ubuntu-20.04/miniconda3/miniconda.sh -b -u -p /curva_cuda-11.6.0_python_3.9_ubuntu-20.04/miniconda3
RUN rm -rf /curva_cuda-11.6.0_python_3.9_ubuntu-20.04/miniconda3/miniconda.sh
RUN /curva_cuda-11.6.0_python_3.9_ubuntu-20.04/miniconda3/bin/conda init bash
RUN /curva_cuda-11.6.0_python_3.9_ubuntu-20.04/miniconda3/bin/conda init zsh
ENV PATH="/curva_cuda-11.6.0_python_3.9_ubuntu-20.04/miniconda3/bin:${PATH}"
RUN conda install -c conda-forge swig numpy cmake zlib boost
RUN conda install -c conda-forge vim
RUN mkdir ./curva
COPY . ./curva
WORKDIR ./curva
RUN mkdir ./build
WORKDIR ./build
RUN cmake .. -DCONDA_DIR=/curva_cuda-11.6.0_python_3.9_ubuntu-20.04/miniconda3
RUN make
WORKDIR ../../..
# Create a new stage for our output container
FROM nvidia/cuda:11.6.0-base-ubuntu20.04
WORKDIR /
RUN mkdir ./curva_cuda-11.6.0_python_3.9_ubuntu-20.04
WORKDIR ./curva_cuda-11.6.0_python_3.9_ubuntu-20.04
RUN mkdir ./miniconda3
ENV PATH="/curva_cuda-11.6.0_python_3.9_ubuntu-20.04/miniconda3/bin:${PATH}"
WORKDIR ./miniconda3
COPY --from=builder /curva_cuda-11.6.0_python_3.9_ubuntu-20.04/miniconda3 ./
WORKDIR ./lib/python3.9/site-packages
RUN mkdir ./pycurva
WORKDIR ./pycurva
COPY --from=builder /curva_cuda-11.6.0_python_3.9_ubuntu-20.04/curva/lib/pycurva.py ./
COPY --from=builder /curva_cuda-11.6.0_python_3.9_ubuntu-20.04/curva/lib/_pycurva.so ./
RUN echo "from .pycurva import *" > __init__.py
WORKDIR /curva_cuda-11.6.0_python_3.9_ubuntu-20.04
RUN mkdir ./curva
ENV LD_LIBRARY_PATH="/curva_cuda-11.6.0_python_3.9_ubuntu-20.04/curva:${LD_LIBRARY_PATH}"
WORKDIR ./curva
COPY --from=builder /curva_cuda-11.6.0_python_3.9_ubuntu-20.04/curva/lib/libcurva.so ./
COPY --from=builder /curva_cuda-11.6.0_python_3.9_ubuntu-20.04/curva/lib/tclcurva.so ./
WORKDIR /curva_cuda-11.6.0_python_3.9_ubuntu-20.04
RUN echo "export PATH=/curva_cuda-11.6.0_python_3.9_ubuntu-20.04/miniconda3/bin:${PATH}"
