FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# Add Miniconda to PATH
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# Install Miniconda
RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz openmpi-bin && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh

# Create new user
ENV NB_USER atari.robot
ENV NB_UID 1000

# Create folders and give ownership to new user
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown $NB_USER $CONDA_DIR -R && \
    mkdir -p /src && \
    chown $NB_USER /src

# Don't change user, because otherwise it can't write to host volumes
# USER $NB_USER

# Change working directory
WORKDIR /src

# Install Tensorflow and Keras
RUN conda install -y python=3.5 && \
    pip install --upgrade pip && \
    pip install tensorflow-gpu && \
    conda install h5py && \
    pip install keras && \
    conda clean -yt

# Install OpenCV and PyGame
RUN apt-get update && \
    apt-get install -y libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake && \
    pip install opencv-contrib-python && \
    pip install pygame

# Clone modified ALE and install it
RUN git clone https://github.com/renattou/Arcade-Learning-Environment.git ALE && \
    cd ALE && git checkout pong-2p && mkdir build && cd build && \
    cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .. && \
    make -j 4 && cd .. && \
    pip install . --user && \
    cd .. && rm -rf ALE

# Download V-REP
RUN wget --quiet http://coppeliarobotics.com/files/V-REP_PRO_EDU_V3_4_0_Linux.tar.gz && \
    tar -zxf V-REP_PRO_EDU_V3_4_0_Linux.tar.gz && \
    mv V-REP_PRO_EDU_V3_4_0_Linux vrep && \
    rm V-REP_PRO_EDU_V3_4_0_Linux.tar.gz

# Copy modified lib with shared memory
COPY libv_repExtRemoteApi.so /src/vrep

# Some environment variables
ENV PYTHONPATH='/src/:$PYTHONPATH'
ENV KERAS_BACKEND tensorflow
