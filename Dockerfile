# * Copyright (c) 2009-2020. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04
CMD nvidia-smi

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
# RUN --gpus all nvidia/cuda:11.4.0-base-ubuntu18.04 nvidia-smi

FROM cytomine/software-python3-base:v2.2.0
# Install Stardist and tensorflow
RUN pip3 install tensorflow-gpu==2.8.0
RUN pip3 install stardist==0.8.0
RUN pip3 install protobuf==3.20.*

#INSTALL
RUN pip3 install numpy==1.22.*
RUN pip3 install shapely
RUN pip3 install tifffile

WORKDIR /models
RUN mkdir -p 2D_versatile_HE
ADD /models/2D_versatile_HE/config.json /models/2D_versatile_HE/config.json
ADD /models/2D_versatile_HE/thresholds.json /models/2D_versatile_HE/thresholds.json
ADD /models/2D_versatile_HE/weights_best.h5 /models/2D_versatile_HE/weights_best.h5
RUN chmod 444 /models/2D_versatile_HE/config.json
RUN chmod 444 /models/2D_versatile_HE/thresholds.json
RUN chmod 444 /models/2D_versatile_HE/weights_best.h5

WORKDIR /models
RUN mkdir -p 2D_versatile_fluo
ADD /models/2D_versatile_fluo/config.json /models/2D_versatile_fluo/config.json
ADD /models/2D_versatile_fluo/thresholds.json /models/2D_versatile_fluo/thresholds.json
ADD /models/2D_versatile_fluo/weights_best.h5 /models/2D_versatile_fluo/weights_best.h5
RUN chmod 444 /models/2D_versatile_fluo/config.json
RUN chmod 444 /models/2D_versatile_fluo/thresholds.json
RUN chmod 444 /models/2D_versatile_fluo/weights_best.h5

WORKDIR /models
RUN mkdir -p 2D_versatile_fluo_sish
ADD /models/2D_versatile_fluo_sish/config.json /models/2D_versatile_fluo_sish/config.json
ADD /models/2D_versatile_fluo_sish/thresholds.json /models/2D_versatile_fluo_sish/thresholds.json
ADD /models/2D_versatile_fluo_sish/weights_best.h5 /models/2D_versatile_fluo_sish/weights_best.h5
RUN chmod 444 /models/2D_versatile_fluo_sish/config.json
RUN chmod 444 /models/2D_versatile_fluo_sish/thresholds.json
RUN chmod 444 /models/2D_versatile_fluo_sish/weights_best.h5

#ADD FILES
RUN mkdir -p /app
ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py

# Set environment variables for GPU support
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENTRYPOINT ["python3", "/app/run.py"]
