# Google Cloud Installation


## Reinstall GPU driver and CUDA9
```bash
sudo add-apt-repository ppa:graphics-drivers
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
# installs both NVIDIA driver and CUDA
sudo apt-get install cuda-9.0

# [OPTIONAL] reset & verify installation
sudo reboot
lsmod | grep nvidia
nvidia-smi
```

### [OPTIONAL] Remove GPU drivers
When above step fails, try below removing previous installations and retry.

```bash
sudo apt-get purge nvidia*
sudo apt-get purge cuda*
```

## Install CuDNN 7 
based on [this link](http://goodtogreate.tistory.com/entry/TensorFlow-GPU-버전-우분투-1604에-설치-하기)

```bash
# Copy libcudnn7* files from nvidia
cd /tmp
sudo dpkg -i libcudnn7_*cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev*cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-doc*cuda9.0_amd64.deb

sudo apt-get install libcupti-dev
```

###set up environment vars

```bash
vi ~/.bashrc
> export PATH=/usr/local/cuda/bin${PATH}
> export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH}
> export CUDA_HOME=/usr/local/cuda
```

###[OPTIONAL] verify installation

```bash
cp -r /usr/src/cudnn_samples_* /tmp
cd /tmp/cudnn_samples_v7/mnistCUDNN
make clean && make
./mnistCUDNN
```

## Setup virtualenv
```bash
git clone https://github.com/DeepLearningCollege/intQA.git
cd ~/intQA
virtualenv -p python3 envIntQA
source envIntQA/bin/activate
```

## Install tensorflow1.4 
```
pip install tensorflow-gpu==1.4
```

### upgrade tensorflow1.4 for for CUDA9 and CuDNN7
per [this link](https://github.com/mind/wheels/releases/tag/tf1.4-gpu-cuda9)

```bash
cd ~/intQA
mkdir ext
cd ext
sudo apt install cmake
git clone https://github.com/01org/mkl-dnn.git
cd mkl-dnn/scripts && ./prepare_mkl.sh && cd ..
mkdir -p build && cd build && cmake .. && make
sudo make install
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> ~/intQA/envIntQA/bin/activate

pip install --ignore-installed --upgrade https://github.com/mind/wheels/releases/download/tf1.4-gpu-cuda9/tensorflow-1.4.0-cp35-cp35m-linux_x86_64.whl
```

## Install intQA dependencies

```bash
# spacy
pip install -U spacy
python -m spacy download en
# pytorch
pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl 
pip3 install torchvision
# cove
mkdir ext
cd ext
git clone https://github.com/salesforce/cove.git
cd cove
pip install -r requirements.txt
python setup.py develop
# etc
pip install boto3
# setup package
```