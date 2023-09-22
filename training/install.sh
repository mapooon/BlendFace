pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install easydict mxnet onnx scikit-learn timm tensorboard scipy==1.8.1

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
apt-get install ffmpeg libsm6 libxext6  git -y
pip install opencv-python-headless==4.5.5.64