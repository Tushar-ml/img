# img



pip install tez efficientnet_pytorch albumentations

pip install albumentations
pip install open

pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html

pip install nfnets-pytorch




#####################################################################
Package                Version                                                                                                                                                                
---------------------- -------------------                                                                                                                                                                 
accelerate             0.3.0                                                                                                                                                                              
albumentations         1.0.0                                                                                                                                                                               
certifi                2021.5.30                                                                                                                                                                           
cycler                 0.10.0                                                                                                                                                                              
dataclasses            0.8                                                                                                                                                                                 
decorator              4.4.2                                                                                                                                                                               
deepspeed              0.3.16                                                                                                                                                                              
efficientnet-pytorch   0.7.1                                                                                                                                                                               
fire                   0.4.0                                                                                                                                                                               
imageio                2.9.0                                                                                                                                                                               
joblib                 1.0.1                                                                                                                                                                               
kiwisolver             1.3.1                                                                                                                                                                               
matplotlib             3.3.4                                                                                                                                                                               
networkx               2.5.1                                                                                                                                                                               
nfnets-pytorch         0.1.3                                                                                                                                                                               
ninja                  1.10.0.post2                                                                                                                                                                        
numpy                  1.19.5                                                                                                                                                                              
opencv-python-headless 4.5.2.54                                                                                                                                                                            
pandas                 1.1.5                                                                                                                                                                               
Pillow                 8.2.0                                                                                                                                                                               
pip                    21.1.2                                                                                                                                                                              
protobuf               3.17.3                                                                                                                                                                              
psutil                 5.8.0                                                                                                                                                                               
pyaml                  20.4.0                                                                                                                                                                              
pyparsing              2.4.7                                                                                                                                                                               
python-dateutil        2.8.1                                                                                                                                                                               
pytz                   2021.1                                                                                                                                                                              
PyWavelets             1.1.1                                                                                                                                                                               
PyYAML                 5.4.1                                                                                                                                                                               
scikit-image           0.17.2                                                                                                                                                                              
scikit-learn           0.24.2                                                                                                                                                                              
scipy                  1.5.4                                                                                                                                                                               
setuptools             49.6.0.post20210108                                                                                                                                                                 
six                    1.16.0                                                                                                                                                                              
tensorboardX           1.8                                                                                                                                                                                 
termcolor              1.1.0                                                                                                                                                                               
tez                    0.1.4                                                                                                                                                                               
threadpoolctl          2.1.0                                                                                                                                                                               
tifffile               2020.9.3                                                                                                                                                                            
torch                  1.7.1+cu101                                                                                                                                                                         
torchvision            0.8.2+cu101                                                                                                                                                                         
tqdm                   4.61.1                                                                                                                                                                              
typing-extensions      3.10.0.0                                                                                                                                                                            
wheel                  0.36.2                                                                                                                                                                             
wincertstore           0.2 



https://nlp123456.s3.ap-northeast-1.amazonaws.com/ztmp.tar.gz.001






+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.27.04    Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   46C    P8    10W /  70W |      0MiB / 15079MiB |      0%      Default |
|                               |                      |                 ERR! |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+





Collecting tez
  Downloading https://files.pythonhosted.org/packages/cb/12/9f9206ab4daa0a263be3ae30ba4fde162e8c3c0f0855b7f93644cd6eb6af/tez-0.0.8-py3-none-any.whl
Requirement already satisfied: torch>=1.6.0 in /usr/local/lib/python3.6/dist-packages (from tez) (1.7.0+cu101)
Requirement already satisfied: dataclasses in /usr/local/lib/python3.6/dist-packages (from torch>=1.6.0->tez) (0.8)
Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from torch>=1.6.0->tez) (0.16.0)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.6/dist-packages (from torch>=1.6.0->tez) (3.7.4.3)
Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from torch>=1.6.0->tez) (1.19.5)
Installing collected packages: tez
Successfully installed tez-0.0.8
Collecting efficientnet_pytorch
  Downloading https://files.pythonhosted.org/packages/4e/83/f9c5f44060f996279e474185ebcbd8dbd91179593bffb9abe3afa55d085b/efficientnet_pytorch-0.7.0.tar.gz
Requirement already satisfied: torch in /usr/local/lib/python3.6/dist-packages (from efficientnet_pytorch) (1.7.0+cu101)
Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from torch->efficientnet_pytorch) (1.19.5)
Requirement already satisfied: dataclasses in /usr/local/lib/python3.6/dist-packages (from torch->efficientnet_pytorch) (0.8)
Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from torch->efficientnet_pytorch) (0.16.0)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.6/dist-packages (from torch->efficientnet_pytorch) (3.7.4.3)
Building wheels for collected packages: efficientnet-pytorch
  Building wheel for efficientnet-pytorch (setup.py) ... done
  Created wheel for efficientnet-pytorch: filename=efficientnet_pytorch-0.7.0-cp36-none-any.whl size=16032 sha256=8bb1ce210dd9c23fa3be67beb8b795cab8a520d494695081fcdd842a6ed8f051
  Stored in directory: /root/.cache/pip/wheels/e9/c6/e1/7a808b26406239712cfce4b5ceeb67d9513ae32aa4b31445c6
Successfully built efficientnet-pytorch
Installing collected packages: efficientnet-pytorch
Successfully installed efficientnet-pytorch-0.7.0
Uninstalling albumentations-0.1.12:
  Would remove:
    /usr/local/lib/python3.6/dist-packages/albumentations-0.1.12.dist-info/*
    /usr/local/lib/python3.6/dist-packages/albumentations/*
Proceed (y/n)? y
  Successfully uninstalled albumentations-0.1.12
Collecting albumentations
  Downloading https://files.pythonhosted.org/packages/03/58/63fb1d742dc42d9ba2800ea741de1f2bc6bb05548d8724aa84794042eaf2/albumentations-0.5.2-py3-none-any.whl (72kB)
     |████████████████████████████████| 81kB 8.1MB/s 
Collecting opencv-python-headless>=4.1.1
  Downloading https://files.pythonhosted.org/packages/96/fc/4da675cc522a749ebbcf85c5a63fba844b2d44c87e6f24e3fdb147df3270/opencv_python_headless-4.5.1.48-cp36-cp36m-manylinux2014_x86_64.whl (37.6MB)
     |████████████████████████████████| 37.6MB 82kB/s 
Requirement already satisfied: scikit-image>=0.16.1 in /usr/local/lib/python3.6/dist-packages (from albumentations) (0.16.2)
Requirement already satisfied: PyYAML in /usr/local/lib/python3.6/dist-packages (from albumentations) (3.13)
Collecting imgaug>=0.4.0
  Downloading https://files.pythonhosted.org/packages/66/b1/af3142c4a85cba6da9f4ebb5ff4e21e2616309552caca5e8acefe9840622/imgaug-0.4.0-py2.py3-none-any.whl (948kB)
     |████████████████████████████████| 952kB 42.1MB/s 
Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from albumentations) (1.4.1)
Requirement already satisfied: numpy>=1.11.1 in /usr/local/lib/python3.6/dist-packages (from albumentations) (1.19.5)
Requirement already satisfied: imageio>=2.3.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image>=0.16.1->albumentations) (2.4.1)
Requirement already satisfied: matplotlib!=3.0.0,>=2.0.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image>=0.16.1->albumentations) (3.2.2)
Requirement already satisfied: PyWavelets>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image>=0.16.1->albumentations) (1.1.1)
Requirement already satisfied: networkx>=2.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image>=0.16.1->albumentations) (2.5)
Requirement already satisfied: pillow>=4.3.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image>=0.16.1->albumentations) (7.0.0)
Requirement already satisfied: Shapely in /usr/local/lib/python3.6/dist-packages (from imgaug>=0.4.0->albumentations) (1.7.1)
Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from imgaug>=0.4.0->albumentations) (1.15.0)
Requirement already satisfied: opencv-python in /usr/local/lib/python3.6/dist-packages (from imgaug>=0.4.0->albumentations) (4.1.2.30)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.16.1->albumentations) (1.3.1)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.16.1->albumentations) (2.4.7)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.16.1->albumentations) (0.10.0)
Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.16.1->albumentations) (2.8.1)
Requirement already satisfied: decorator>=4.3.0 in /usr/local/lib/python3.6/dist-packages (from networkx>=2.0->scikit-image>=0.16.1->albumentations) (4.4.2)
Installing collected packages: opencv-python-headless, imgaug, albumentations
  Found existing installation: imgaug 0.2.9
    Uninstalling imgaug-0.2.9:
      Successfully uninstalled imgaug-0.2.9
Successfully installed albumentations-0.5.2 imgaug-0.4.0 opencv-python-headless-4.5.1.48




windows
Wed Jun 23 00:31:39 2021                                                                                                                                       
+-----------------------------------------------------------------------------+                                                                                
| NVIDIA-SMI 419.71       Driver Version: 419.71       CUDA Version: 10.0     |                                                                                
|-------------------------------+----------------------+----------------------+                                                                                
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |                                                                                
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |                                                                                
|===============================+======================+======================|                                                              
|   0  GeForce GTX 1650   WDDM  | 00000000:01:00.0 Off |                  N/A |                                                              
| N/A   68C    P8     3W /  N/A |    132MiB /  4096MiB |      0%      Default |                                                              
+-------------------------------+----------------------+----------------------+                                                              
                                                                                                                                             
+-----------------------------------------------------------------------------+                                                              
| Processes:                                                       GPU Memory |                                                              
|  GPU       PID   Type   Process name                             Usage      |                                                              
|=============================================================================|                                                              
|  No running processes found                                                 |                                                              
+-----------------------------------------------------------------------------+                                                              
WARNING: infoROM is corrupted at gpu 0000:01:00.0  



