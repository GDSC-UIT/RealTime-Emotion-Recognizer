<div align="center">

# Emotion recognizer
</div>

#### Table of contents
1. [Clone project](#1-clone-project)
2. [Dataset](#2-dataset)
3. [Install dependencies](#3-install-dependencies)   
4. [Methods](#4-methods)
5. [Main program](#5-run-main-program)


#### 1. Clone project
```bash
git clone https://github.com/GDSC2021/emotion-recognizer 
```

#### 2. Dataset
- Download [here](https://drive.google.com/file/d/15s1hy8_7QBcX-RiSELkKMXz9BOGVK3Dr/view?usp=sharing)

#### 3. Install dependencies
##### a. Download and install Anaconda:
- Download latest version of [Python](https://www.python.org/downloads/) and [pip](https://pip.pypa.io/en/stable/installation/)
- Download [Anaconda package](https://www.anaconda.com/products/individual) 
- Installation:
    - For [macOS](https://docs.anaconda.com/anaconda/install/mac-os/)
    - For [Windows](https://docs.anaconda.com/anaconda/install/windows/)
    - For [Linux](https://docs.anaconda.com/anaconda/install/linux/)
##### b. Create environment
```bash
conda init
conda create -n emotion-detect python=3.8 -y
conda activate emotion-detect
```
##### c. Install prerequisites
- Open Terminal (macOS and Linux) or Anaconda Prompt (Windows)
- Go to your cloned directory
- Run:
    - For Windows
        ```bash
        sh dependencies/win/win.sh
        ```
    - For macOS
        ```bash
        chmod +x dependencies/mac/dependencies.sh && ./dependencies/mac/dependencies.sh
        ```
    - For Linux
        ```bash
        chmod +x dependencies/linux/linux.sh && ./dependencies/linux/linux.sh
        ```

#### 4. Methods
| Architecture name | Structures | Accuracy |
| :--: | :--: | :--: |
| VGGFace upgraded | The same as VGGFace structure but with 2 extra Convolutional layers and 1 Pooling before Flatten  | |

#### 5. Run main program
```bash
python src/main.py  #For Windows
python3 src/main.py #For macOS and Linux
```