<div align="center">

# Emotion recognizer
</div>

#### Table of contents
1. [Clone project](#1-clone-project)
2. [Install dependencies](#2-install-dependencies)   
3. [Main program](#3-run-main-program)

#### 1. Clone project
```bash
git clone https://github.com/GDSC2021/emotion-recognizer 
```

#### 2. Install dependencies
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
#### 3. Run main program
```bash
python src/face-bounding-box.py  #For Windows
python3 src/face-bounding-box.py #For macOS and Linux
```
