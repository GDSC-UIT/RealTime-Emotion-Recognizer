<div align="center">

# Emotion recognizer
</div>

#### Table of contents
1. [Clone project](#1-clone-project)
2. [Dataset](#2-dataset)
3. [Install dependencies](#3-install-dependencies)   
4. [Main program](#4-run-main-program)

#### 1. Clone project
```bash
git clone https://github.com/GDSC2021/emotion-recognizer 
```

#### 2. Dataset 
- Download [here](https://drive.google.com/file/d/15s1hy8_7QBcX-RiSELkKMXz9BOGVK3Dr/view?usp=sharing)

#### 3. Install dependencies
##### a. Download and install Anaconda:
- Download latest version of [Python](https://www.python.org/downloads/)
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
        pip install matplotlib pandas pillow autopep8 sklearn tensorflow keras opencv-python seaborn
        ```
    - For macOS
        ```bash
        chmod +x dependencies/mac/dependencies.sh && ./dependencies/mac/dependencies.sh
        ```
    - For Linux
        ```bash
        chmod +x dependencies/linux/linux.sh && ./dependencies/linux/linux.sh
        ```

#### 4. Run main program
- Create a folder named `model` on your local
- Download 2 following files and put them into your `model` folder
    - [![](https://img.shields.io/badge/vggface.h5-blue.svg?longCache=true&style=flat&logo=google-drive)](https://drive.google.com/drive/folders/1VruHPA0WRbPMo8vVFe9TTt6d-Pg4vmWo?usp=sharing) 
    - [![](https://img.shields.io/badge/vggfacemodel.json-blue.svg?longCache=true&style=flat&logo=google-drive)](https://drive.google.com/file/d/1mXYUAnXZDz5jDLD4zVycgohW4SFbnN5e/view?usp=sharing) 
- Run program:
```bash
python src/main.py  #For Windows
python3 src/main.py #For macOS and Linux
```
- To abort program, press the combination of `Ctrl + C`