<div align="center">

# Emotion recognizer
</div>

#### Table of contents
1. [Clone project](#1-clone-project)
2. [Dataset](#2-dataset)
3. [Install dependencies](#3-install-dependencies)   
4. [Main program](#4-run-main-program)
5. [Demo](#5-demo)

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
- Run program:
```bash
python src/main.py  #For Windows
python3 src/main.py #For macOS and Linux
```
- To abort program, press the combination of `Ctrl + C`
#### 5. Demo
<div align='center'>
    
<img width="626" alt="Screen Shot 2021-09-19 at 11 04 25" src="https://user-images.githubusercontent.com/67086934/133915043-37d1631c-244a-4687-986c-0798c990ec35.png">
    
</div>
