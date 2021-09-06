<div align="center">

# Emotion recognizer
</div>

#### I. Install dependencies
##### 1. Download and install Anaconda:
- [Download package](https://www.anaconda.com/products/individual) 
- Installation:
    - For [macOS](https://docs.anaconda.com/anaconda/install/mac-os/)
    - For [Windows](https://docs.anaconda.com/anaconda/install/windows/)
    - For [Linux](https://docs.anaconda.com/anaconda/install/linux/)
##### 2. Create environment
```bash
conda create -n emotion-detect python=3.8 -y
conda activate emotion-detect
```
##### 3. Install prerequisites
- For Windows
```bash
chmod +x dependencies/win/win.sh && ./dependencies/win/win.sh
```
- For macOS
```bash
chmod +x dependencies/mac/dependencies.sh && ./dependencies/mac/dependencies.sh
```
- For Linux
```bash
chmod +x dependencies/linux/linux.sh && ./dependencies/linux/linux.sh
```
#### II. Run main program
```bash
python3 src/face-bounding-box.py
```
