
# Deep Generative Model for Patient Specific Longitudinal Analysis

My PMSD project

## Requirements
* Tensorflow 1.15

## How to Run
* Open a Google Colab notebook. 
* Download and extract PASCAL VOC dataset. (Replace with actual dataset later)
```
! curl -O http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar  
print("Dataset downloaded.")
! tar -xf VOCtrainval_11-May-2012.tar
print("Dataset extracted.")
! rm VOCtrainval_11-May-2012.tar
print("'.tar' file removed.")
```
* To mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
* Clone this repository inside `/content` folder.
```
! cd /content && git clone https://github.com/umutkucukaslan/pmsd-project.git
```
* Run the module main
```
! cd /content/pmsd-project && git pull -q
print('Pulled recent changes from the repository.')
print('Running module main...')
! cd /content/pmsd-project && python3 -m main
```