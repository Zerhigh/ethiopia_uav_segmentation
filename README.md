This project is part of the Li4Lam workshop held in Addis Ababa between October 27-30, 2024.

### Project Setup

This is a Python 3.9 project. First create a virtual environment in this projects directory and install the required packages (this might take some time):

```
python -m venv venv

# In cmd.exe
venv\Scripts\activate.bat

# In PowerShell
venv\Scripts\Activate.ps1

# install provided packages
pip install -r requirements.txt
```

### Data access

Download the necessary data (4GB) here (account is required): [UAV dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset/discussion/289363)

Or get the data via USB stick from one of the presenters.

### Training 

To train a model, execute the script ```main.py```. Please take care to change the ```BASE_PATH``` variable according to your directory structure.

To run inference on your trained model execute the ```inference.py``` script. It is necessary to change your filepaths accordingly as well.

### Cloud solution

The file ````ethiopia_uav.ipynb```` contains the whole functionalities of this repository in a ```.ipynb``` notebook. Upload this notebook into a Cloud IDE such as Google Collab. 

### Sources

Data: [http://dronedataset.icg.tugraz.at](http://dronedataset.icg.tugraz.at)

Code (modified): [kaggle](https://www.kaggle.com/code/ligtfeather/semantic-segmentation-is-easy-with-pytorch#Evaluation)

