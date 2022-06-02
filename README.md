# Extraction of Liver Vessel Systems From CT-Image


## About The Project


### Introduction

Extraction of the blood vessel system of a liver is a challenging task in the field of medical image processing. Normally, doctors must examine each slice manually to achieve accurate vessel segmentation. Our solution is called **Extraction of Liver Vessel Systems From CT-Image**, an automatic vessel segmentation. 

This is also a *scientific research* I and Duong, a friend of mine, conducted together. Here is the link to our paper [Extraction of Liver Vessel Systems From CT-Image](https://drive.google.com/file/d/1ZSpyWUpRR0TInZEg4b-lZaciS8tsj_mM/view?usp=sharing).

This project is trained and evaluated on [the public IRCAD dataset.](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/)


### Results
<img src="https://github.com/phkhanhtrinh23/vessel_segmentation/blob/main/images/1.png">
<img src="https://github.com/phkhanhtrinh23/vessel_segmentation/blob/main/images/2.png">
<img src="https://github.com/phkhanhtrinh23/vessel_segmentation/blob/main/images/3.png">
<img src="https://github.com/phkhanhtrinh23/vessel_segmentation/blob/main/images/4.png">
<img src="https://github.com/phkhanhtrinh23/vessel_segmentation/blob/main/images/5.png">
<img src="https://github.com/phkhanhtrinh23/vessel_segmentation/blob/main/images/6.png">


### Built With

My application is built with these frameworks and tools:
* [Google Colab Notebook](https://colab.research.google.com/)
* [NVIDIA Tesla P100 – PCIE GPU (16 GB memory)](https://www.nvidia.com/en-us/data-center/tesla-p100/)
* [Tensorflow 2.6.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.6.0)
* [Python 3.7](https://www.python.org/downloads/release/python-3712/)


## Getting Started

To get started, you should have prior knowledge on **Python** and **Tensorflow** at first. A few resources to get you started if this is your first Python or Tensorflow project:

- [Tensorflow: Guide](https://www.tensorflow.org/guide)
- [Python for Beginners](https://www.python.org/about/gettingstarted/)


## Installation and Run

1. Clone the repo

   ```sh
   git clone https://github.com/phkhanhtrinh23/vessel_segmentation.git
   ```
  
2. Use any code editor to open the folder **vessel_segmentation**.


## Step-by-step

1.  Run `preprocess_train_data.py` and `preprocess_valid_data.py` to *preprocess* and *generate* train and validation CT-Images, and then save them in the folder `./data/artery`.

2. Open the notebook `train.ipynb` and follow the pipeline in that notebook.

3. The folder `results` is the folder of pre-saved *weights* and *predictions* that I have trained and validated. Up till now, our best Dice Score is **79.6%**.

4. Enjoy!


## Contribution

Contributions are what make GitHub such an amazing place to be learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the project
2. Create your Contribute branch: `git checkout -b contribute/Contribute`
3. Commit your changes: `git commit -m 'add your messages'`
4. Push to the branch: `git push origin contribute/Contribute`
5. Open a pull request


## Contact

Email: phkhanhtrinh23@gmail.com

LinkedIn: https://www.linkedin.com/in/trinh-pham-3103081a0/

Project Link: https://github.com/phkhanhtrinh23/vessel_segmentation.git
