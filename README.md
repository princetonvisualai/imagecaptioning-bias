# Understanding and Evaluating Racial Biases in Image Captioning
### [Project Page](https://princetonvisualai.github.io/imagecaptioning-bias/) | [Paper](https://arxiv.org/abs/2106.08503) | [Annotations](https://forms.gle/FBi3ZsMDficweyP96)

This repo provides the code for our paper "Understanding and Evaluating Racial Biases in Image Captioning."

## Citation
    @article{zhao2021captionbias,
       author = {Dora Zhao and Angelina Wang and Olga Russakovsky},
       title = {Understanding and Evaluating Racial Biases in Image Captioning},
       year = {2021}
    }

## Requirements
* python = 3.8.2
* numpy = 1.18.5
* pandas = 1.1.3
* spacy = 2.3.5
* spacy_universal_sentence_encoder = 0.3.4
* pycocotools = 2.0.2
* scikit-learn = 0.23.2
* scipy = 1.4.1
* vadersentiment = 3.3.2

## Data annotations
To run the analyses, place the downloaded annotations in the folder ```annotations``` as well as the annotations provided with the [COCO 2014 dataset](https://cocodataset.org/#download).

## Models
We use six different image captioning models in our paper. The models are adapted from the following Github repositories and are trained using the respective protocols detailed in their papers. In our work, we train each model on the COCO 2014 training set and evaluate on the COCO 2014 validation set.
* **FC**, **Att2In**, **Transformer**: These models are adapted from Ruotian Luo's [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch).
* **DiscCap**: This model is adapted from Ruotian Luo's [DiscCaptioning](https://github.com/ruotianluo/DiscCaptioning).
* **AoA Net**: This model is adapted from Lun Huang's [AoANet](https://github.com/husthuaan/AoANet).
* **Oscar**: This model is adapted from Microsoft's [Oscar](https://github.com/microsoft/Oscar).

Place the model results in the ```results``` folder.  

## Analyses
All analysis files are found in the ```code``` folder.

* ```find_descriptors```: Gets occurrences of descriptors 
* ```image_appearance.py```: Analyze the simple image statistics of ground-truth COCO images.
* ```sentiment_analysis.py```: Analyze the sentiment in ground-truth and generated captions using VADER.
* ```caption_content.py```: Analyze the differentiability of ground-truth and generated captions.

To evaluate captions, you will need to follow the setup protocol [here](https://github.com/salaniz/pycocoevalcap) for pycocoevalcap. Our evaluation code can be found in ```evaluate_captions.py```.
