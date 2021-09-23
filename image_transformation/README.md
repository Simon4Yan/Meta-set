
# Are Labels Always Necessary for Classifier Accuracy Evaluation? 


## PyTorch Implementation

This fold contains:

- CIFAR-10/CIFAR-100 and COCO Setups (use [imgaug](https://imgaug.readthedocs.io/en/latest/) to generate Meta-set).

Please follow the instruction below to install it and run the experiment demo.

### Prerequisites
* Linux (tested on Ubuntu 16.04LTS)
* NVIDIA GPU + CUDA CuDNN (tested on GTX 2080 Ti)
* [COCO 2017 Dataset](http://cocodataset.org) (download and unzip to ```PROJECT_DIR/extra_data/```)
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (download and unzip to ```PROJECT_DIR/dataset/```)
* You might need to change the file paths, and please be sure you change the corresponding paths in the codes as well     

## Getting started
0. Install dependencies 
    ```bash
   # Imgaug (or see https://imgaug.readthedocs.io/en/latest/source/installation.html)
    conda config --add channels conda-forge
    conda install imgaug
    ```
    
1. Customize the image transformations

    ```angular2
    # Customize the image transformations for constructing Meta-set. Here, we provide two examples
    # Example 1
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    list = [
        iaa.pillike.Autocontrast(),  # adjust contrast by cutting off p% of lowest/highest histogram values
        iaa.Multiply((0.1, 1.9), per_channel=0.2),  # make some images brighter and some darker
        iaa.pillike.EnhanceColor(),  # remove a random fraction of color from input images
        iaa.Solarize(0.5, threshold=(32, 128)),  # invert the colors
        iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5), # degrade the quality of images by JPEG-compressing them
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-30, 30)
        )), # add affine transformation
        iaa.Sharpen(alpha=(0.1, 1.9)),  # apply a sharpening filter kernel to images
    ]

    # Example 2
    # add more transformations into the list based on the users' need
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    list = [
         iaa.pillike.Autocontrast(),  # adjust contrast by cutting off p% of lowest/highest histogram values
         iaa.Multiply((0.1, 1.9), per_channel=0.2),  # make some images brighter and some darker
         iaa.pillike.EnhanceColor(),  # remove a random fraction of color from input images
         iaa.Solarize(0.5, threshold=(32, 128)),  # invert the colors
         iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast of images
         sometimes(iaa.Affine(
             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
             translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
             rotate=(-30, 30)
         )), # add affine transformation
         iaa.Sharpen(alpha=(0.0, 1.0)),  # apply a sharpening filter kernel to images
         iaa.LinearContrast((0.8, 1.2), per_channel=0.5),  # improve or worsen the contrast of images
         iaa.Rain(speed=(0.1, 0.3)),  # add rain to small images
         iaa.JpegCompression(compression=(70, 99)), # degrade the quality of images by JPEG-compressing them
         iaa.pillike.FilterDetail(),  # apply a detail enhancement filter kernel to images
         iaa.pillike.EnhanceSharpness(),  # alpha-blend two image sources using an alpha/opacity value
         iaa.MultiplyHue((0.8, 1.2)),  # change the sharpness of images
         iaa.Emboss(alpha=(0.0, 0.5), strength=(0.5, 1.0)),  # emboss an image, then overlay the results with the original
         iaa.AddToSaturation((-25, 25))  # add random values to the saturation of images
     ]
    ```
    
 2. Creat synthetic sets for CIFAR-10 setup
    ```bash
    # By default it creates 1000 synthetic sets
    python image_tranformation/synthesize_set_cifar.py
    ```
 3. Creat synthetic sets for COCO setup
    ```bash
    # By default it creates 1000 synthetic sets
    python image_tranformation/synthesize_set_coco.py
    ```
    
## Citation
If you use the code in your research, please cite:
```bibtex
    @inproceedings{deng2020labels,
    author={Deng, Weijian and Zheng, Liang},
    title     = {Are Labels Always Necessary for Classifier Accuracy Evaluation?},
    booktitle = {Proc. CVPR},
    year      = {2021},
    }
```

## License
MIT
