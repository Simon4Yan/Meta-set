
# Are Labels Always Necessary for Classifier Accuracy Evaluation? 
## [[Paper]](https://arxiv.org/abs/2007.02915) [[Project]](http://weijiandeng.xyz/AutoEval/)
![](http://weijiandeng.xyz/AutoEval/figs/fig1.png)


## PyTorch Implementation

This repository contains:

- the PyTorch implementation of AutoEavl
- the example on MNIST setup
- FD calculation and two regression methods
- CIFAR-10/CIFAR-100 and COCO Setups (use [imgaug](https://imgaug.readthedocs.io/en/latest/) to generate Meta-set).
  Please see ```PROJECT_DIR/image_transformation/```

Please follow the instruction below to install it and run the experiment demo.

### Prerequisites
* Linux (tested on Ubuntu 16.04LTS)
* NVIDIA GPU + CUDA CuDNN (tested on GTX 2080 Ti)
* [COCO 2017 Dataset](http://cocodataset.org) (download and unzip to ```PROJECT_DIR/extra_data/```)
* [MNIST dataset-link](https://drive.google.com/file/d/1wq8pIdayAbCu5MBfT1M38BATcShsaaeq/view?usp=sharing) (download and unzip to ```PROJECT_DIR/dataset/```)
* Please use PyTorch1.1 to avoid compilation errors (other versions should be good)
* You might need to change the file paths, and please be sure you change the corresponding paths in the codes as well     

## Getting started
0. Install dependencies 
    ```bash
   # COCOAPI
    cd $DIR/libs
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    python setup.py build_ext install
   
    ```
 1. Creat Meta-set
    ```bash
    # By default it creates 300 sample sets
    python meta_set/main.py
    ```
 2. Learn classifier
    ```bash
    # Save as "PROJECT_DIR/learn/mnist_cnn.pt"
    python learn/train.py
    ```
 3. Test classifier on Meta-set
    ```bash
    # Get "PROJECT_DIR/learn/accuracy_mnist.npy" file
    python learn/many_test.py
    ```
 4. Calculate FD on Meta-set
    ```bash
    # Get "PROJECT_DIR/FD/fd_mnist.npy" file
    python FD/many_fd.py
    ```
 5. Linear regression
    ```bash
    # You will see linear_regression_train.png;
    # then check if FD and Accuracy have a linear relationship;
    # If so, it is all good, and please go back to step 1 and create 3000 sample sets.
    python FD/linear_regression.py
    ``` 
 6. Network regression
    ```bash
    # Please follow the instructions in the script to train the model
    # Save as "PROJECT_DIR/FD/mnist_regnet.pt"
    python FD/train_regnet.py
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
