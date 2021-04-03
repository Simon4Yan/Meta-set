import os
import pathlib
import sys

sys.path.append(".")
import PIL
import numpy as np
from tqdm import trange

from meta_set.mnist import loadMnist, random_rotation_new


def creat_bg(input_image, img, change_colors=False):
    # Rotate image
    input_image = random_rotation_new(input_image)
    # Extend to RGB
    input_image = np.expand_dims(input_image, 2)
    input_image = input_image / 255.0
    input_image = np.concatenate([input_image, input_image, input_image], axis=2)

    # Convert the MNIST images to binary
    img_binary = (input_image > 0.5)

    # Take a random crop of the Lena image (background)
    x_c = np.random.randint(0, img.size[0] - 28)
    y_c = np.random.randint(0, img.size[1] - 28)
    image_bg = img.crop((x_c, y_c, x_c + 28, y_c + 28))
    # Conver the image to float between 0 and 1
    image_bg = np.asarray(image_bg) / 255.0

    if change_colors:
        # Change color distribution; this leads to more diverse changes than transformations
        for j in range(3):
            image_bg[:, :, j] = (image_bg[:, :, j] + np.random.uniform(0, 1)) / 2.0

    # Invert the colors at the location of the number
    image_bg[img_binary] = 1 - image_bg[img_binary]

    image_bg = image_bg / float(np.max(image_bg)) * 255
    return image_bg


def makeMnistbg(img, num=1):
    """
    Change the background of  MNIST images
    Select all testing samples from MNIST
    Store in numpy file for fast reading
    """

    index = str(num).zfill(5)
    np.random.seed(0)

    # Empty arrays
    test_data = np.zeros([28, 28, 3, 10000])
    test_label = np.zeros([10000])

    train_data = np.zeros([28, 28, 3, 50000])
    train_label = np.zeros([50000])

    try:
        os.makedirs('./dataset_bg/mnist_bg_%s/images/' % (index))
    except:
        None

    # testing images
    i = 0
    for j in range(10000):
        sample = all_samples_test[i]

        sample_rot = random_rotation_new(sample[0])
        test_data[:, :, :, j] = creat_bg(sample_rot, img)
        test_label[j] = sample[1]
        i += 1
        # save images only for visualization; I use npy file for dataloader
        # imsave("./dataset_bg/mnist_bg_%s/images/" % (index) + str(i) + ".jpg", test_data[:, :, :, j])
    np.save('./dataset_bg/mnist_bg_%s/test_data' % (index), test_data)
    np.save('./dataset_bg/mnist_bg_%s/test_label' % (index), test_label)


def makeMnistbg_path(img_paths, num=1):
    """
    Change the background of  MNIST images
    Select all testing samples from MNIST
    Store in numpy file for fast reading
    """
    index = str(num).zfill(5)
    np.random.seed(0)

    # Empty arrays
    test_data = np.zeros([28, 28, 3, 10000])
    test_label = np.zeros([10000])

    train_data = np.zeros([28, 28, 3, 50000])
    train_label = np.zeros([50000])

    try:
        os.makedirs('./dataset_bg/mnist_bg_%s/images/' % (index))
    except:
        None

    # testing images
    i = 0
    for j in range(10000):
        file = img_paths[np.random.choice(len(img_paths), 1)]
        img = PIL.Image.open(file[0])

        sample = all_samples_test[i]
        sample_rot = random_rotation_new(sample[0])
        test_data[:, :, :, j] = creat_bg(sample_rot, img)
        test_label[j] = sample[1]
        i += 1
        # save images only for visualization; I use npy file for the dataloader
        # imsave("./dataset_bg/mnist_bg_%s/images/" % (index) + str(i) + ".jpg", test_data[:, :, :, j])
    np.save('./dataset_bg/mnist_bg_%s/test_data' % (index), test_data)
    np.save('./dataset_bg/mnist_bg_%s/test_label' % (index), test_label)


if __name__ == '__main__':
    # ---- load mnist images ----#
    mnist_path = './dataset/mnist/'
    all_samples_test = loadMnist('test', mnist_path)
    all_samples_train = loadMnist('train', mnist_path)

    # ---- coco dataset - using coco training set following the  paper ----#
    path = './extra_data/train2014/'
    path = pathlib.Path(path)
    files = sorted(list(path.glob('*.jpg')) + list(path.glob('*.png')) + list(path.glob('*.JPEG')))

    # ---- generate smaple sets ----#
    print('==== generating sample sets ====')
    num = 200  # the number of sample sets (3000 sample sets; recommend use 200 for check the all codes)
    conut = 0

    # two ways of selecting coco images as background, both ways are similar in terms of meta set diversity
    # ----------- way 1 ----------- #
    for i in trange(num):
        try:
            img = PIL.Image.open(files[i])
            makeMnistbg(img, conut)
            conut += 1
        except:
            print('jump an image!')

    # ----------- way 2 ----------- #
    # for _ in trange(num):
    #     try:
    #         b_indice = np.random.choice(len( files), 1)
    #         img_paths = np.array(files)[b_indice]
    #         makeMnistbg_path(img_paths, conut)
    #         conut += 1
    #     except:
    #         print('jump an image!')
