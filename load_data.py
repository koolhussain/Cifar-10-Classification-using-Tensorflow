import download
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
path = os.getcwd()

def download_data():
    if os.path.isdir(os.path.join(path,"cifar-10-batches-py")):
        print("File already Exists")
    else:
        download.download(url, path, kind='tar.gz', progressbar = True, replace = False, verbose = False)

def get_filepath():
    return os.path.join(path,"cifar-10-batches-py")

def unpickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    img_data = data["data"]
    img_labels = data["labels"]
    return img_data, img_labels

def convert_images(data):
    X_image = []
    for im in data:
        im_r = im[0:1024]
        im_g = im[1024:2048]
        im_b = im[2048:]

        img = np.dstack((im_r,im_g,im_b))
        img_r = np.reshape(img,[32,32,3])
        X_image.append(img_r)
    return X_image
    
def load_data():
    X_train = []
    Y_train = []
    filepath = get_filepath()
    for i in range(1,6):
        filename = os.path.join(filepath,'data_batch_' + str(i))
        img_data, img_labels = unpickle(filename)
        if i == 1:
            X_train = img_data
            Y_train = img_labels
        else:
            X_train = np.concatenate([X_train, img_data], axis=0)
            Y_train = np.concatenate([Y_train, img_labels], axis=0)

    X_train = convert_images(X_train)
    
    filename = os.path.join(filepath,"test_batch")
    X_test, Y_test = unpickle(filename)

    X_test = convert_images(X_test)

    return (X_train, Y_train), (X_test, Y_test)

def plotting():
    (X_train, Y_train), (X_test, Y_test) = load_data()
    print(X_train[0].shape)
    plt.imshow(X_train[1000])
    plt.show()

if __name__ == "__main__":
    plotting()




    
load_data()
