# import working as w

# print(w.MyFunc())


import urllib.request
import gzip
import pickle
import os
import numpy as np
from PIL import Image

output_dir = os.path.dirname(os.path.abspath(__file__))
cache_file = "data.pkl"
img_size = 784  # TODO: Don't hard code ?
url_base = 'http://yann.lecun.com/exdb/mnist/'
files = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}


def _get_file_dir(file_name):
    return output_dir + "/" + file_name


def _download_file(url, file_name):
    """
    Downloads a file from a given url

    Parameters
    ----------
    url : The URL of the resource to download
    file_name : The final name of the downloaded resource
    """
    file_path = _get_file_dir(file_name)

    if os.path.exists(file_path):
        print("File", file_name, "already exists")
        return

    print("Downloading", file_name, "from", url, " ... ", end="")
    urllib.request.urlretrieve(url, file_path)
    print("Done")


def _load_solutions(file_name):
    """
    Loads an array of solutions to each image

    Parameters
    ----------
    file_name : The name of the solution gzip
    """
    file_path = _get_file_dir(file_name)
    with gzip.open(file_path, 'rb') as f:
        solutions = np.frombuffer(f.read(), np.uint8, offset=8)
    return solutions


def _load_images(file_name):
    """
    Loads an array of images

    Parameters
    ----------
    file_name : The name of the solution gzip
    """
    file_path = _get_file_dir(file_name)

    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    # Split every img_size th line into a new row
    data = data.reshape(-1, img_size)
    return data

def _load_dataset():
    dataset = {}
    dataset['train_img'] = _load_images(files['train_img'])
    dataset['train_label'] = _load_solutions(files['train_label'])
    dataset['test_img'] = _load_images(files['test_img'])
    dataset['test_label'] = _load_solutions(files['test_label'])
    return dataset


def _save_file(file_name, data):
    file_path = _get_file_dir(file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, -1)


def load_dataset(_output_dir):
    # Set the output directory
    global output_dir
    output_dir = _output_dir

    if not os.path.exists(cache_file):
        # Download each file if necessary
        for value in files.values():
            _download_file(url_base + value, value)

        # Store data in binary file
        _save_file(cache_file, _load_dataset())

    with open(cache_file, 'rb') as f:
        dataset = pickle.load(f)

    print(dataset)


load_dataset(os.path.dirname(os.path.abspath(__file__)))
