# import working as w

# print(w.MyFunc())


import urllib.request
import gzip
import pickle
import os
import numpy as np
from PIL import Image

output_dir = os.path.dirname(os.path.abspath(__file__)) + '/dist'
cache_file = 'data.pkl'
img_size = 784  # TODO: Don't hard code ?
url_base = 'http://yann.lecun.com/exdb/mnist/'
files = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_solutions': 'train-labels-idx1-ubyte.gz',
    'test_images': 't10k-images-idx3-ubyte.gz',
    'test_solutions': 't10k-labels-idx1-ubyte.gz'
}


def _get_file_dir(file_name):
    return output_dir + '/' + file_name


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
        print('File', file_name, 'already exists')
        return

    print('Downloading', file_name, 'from', url, end=' ...\t')
    urllib.request.urlretrieve(url, file_path)
    print('Done')


def _load_solutions(file_name):
    """
    Parameters
    ----------
    file_name : The name of the solution gzip

    Returns
    ----------
    An array of numbers corresponding to image solutions
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

    Returns
    ----------
    An array of images, each represented as an array of pixels
    """
    file_path = _get_file_dir(file_name)

    # Loads the image interpreting each byte as an unsigned int
    print('Loading image', file_name, end='...\t')
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    print('Done')

    # The initial format is a 1D array of pixels where every ${img_size} bytes
    # represents a single image
    print('\t>', 'Loaded', len(data), 'pixels')

    # We'll split every ${img_size} bytes into it's own array. Each element of data is
    # now a ${img_size} length array representing one image. Reshape converts our array
    # into a new N x M shape. -1 tells numpy to infer the dimenson length from the other
    # parameter
    data = data.reshape(-1, img_size)

    total_images = len(data)
    image_size = len(data[0])

    print('\t> Parsed', total_images, 'images, each', image_size, 'pixels')
    return data


def _load_dataset():
    dataset = {}
    dataset['train_images'] = _load_images(files['train_images'])
    dataset['train_solutions'] = _load_solutions(files['train_solutions'])
    dataset['test_images'] = _load_images(files['test_images'])
    dataset['test_solutions'] = _load_solutions(files['test_solutions'])
    return dataset


def _save_file(file_name, data):
    file_path = _get_file_dir(file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, -1)


def _reshape_solution_to_bitmap(solution_array):
    """"
    Parameters
    ----------
    solution_array: Array of numbers corresponding to image solutions

    Returns
    An array of bitmaps, which each have 10 elements representing 0 - 9 where the index
    represents the number and 0 and 1 represents the answer.
    Example: 3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    """

    bitmap_array = np.zeros((solution_array.size, 10))

    for index, row in enumerate(bitmap_array):
        value = solution_array[index]
        row[value] = 1

    return bitmap_array


def load_dataset(normalize=True, flatten=True, bitmapped=False):
    if not os.path.exists(_get_file_dir(cache_file)):
        # Download each file if necessary
        for value in files.values():
            _download_file(url_base + value, value)

        # Store data in binary file
        _save_file(cache_file, _load_dataset())

    with open(_get_file_dir(cache_file), 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        # Map image data from [0 to 255] to [0 to 1]
        for key in ('train_images', 'test_images'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if not flatten:
        # Convert images to 28 x 28 arrays
        # TODO: Get rid of second dimension ?
        for key in ('train_images', 'test_images'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    if bitmapped:
        for key in ('train_solutions', 'test_solutions'):
            dataset[key] = _reshape_solution_to_bitmap(dataset[key])

    return (
        (dataset['train_images'], dataset['train_solutions']),
        (dataset['test_images'], dataset['test_solutions']))


load_dataset()
