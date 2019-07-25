import os
from six.moves.urllib.request import urlretrieve
import zipfile
import tensorflow as tf

def reverse(alist):
    newlist = []
    for i in range(1, len(alist) + 1):
        newlist.append(alist[-i])
    return newlist

def maybe_download(url, filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
        filepath = ''
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('Downloaded and verified %s' % filename)
        else:
            print(statinfo.st_size)
            raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    else:
        print('Found and verified %s' % filename)
    #return filepath

def split_data(filename,train_ratio):
    """Download a file if not present, and make sure it's the right size."""
    with zipfile.ZipFile(filename) as f:
        name = f.namelist()[0]
        data = tf.compat.as_str(f.read(name)) #Converts any string-like python input types to unicode
        train_size=round(len(data)*train_ratio)
        train_text=data[:train_size]
        valid_text=data[train_size:]
        valid_size=len(valid_text)
        print('Data size %d' % len(data))
        print('train size: ', train_size, 'train text preview: ', train_text[:64])
        print('valid size: ', valid_size, 'valid text preview: ' ,valid_text[:64])
    return train_text, valid_text, train_size, valid_size
  
