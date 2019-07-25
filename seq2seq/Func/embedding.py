
import numpy as np
import string


def char2id(char,first_letter):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 2
    elif char == ' ':
        return 1
    elif char=='#':
        return 0
    else:
        print('Unexpected character: %s' % char)
    return 0

def id2char(dictid,first_letter):
    if dictid > 1:
        return chr(dictid + first_letter - 2)
    elif dictid==1:
        return ' '
    elif dictid==0:
        return '#'

def characters(probabilities,first_letter):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c,first_letter) for c in np.argmax(probabilities, 1)]

def batches2string(batches,first_letter):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b,first_letter))]
    return s

