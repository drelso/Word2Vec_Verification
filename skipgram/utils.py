###
#
# SkipGram Utilities
#
###

import numpy as np
import csv
import torch

# Negative Sampling auxiliary function
def init_sample_table(vocab_counts):
    """
    Construct table according to word
    frequencies (scaled by a factor of
    3/4). Elements of the table are word
    indices repeated according to their
    frequencies. This helps speed up
    performance, but size of the table
    depends on the size of the dataset,
    so size can become prohibitively
    large.
    
    This function was taken from Lanting
    Fang's SkipGram implementation in
    https://github.com/fanglanting/skip-gram-pytorch.git
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    vocab_counts : list[int]
        number of appearances of each word
        in the corpus
    
    Returns
    -------
    np.array
        1-D array containing word indices
        repeated according to their
        frequencies
    """
    count = [ele for ele in vocab_counts]
    pow_frequency = np.array(count)**0.75
    power = sum(pow_frequency)
    ratio = pow_frequency / power
    table_size = 1e8
    count = np.round(ratio*table_size)
    sample_table = []
    for idx, x in enumerate(count):
        sample_table += [idx]*int(x)
    return np.array(sample_table)


def neg_sampling(vocab_ratios, num_samples=5, batch_size=1):
    """
    Sample word indices according to their frequencies.
    This implementation is slower than new_neg_sampling,
    but more memory efficient.
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    vocab_ratios : list[float]
        scaled vocabulary frequencies obtained by the
        following calculation:
            pow_counts = np.array(vocab_counts)**0.75
            normaliser = sum(pow_counts)
            vocab_ratios = pow_counts / normaliser
    num_samples : int, optional
        number of indices to sample per batch
        (default: 5)
    batch_size : int, optional
        size of the batch (default: 1)
    
    Returns
    -------
    ndarray
        array of sampled indices of size
        num_samples x batch_size
        
    """
    negative_ixs = np.random.choice(len(vocab_ratios), size=(num_samples, batch_size), p=vocab_ratios)
    
    return negative_ixs


def new_neg_sampling(sample_table, num_samples=5, batch_size=1):
    """
    Faster implementation, dependent on a 1-D
    sample table made up of indices repeated
    according to their corresponding word
    frequencies
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    sample_table : list[int]
        list of word indices repeated according
        to their frequencies
    num_samples : int, optional
        number of indices to sample per batch
        (default: 5)
    batch_size : int, optional
        size of the batch (default: 1)
    
    Returns
    -------
    ndarray
        array of sampled indices of size
        num_samples x batch_size
    """
    neg_v = np.random.choice(sample_table, size=(batch_size, num_samples))
    return neg_v


def get_word2vec_vectors(gensim_path='data/word_embeddings/word2vec-google-news-300_voc1.csv'):
    """
    Load a CSV of word2vec Gensim vectors
    
    Requirements
    ------------
    import csv
    import torch
    
    Parameters
    ----------
    gensim_path : str
        filepath to the CSV embeddings file
    
    Returns
    -------
    torch.tensor
        tensor of word embeddings, dimensions
        are num_words x 300
    """
    with open(gensim_path, 'r') as v:
        i = 0
        word_vectors = []
        vocab = csv.reader(v)
        for row in vocab:
            vec = [float(d) for d in row[1:]]
            word_vectors.append(vec)
    
    return torch.tensor(word_vectors)


def save_param_to_npy(model, param_name, path):
    """
    Save PyTorch model parameter to NPY file
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    model : PyTorch model
        the model from which to get the
        parameters
    param_name : str
        name of the parameter weights to
        save
    path : str
        path to the file to save the parameters
        to
    
    """
    for name, param in model.named_parameters():
        if name == param_name + '.weight':
            weights = param.data.cpu().numpy()
    
    param_file = path + '-' + param_name
    
    np.save(param_file, weights)
    
    print("Saved ", param_name, " to ", path)