###
#
# True data distribution: p(w_context | w_focus)
#
###

import time
import os

import csv
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def context_count_matrix(proc_lt_npy_dataset, vocab_file, counts_savefile):
    """
    Calculate the count matrix for words within
    each other's context (directional, i.e. only
    adds up counts where focus words (rows)
    appear together with context words (columns))
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    proc_lt_npy_dataset : str
        path to processed lightweight data file
    vocab_file : str
        path to dictionary file
    counts_savefile : str
        filepath to the file to save the counts
        matrix to
    """
    
    print('Getting vocabulary size from:', vocab_file)
    vocab_size = len(open(vocab_file, 'r', encoding='utf-8').readlines())
    print('Vocabulary size:', vocab_size)
    
    print('Loading dataset NumPy file:', proc_lt_npy_dataset)
    data = np.load(proc_lt_npy_dataset, mmap_mode='r')
    
    print('Data size:', data.shape)
    
    print('First row: ', data[0])
    print('Types:', type(data[0,0]), type(data[0,1]))
    
    count_mtx = np.zeros((vocab_size, vocab_size))
    
    for row in data:
        count_mtx[row[0], row[1]] += 1
    
    np.save(counts_savefile, count_mtx)
    print('Saved file to', counts_savefile)


def counts_mtx_norm_vect(counts_matrix_file, counts_norm_file):
    counts = np.load(counts_matrix_file, mmap_mode='r')
    
    norms = []
    zeros = 0
    
    for i, row in enumerate(counts):
        total = np.sum(row)
        
        norms.append(total)
        
        if total == 0:
            # print('Zeros row', i)
            # print(row)
            # break
            zeros += 1
    
    print('Number of zeros:', zeros)
    
    print('Saving vector of normalisation terms to', counts_norm_file)
    np.save(counts_norm_file, norms)


def normalise_counts_mtx(counts_matrix_file, norm_counts_matrix_file):
    counts = np.load(counts_matrix_file, mmap_mode='r')
    
    for i, row in enumerate(counts):
        total = np.sum(row)
        
        if total == 0:
            print('Zeros row', i)
            print(row)
            break
    # norm_counts = normalize_rows(counts)
    # np.save(norm_counts_matrix_file, norm_counts)


def normalize_rows(x: np.ndarray):
    """
    @FROM https://necromuralist.github.io/neural_networks/posts/normalizing-with-numpy/
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    

def cross_entropy_single_word(word_ctx_counts, word_norm_dot_prod):    
    # data = np.load(counts_matrix_file, mmap_mode='r')
    # word_counts = data[index]
    # del data
    
    # normaliser = np.sum(word_ctx_counts)
    # word_freqs = word_total_counts / normaliser
    # word_log_freqs = np.log(word_freqs)
    
    # print('word_counts', word_ctx_counts)
    # print('normaliser', normaliser)
    # print('word_freqs', word_freqs)
    # print('word_log_freqs', word_log_freqs)
    
    # cross_ent = 0
    
    # for i, freq in enumerate(word_freqs):
    #     p = freq
    #     cross_ent -= freq * word_log_freqs[i]
    
    # return cross_ent
    
    # print('Word context counts shape', word_ctx_counts.shape)
    # print('Word norm dot prod shape', word_norm_dot_prod.shape)
    # print('Word norm dot prod shape', np.log(word_norm_dot_prod).shape)
    
    num_0s = 0
    for i in word_norm_dot_prod:
        if i == 0: num_0s += 1
        
    norm_ctx_counts = word_ctx_counts / np.sum(word_ctx_counts)
    
    # print('Number of zero elements:', num_0s)
    # print('Sum of word_norm_dot_prod:', np.sum(word_norm_dot_prod))
    
    return -np.dot(norm_ctx_counts, np.log(word_norm_dot_prod))
    # np.dot


def cross_entropy_all_words(ctx_counts_file, counts_norm_file, norm_dot_prods_file, cross_ent_savefile):
    
    ctx_counts = np.load(ctx_counts_file, mmap_mode='r')
    counts_norm = np.load(counts_norm_file)
    norm_dot_prods = np.load(norm_dot_prods_file, mmap_mode='r')
    
    cross_entropies = []
    
    if ctx_counts.shape == norm_dot_prods.shape:
        print('Shapes match, calculating cross entropies')
        
        for i in range(len(ctx_counts)):
            num_nonzero = 0
            if counts_norm[i] > 0:
                norm_ctx_counts = ctx_counts[i] / counts_norm[i]
                
                word_cross_ent = -np.dot(norm_ctx_counts, np.log(norm_dot_prods[i]))
                
                cross_entropies.append(word_cross_ent)
                num_nonzero += 1
            else:
                # To maintain the right word
                # indices append a NaN to the
                # cross entropies array
                cross_entropies.append(np.nan)
            
            # print('Word cross ent:', word_cross_ent)    
            
        print('Saving cross entropy scores to', cross_ent_savefile)
        np.save(cross_ent_savefile, cross_entropies)
    else:
        print('Shape mismatch: ctx_counts shape = %r  norm_dot_prods shape = %r' % (ctx_counts.shape, norm_dot_prods.shape))
    
    valid_cross_ents = [cross_ent for cross_ent in cross_entropies if not np.isnan(cross_ent)]
    return np.mean(valid_cross_ents)

def all_dot_prods(i_embs, o_embs, embs_dot_prod_file):
    """
    Matrix multiplication between input and output
    embedding matrices R = I * O.T
    After matrix multiplication exponentiate all
    matrix entries, this requires datatype to be
    Float64 to prevent overflow
    Faster to calculate with CUDA, but is limited
    to the GPU's memory
    
    Requirements
    ------------
    import torch
    import numpy as np
    
    Parameters
    ----------
    i_embs : NumPy array[float]
        matrix of (Word2Vec style) input
        embeddings of size |V| x D
    o_embs : NumPy array[float]
        matrix of (Word2Vec style) output
        embeddings of size |V| x D
    embs_dot_prod_file : str
        file path to the destination file for
        the results
    """
    if torch.cuda.is_available(): print('CUDA is available, running on GPU')

    # If CUDA is available, default all tensors to CUDA tensors
    # torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
    torch.set_default_tensor_type(torch.DoubleTensor)
    
    
    i_embs_tensor = torch.from_numpy(i_embs)#.to(torch.device("cuda"))
    o_embs_tensor = torch.from_numpy(o_embs)#.to(torch.device("cuda"))
    
    print('Shapes:', i_embs_tensor.shape, torch.t(o_embs_tensor).shape)
    print('Types', i_embs_tensor.type(), o_embs_tensor.type())
    print('CUDA?', i_embs_tensor.is_cuda, o_embs_tensor.is_cuda)
    
    print('Calculating dot product')
    result_tensor = torch.mm(i_embs_tensor, torch.t(o_embs_tensor)) #i_embs_tensor * torch.t(o_embs_tensor)
    
    dot_prods = result_tensor.cpu().numpy()
    # exp_dot_prods = np.exp(result_tensor.cpu().numpy(), dtype='Float64')
    
    print('Saving to ', embs_dot_prod_file)
    np.save(embs_dot_prod_file, dot_prods)


def exp_dot_prods(embs_dot_prod_file, embs_exp_dot_prod_file, del_dot_prods_file=True):
    dot_prods = np.load(embs_dot_prod_file, mmap_mode='r')
    
    exp_dot_prods = np.exp(dot_prods, dtype='Float64')
    print('Exp dtype:', exp_dot_prods.dtype)
    print('Saving exponentiated dot products to:', embs_exp_dot_prod_file)
    np.save(embs_exp_dot_prod_file, exp_dot_prods)
    
    if del_dot_prods_file:
        print('Deleting previous (non-exponentiated) file: ', embs_dot_prod_file)
        os.remove(embs_dot_prod_file)
    

def normalisation_vector(embs_dot_prod_file, norm_vector_file):
    """
    Calculate the normalisation term for every
    embedding w.r.t. the rest of the embeddings
    in the vocabulary:
        sum_{w in V}(v'_w * v_f)
        
    This is done by suming across all columns
    in the pre-calculated dot product matrix
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    embs_dot_prod_file : str
        file containing the pre-calculated dot
        products
    norm_vector_file : str
        destination file to store the
        normalisation terms
    """
    dot_prods = np.load(embs_dot_prod_file, mmap_mode='r')
    
    print('Calculating row sums for', embs_dot_prod_file)
    
    norm_vector = dot_prods.sum(axis=1)
    
    print('Saving normalisation vector', norm_vector_file)
    np.save(norm_vector_file, norm_vector)


def normalised_dot_prods(embs_dot_prod_file, norm_vector_file, norm_dot_prods_file, valid_index=143, del_dot_prods_file=True):
    """
    Calculate the normalised dot products by
    multiplying every row of the dot product
    matrix by the reciprocal of the word's
    normalisation term and save the resulting
    |V|x|V| matrix into a file
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    embs_dot_prod_file : str
        file containing the pre-calculated dot
        products
    norm_vector_file : str
        destination file to store the
        normalisation terms
    norm_dot_prods_file : str
        destination file to store the
        normalisation terms
    """
    dot_prods = np.load(embs_dot_prod_file, mmap_mode='r')
    norm_vector = np.reciprocal(np.load(norm_vector_file))
    
    print('Calculating normalised dot products')
    norm_dot_prods = (dot_prods.T * norm_vector).T
    
    print('Saving normalised dot products to', norm_dot_prods_file)
    np.save(norm_dot_prods_file, norm_dot_prods)
    
    ### VERIFYING RESULTS
    calc_norm_dot_prod = np.exp(dot_prods[valid_index]) / norm_vector[valid_index]
    difference = norm_dot_prods[valid_index] - calc_norm_dot_prod
    print(difference)
    print('Sum of difference', np.sum(difference))
    print('Sum of calculated norm dot prods row:', np.sum(calc_norm_dot_prod))
    print('Sum of norm dot prods row:', np.sum(norm_dot_prods[valid_index]))
    ###
    
    if del_dot_prods_file:
        print('Deleting previous (non-normalised) file: ', embs_dot_prod_file)
        os.remove(embs_dot_prod_file)


def verify_dot_prod(i_embs, o_embs, embs_dot_prod_file, index_1=20, index_2=543, is_exp=False):
    
    i_emb = i_embs[index_1]
    o_emb = o_embs[index_2]
    # dot_prod = np.exp(np.dot(i_emb, o_emb))
    dot_prod = np.dot(i_emb, o_emb)
    
    dot_prods = np.load(embs_dot_prod_file, mmap_mode='r')
    
    print('Dot prods shape', dot_prods.shape)
    print('Dot prods dtype', dot_prods.dtype)
    
    
    if is_exp:
        print('Exp dot prod: ', np.exp(dot_prod, dtype='Float64'))
    else:
        print('Dot prod: ', dot_prod)
    print('Pre-calc dot prod', dot_prods[index_1, index_2])
    
    
    
def verify_norm_dot_prod(embs_dot_prod_file, norm_vector_file, norm_dot_prods_file, index=53140):
    """
    Verify if the pre-computed normalised exp dot
    products match the manual calculation for a
    single word vector pair for a given index
    """
    
    dot_prods = np.load(embs_dot_prod_file, mmap_mode='r')
    norm_vector = np.load(norm_vector_file)
    norm_dot_prods = np.load(norm_dot_prods_file, mmap_mode='r')
    
    print('Dot prods shape', dot_prods.shape)
    print('Norm vector shape', norm_vector.shape)
    print('Norm dot prods shape', norm_dot_prods.shape)
    
    calc_norm_dot_prod = np.exp(dot_prods[index]) / norm_vector[index]
    
    if np.array_equal(calc_norm_dot_prod, norm_dot_prods[index]):
        print('Pre-calculated and calculated arrays are equal')
    else:
        print('Pre-calculated and calculated arrays are different')
        difference = norm_dot_prods[index] - calc_norm_dot_prod
        print(difference)
        print('Sum of difference', np.sum(difference))
    
    print('Sum of calculated norm dot prods row:', np.sum(calc_norm_dot_prod))
    print('Sum of norm dot prods row:', np.sum(norm_dot_prods[index]))


def verify_norm_exp_dot_prod(i_embs, o_embs, norm_vector_file, norm_exp_dot_prods_file, index=53140):
    """
    Verify if the pre-computed normalised exp dot
    products match the manual calculation for a
    single word vector pair for a given index
    """
    
    i_emb = i_embs[index]
    o_emb = o_embs[index]
    
    print('i emb shape', i_emb.shape)
    print('o embs shape', o_embs.shape)
    
    dot_prods = np.dot(o_embs, i_emb)
    print('Dot prods shape', dot_prods.shape)
    print('Dot prods sum', np.sum(dot_prods))
    
    print('Single dot prod: ', np.dot(o_emb, i_emb))
    print('Pre-computed dot prod: ', dot_prods[index])
    
    # dot_prod = np.dot(i_emb, o_emb)
    
    # dot_prods = np.load(embs_dot_prod_file, mmap_mode='r')
    norm_vector = np.load(norm_vector_file)
    norm_exp_dot_prods = np.load(norm_exp_dot_prods_file, mmap_mode='r')
    
    # print('Norm vector shape', norm_vector.shape)
    print('Norm dot prods shape', norm_exp_dot_prods.shape)
    print('Norm exp dot prod', norm_exp_dot_prods[index])
    print('Norm exp dot prod sum', np.sum(norm_exp_dot_prods[index]))
    
    calc_norm_dot_prod = np.exp(dot_prods) / norm_vector[index]
    print('Calculated norm exp dot prod', calc_norm_dot_prod)
    
    if np.array_equal(calc_norm_dot_prod, norm_exp_dot_prods[index]):
        print('Pre-calculated and calculated arrays are equal')
    else:
        print('Pre-calculated and calculated arrays are different')
        difference = np.abs(norm_exp_dot_prods[index] - calc_norm_dot_prod)
        print(difference)
        print('Sum of difference', np.sum(difference))
    
    # print('Sum of calculated norm dot prods row:', np.sum(calc_norm_dot_prod))
    # print('Sum of norm dot prods row:', np.sum(norm_dot_prods[index]))



def plot_cross_ents_histogram(cross_ents_file, save_file=None, plot=False, title='Cross entropies histogram'):
    """
    Plot histograms of individual cross entropies
    from a NumPy file
    
    Requirements
    ------------
    import numpy as np
    import matplotlib.pyplot as plt
    
    Parameters
    ----------
    cross_ents_file : str
        file containing the cross entropy scores
    save_file : str, optional
        path to save the plot to (default: None)
    plot : bool, optional
        if true renders plot to screen
        (default: False)
    title : str, optional
        title for the plot (default: 'Cross
        entropies histogram')
    """
    fig, ax = plt.subplots()
    
    data = np.load(cross_ents_file)
    
    cross_ents = [cross_ent for cross_ent in data if not np.isnan(cross_ent)]
    
    ax.hist(cross_ents, bins=50)
    
    # bins = dist_hist[0, 5:65, 0]
    # bin_ticks = ["{0:.2f}".format(b) for b in bins]
    # xs = np.arange(len(bin_ticks))
    # bins = np.linspace(bins_ar.min(), bins_ar.max(), bins_ar.shape[0])
    # dists = dist_hist[0, 5:65, 1]
    
    # ax.bar(xs, dists, alpha=0.5, label=name)    
    
    # x_ticks = [x for x in xs if x % 10 == 1]
    # ax.set_xticks(x_ticks)#xs)
    # b_ticks = [bin_ticks[i] for i in x_ticks]
    # print('bins', b_ticks)
    # ax.set_xticklabels(b_ticks)
    
    # ax.legend()
    
    # ax.set_ylabel('Cross entropies')
    
    ax.set_title(title)
    
    if not save_file is None:
        print('Saving histogram to', save_file)
        plt.savefig(save_file)
    
    if plot: plt.show()


def cross_ent_luke(lt_npy_data_file, norm_exp_dot_prods_file, cross_ents_savefile):
    data = np.load(lt_npy_data_file, mmap_mode='r')
    norm_exp_dot_prods = np.load(norm_exp_dot_prods_file, mmap_mode='r')
    
    num_datapoints = data.shape[0]
    
    print('Num datapoints: ', num_datapoints)
    
    cross_ent = 0.
    
    for focus,context in data:
        # print('Focus:', focus)
        # print('Context:', context)
        # break
        cross_ent -= np.log(norm_exp_dot_prods[focus,context])
    
    return cross_ent / num_datapoints


if __name__ == '__main__':
    start_time = time.time()
    
    proc_data_dir = 'data/processed_data/'
    vocab_dir = 'data/vocabulary/'
    counts_dir = 'data/counts/'
    
    voc_threshold = 1
    vocab_file = vocab_dir + 'vocab_' + str(voc_threshold) + '.csv'
    
    ctx_size = 5
    
    skipgram_prefix = 'skipgram_data_ctx' + str(ctx_size) + '_lt_v' + str(voc_threshold)

    skipgram_filename_lt = proc_data_dir + skipgram_prefix
    lt_npy_proc_train_data_file = skipgram_filename_lt + '_train_1.npy'
    lt_npy_proc_val_data_file = skipgram_filename_lt + '_val_1.npy'
    
    skipgram_counts_filename_lt = counts_dir + skipgram_prefix
    
    lt_npy_train_counts_file = skipgram_counts_filename_lt + '_train_counts_1.npy'
    lt_npy_val_counts_file = skipgram_counts_filename_lt + '_val_counts_1.npy'
    
    context_count_matrix(lt_npy_proc_train_data_file, vocab_file, lt_npy_train_counts_file)
    context_count_matrix(lt_npy_proc_val_data_file, vocab_file, lt_npy_val_counts_file)
    
    lt_npy_train_counts_norm_file = skipgram_counts_filename_lt + '_train_counts_norm_1.npy'
    lt_npy_train_counts_norm_vector_file = skipgram_counts_filename_lt + '_train_counts_norm_vector_1.npy'
    lt_npy_val_counts_norm_file = skipgram_counts_filename_lt + '_val_counts_norm_1.npy'
    lt_npy_val_counts_norm_vector_file = skipgram_counts_filename_lt + '_val_counts_norm_vector_1.npy'
    
    # normalise_counts_mtx(lt_npy_train_counts_file, lt_npy_train_counts_norm_file)
    # counts_mtx_norm_vect(lt_npy_train_counts_file, lt_npy_train_counts_norm_vector_file)
    
    # normalise_counts_mtx(lt_npy_val_counts_file, lt_npy_val_counts_norm_file)
    # counts_mtx_norm_vect(lt_npy_val_counts_file, lt_npy_val_counts_norm_vector_file)
    
    model_names = [
            'prev_rand_init-no_syns-10e-voc1-emb300',
            'rand_init-no_syns-10e-voc1-emb300',
            'rand_init-syns25-10e-voc1-emb300',
            'rand_init-syns16-10e-voc1-emb300',
            'w2v_init-no_syns-10e-voc1-emb300',
            'w2v_init-syns25-10e-voc1-emb300',
            'w2v_init-syns16-10e-voc1-emb300'
    ]
    
    # model_name = 'rand_init-no_syns-10e-voc1-emb300'
    
    cross_entropies = [['model_name', 'average_cross_ent_train', 'average_cross_ent_val', 'luke_cross_ent_train', 'luke_cross_ent_val']]
    
    for model_name in model_names:
        print('\nProcessing', model_name)
        model_path = 'model/' + model_name + '/'
        model_file_prefix = model_path + model_name
        
        i_emb_npy_file = model_file_prefix + '-i_embedding.npy'
        o_emb_npy_file = model_file_prefix + '-o_embedding.npy'
        
        embs_dot_prod_file = model_file_prefix + '-embs_dot_prod.npy'
        embs_exp_dot_prod_file = model_file_prefix + '-embs_exp_dot_prod.npy'
        
        norm_vector_file = model_file_prefix + '-norm_vector.npy'
        norm_exp_dot_prods_file = model_file_prefix + '-embs_norm_exp_dot_prod.npy'
        cross_ent_train_file = model_file_prefix + '-cross_entropies_train.npy'
        cross_ent_val_file = model_file_prefix + '-cross_entropies_val.npy'
        
        i_embs = np.load(i_emb_npy_file)
        o_embs = np.load(o_emb_npy_file)
        
        # all_dot_prods(i_embs, o_embs, embs_dot_prod_file)
        # verify_dot_prod(i_embs, o_embs, embs_dot_prod_file, index_1=112, index_2=45154, is_exp=False)
        # exp_dot_prods(embs_dot_prod_file, embs_exp_dot_prod_file)
        # verify_dot_prod(i_embs, o_embs, embs_exp_dot_prod_file, index_1=11212, index_2=24515, is_exp=True)
        
        # exp_dot_prods = np.load(embs_exp_dot_prod_file, mmap_mode='r')
        # print('Max exp dot prod:', np.max(exp_dot_prods))
        
        # normalisation_vector(embs_exp_dot_prod_file, norm_vector_file)
        # normalised_dot_prods(embs_exp_dot_prod_file, norm_vector_file, norm_exp_dot_prods_file, del_dot_prods_file=True)
        # verify_norm_dot_prod(embs_exp_dot_prod_file, norm_vector_file, norm_exp_dot_prods_file)
        
        avg_cross_ent_train = cross_entropy_all_words(lt_npy_train_counts_file, lt_npy_train_counts_norm_vector_file, norm_exp_dot_prods_file, cross_ent_train_file)
        avg_cross_ent_val = cross_entropy_all_words(lt_npy_val_counts_file, lt_npy_val_counts_norm_vector_file, norm_exp_dot_prods_file, cross_ent_val_file)
        
        print('Average train cross validation', avg_cross_ent_train)
        print('Average val cross validation', avg_cross_ent_val)
        
        # model_name = 'rand_init-no_syns-10e-voc1-emb300'
        plot_filepath = 'plots/' + model_name 
        
        lt_npy_val_cross_ents_hist_file = plot_filepath +  model_name + '_val_cross_ents'
        
        lcross_ent_train_file = 'model/rand_init-no_syns-10e-voc1-emb300/luke-cross_entropies_train.npy'
        
        print('Calculating Luke\'s cross entropy for', lt_npy_proc_train_data_file)
        
        lcross_ents_train = cross_ent_luke(lt_npy_proc_train_data_file, norm_exp_dot_prods_file, lcross_ent_train_file)
        lcross_ents_val = cross_ent_luke(lt_npy_proc_val_data_file, norm_exp_dot_prods_file, lcross_ent_train_file)
        
        print('Luke\'s cross entropy train:', lcross_ents_train)
        print('Luke\'s cross entropy val:', lcross_ents_val)
        
        cross_entropies.append([model_name, avg_cross_ent_train, avg_cross_ent_val, lcross_ents_train, lcross_ents_val])
        
        # plot_cross_ents_histogram(cross_ent_val_file, save_file=lt_npy_val_cross_ents_hist_file)
        
        
        ###### SECOND VERIFICATION
        # verify_norm_exp_dot_prod(i_embs, o_embs, norm_vector_file, norm_exp_dot_prods_file, index=53140)
        
    cross_ents_savefile = 'cross_ent_results_1.csv'
    
    with open(cross_ents_savefile, 'w+') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(cross_entropies)
    
    elapsed_time = time.time() - start_time
    print('\nTotal elapsed time: ', elapsed_time)