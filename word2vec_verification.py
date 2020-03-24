###
#
# Word2Vec Verification
#
# Comparison between learned embeddings and
# "true" language model
#
###

from skipgram.train import train_augm_w2v

if __name__ == '__main__':
    proc_data_dir = 'data/processed_data/'
    vocab_dir = 'data/vocabulary/'

    voc_threshold = 1
    vocab_file = vocab_dir + 'vocab_' + str(voc_threshold) + '.csv'

    ctx_size = 5
    skipgram_filename_lt = proc_data_dir + 'skipgram_data_ctx' + str(ctx_size) + '_lt_v' + str(voc_threshold)

    lt_proc_train_data_file = skipgram_filename_lt + '_train_1.csv'
    lt_proc_train_syns_data_file = skipgram_filename_lt + '_train_syns_1.csv'
    lt_proc_val_data_file = skipgram_filename_lt + '_val_1.csv'
    
    model_name = 'rand_init-syns-10e-voc1-emb300'
    model_directory = 'model/' + model_name + '/'
    embs_npy_path = model_directory + model_name
    
    train_augm_w2v(lt_proc_train_data_file, vocab_file, model_name, lt_proc_val_data_file, syns_file=lt_proc_train_syns_data_file, embedding_size=300, epochs=10, batch_size=10, num_neg_samples=5, learning_rate=0.01, w2v_init=False, w2v_path=None, emb_npy_file=embs_npy_path, data_augmentation_ratio=.25)