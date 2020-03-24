###
#
# Data Processing Support Functions
#
###

import spacy
from nltk import pos_tag
from nltk.corpus import wordnet as wn
import re
import os.path


def process_augm_data(text_file, dataset_file, augm_dataset_file, doc_num=-1, ctx_size=5 ):
    
    """
    Generate datasets in the Skip Gram format
    (Mikolov et al., 2013): word pairs
    consisting of a centre or 'focus' word and
    the words within its context window
    A 'natural' and an augmented dataset are 
    constructed from a text file. The natural
    dataset cycles through each word and treats
    it as a focus word to construct the
    corresponding word pairs. The augmented
    dataset replaces each focus word (only
    adjectives, adverbs, nouns, and verbs)
    with its synonyms from WordNet
    
    The datasets are saved to two CSV files
    with the following columns:
        Natural dataset:
        - 0 : focus_word
        - 1 : context_word
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
        - 5 : book_number
        
        Augmented dataset:
        - 0 : synonym
        - 1 : context_word
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
        - 5 : focus_word
        - 6 : book_number
    
    Requirements
    ------------
    import spacy
        spaCy NLP library
    from nltk import pos_tag
        NLTK POS tagger
    from nltk.corpus import wordnet as wn
        WordNet and FrameNet NLTK libraries
    import re
        regular expression library
    import os.path
        filepath functions
    
    Parameters
    ----------
    read_file : str
        path to raw text source file
    dataset_file : str
        path to dataset save file
    augm_dataset_file : str
        path to augmented dataset save file
    doc_num : int, optional
        document index (useful when
        processing multiple documents)
        (default: -1)
    ctx_size : int, optional
        context window size (default: 5)
    
    Returns
    -------
    bool
        False if file is too large to be
        processed, True otherwise
    
    TODO
    ----
    - Unifying upper/lowercase
    - Addressing Named Entities (token.ent_iob_, token.ent_type_)
    """
    num_lines = 0
    num_words = 0
    full_counter = Counter()
    
    vocabulary = []
    
    # Disable NER and categorisation to lighten processing
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    
    # If the dataset file does not exist, add a header
    if os.path.exists(dataset_file):
        dataset = []
    else:
        dataset = [['focus_word', 'context_word', 'sent_num', 'focus_index', 'context_position']]
    if os.path.exists(augm_dataset_file):
        augm_dataset = []
    else:
        augm_dataset = [['synonym', 'context_word', 'sent_num', 'focus_index', 'context_position', 'focus_word']]
    
    # Open the file with UTF-8 encoding. If a character
    # can't be read, it gets replaced by a standard token
    with open(text_file, 'r', encoding='utf-8', errors='replace') as f:
        print('Cleaning and processing ', text_file)
        
        # Go through all sentences tokenised by spaCy
        # Word pairs are constrained to sentence appearances,
        # i.e. no inter-sentence word pairs
        for sent_num, raw_sent in enumerate(data.readlines()):
            sent = nlp(raw_sent)
            
            token_list = [token for token in sent]
            num_tokens = len(token_list)
            
            # Skip processing if sentence is only one word
            if len(token_list) > 1:
                for focus_i, token in enumerate(token_list):
                    t_pos = token.pos_
                    
                    # List of ignored tags
                    ignore_pos = ['PUNCT', 'SYM', 'X']
                    
                    # Temporary list of generated pairs for the current
                    # focus word. Will be used later when searching for
                    # synonyms for the focus word
                    word_pairs = []
                    augment_pairs = []
                    
                    # Only process if focus word is not punctuation
                    # (PUNCT), symbol (SYM), or unclassified (X), and
                    # if token is not only a symbol (not caught by spaCy)
                    if (t_pos not in ignore_pos
                        and re.sub(r'[^\w\s]', '', token.text).strip() != ''):
                        
                        # BYPASSED: original formulation, sampling context
                        # size, from 1 to ctx_size
                        #context_size = random.randint(1, ctx_size)
                        context_size = ctx_size
                        
                        context_min = focus_i - context_size if (focus_i - context_size >= 0) else 0
                        
                        context_max = focus_i + context_size if (focus_i + context_size < num_tokens-1) else num_tokens-1
                        
                        focus_word = token.text.lower()
                        
                        # Go through every context word in the window
                        for ctx_i in range(context_min, context_max+1):
                            # Check that context index is not the same as
                            # focus, that the context word is not in our
                            # 'ignore' list, and that the context is not
                            # white space or punctuation
                            if (ctx_i != focus_i
                                and token_list[ctx_i].pos_ not in ignore_pos
                                and re.sub(r'[^\w\s]', '', token_list[ctx_i].text).strip() != ''):
                                # Changing everything to lower case
                                # A more principled approach would uppercase
                                # named entities such as persons, companies
                                # or countries:
                                #   if token.ent_iob_ != 'O':
                                #       token.text.capitalize()
                                context_word = token_list[ctx_i].text.lower()
                                
                                ctx_pos = ctx_i - focus_i
                                
                                # If passes all checks (context different
                                # from target and neither tagged as)
                                word_pairs.append([focus_word, context_word,sent_num, focus_i, ctx_pos])
                                
                        # If word_pairs is not empty, that means there is
                        # at least one valid word pair. For every non-stop focus
                        # word in these pairs, augment the dataset with external
                        # knowledge bases
                        if len(word_pairs) > 0 and not token.is_stop:
                            
                            # Convert to list of text words for NLTK
                            text_list = [token.text for token in token_list]
                            # POS tag with NLTK
                            nltk_pos = [pos_tag(text_list)]
                            
                            # We create nltk_pos with a single sentence
                            # so we can access it directly. We are accessing:
                            #   - Sentence 0
                            #   - Word number focus index
                            #   - POS tag (second column)
                            nltk_pos_tag = nltk_pos[0][focus_i][1]
                            
                            # If the POS tags of spaCy and NLTK agree
                            # then continue with the augmentation
                            if token.tag_ == nltk_pos_tag:
                                #print('Tags agree for ', nltk_pos[0][focus_i][1])
                                
                                # Convert universal POS tags to WordNet types
                                # https://universaldependencies.org/u/pos/all.html
                                # (skip proper nouns)
                                wn_tag_dict = {
                                    'ADJ': wn.ADJ,
                                    'ADV': wn.ADV,
                                    'NOUN': wn.NOUN,
                                    #'PROPN': wn.NOUN,
                                    'VERB': wn.VERB
                                }
                                
                                # If the POS tag is part of the
                                # pre-specified tags
                                if token.pos_ in wn_tag_dict:
                                    synsets = wn.synsets(focus_word, wn_tag_dict[token.pos_])
                                    
                                    # Keep track of accepted synonyms,
                                    # to avoid adding the same synonym
                                    # multiple times to the dataset
                                    accepted_synonyms = []
                                    
                                    # Cycle through the possible synonym
                                    # sets in WordNet
                                    for syn_num, syn in enumerate(synsets):
                                        # Cycle through all the lemmas in
                                        # every synset
                                        for lem in syn.lemmas():
                                            # Get the synonym in lowercase
                                            synonym = lem.name().lower()
                                            
                                            # Removes multi-word synonyms
                                            # as well as repeated synonyms
                                            if not re.search('[-_]+', synonym) and focus_word != synonym and synonym not in accepted_synonyms:
                                                accepted_synonyms.append(synonym)
                                                
                                                for fw, c, sn, fi, cp, _ in word_pairs:
                                                    augment_pairs.append([synonym, c, sn, fi, cp, fw])
                    
                    if len(word_pairs) > 0:
                        dataset.extend(word_pairs)
                    
                    if len(augment_pairs) > 0:
                        augm_dataset.extend(augment_pairs)
                    

    if len(dataset) > 0: print('Original dataset: ', len(dataset), len(dataset[0]))
    if len(augm_dataset) > 0: print('Augmented dataset: ', len(augm_dataset), len(augm_dataset[0]))
    
    # Look for the dataset file, if it doesn't exist create it ('a+')
    with open(dataset_file, 'a+', newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerows(dataset)
        
    with open(augm_dataset_file, 'a+', newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerows(augm_dataset)      
    
    return True
