"""Creates way to make a word embedding file that is usable by numpy.
"""

import numpy as np
import operator
import os
import io
import preprocessing.constants as constants
import preprocessing.chars as chars

def load_word_embeddings_including_unk_and_padding(options):
    if options.exobrain_korean_dataset:
        embeddings = np.load(os.path.join(options.data_dir,
            constants.FASTTEXT_EMBEDDING_FILE))
    else:
        embeddings = np.load(os.path.join(options.data_dir,
            constants.EMBEDDING_FILE))
    # Add in all 0 embeddings for the padding and unk vectors
    return np.concatenate((embeddings, np.zeros((2, embeddings.shape[1]))))

def load_word_char_embeddings(options):
    return np.load(os.path.join(options.data_dir, constants.VOCAB_CHARS_FILE))

def _get_line_count(filename):
    num_lines = 0
    with open(filename, "r", encoding="utf-8") as f:
        for _ in f:
            num_lines += 1
    return num_lines

def _get_line_count_from_fasttext_vec(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    return n

def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0.0]*(target_len - len(some_list))

def split_vocab_and_embedding(options, data_dir, download_dir):
    input_file = os.path.join(download_dir, constants.VECTOR_FILE)
    embedding_output_file = os.path.join(data_dir, constants.EMBEDDING_FILE)
    vocab_output_file = os.path.join(data_dir, constants.VOCAB_FILE)
    vocab_chars_output_file = os.path.join(data_dir, constants.VOCAB_CHARS_FILE)
    if all([os.path.exists(f) for f in 
        [embedding_output_file, vocab_output_file, vocab_chars_output_file]]):
        print("Word embedding and vocab files already exist")
        return
    print("Creating NumPy word embedding file and vocab files")

    num_lines = _get_line_count(input_file)
    print("Vocab size: %d" % num_lines)
    # Include 4 entries for bos/eos/unk/pad (they will all be left as 0 vectors).
    embedding = np.zeros((num_lines + 4, constants.WORD_VEC_DIM), dtype=np.float32)
    vocab_o_file = open(vocab_output_file, "w", encoding="utf-8")
    # Get IDs for the total vocab, not just the words. This includes
    # the bos/eos/unk/pad.
    vocab_chars = np.zeros((num_lines + 4, constants.MAX_WORD_LEN), dtype=np.uint8)
    i_file = open(input_file, "r", encoding="utf-8")
    i = 0
    char_counts = {}
    vocab_list = []
    for line in i_file:
        idx = line.index(" ")
        word = line[:idx]
        vocab_list.append(word)
        for c in word:
            if c in char_counts:
                char_counts[c] += 1
            else:
                char_counts[c] = 1
        vocab_o_file.write(word + "\n")
        embedding[i] = np.fromstring(line[idx + 1:], dtype=np.float32, sep=' ')
        i += 1
        if i % 10000 == 0 or i == num_lines:
            print("Processed %d of %d (%f percent done)" % (i, num_lines, 100 * float(i) / float(num_lines)), end="\r")
    sorted_chars = sorted(char_counts.items(), key=operator.itemgetter(1),
        reverse=True)
    frequent_chars = dict((x[0], i) for i, x in enumerate(
        sorted_chars[:chars.MAX_CHARS]))
    print("")
    print("Creating word character data")
    for z in range(len(vocab_list)):
        word = vocab_list[z]
        vocab_chars[z, 0] = chars.CHAR_BOW_ID
        for zz in range(constants.MAX_WORD_LEN - 1):
            insert_index = zz + 1
            if zz >= len(word):
                vocab_chars[z, insert_index] = chars.CHAR_PAD_ID
            elif word[zz] not in frequent_chars:
                vocab_chars[z, insert_index] = chars.CHAR_UNK_ID
            else:
                vocab_chars[z, insert_index] = frequent_chars[word[zz]]
        vocab_chars[z, min(1 + len(word), constants.MAX_WORD_LEN - 1)] = \
            chars.CHAR_EOW_ID
    # The order of the following must match that of vocab.py
    vocab_chars[num_lines, :] = chars.CHAR_BOS_ID
    vocab_chars[num_lines + 1, :] = chars.CHAR_EOS_ID
    vocab_chars[num_lines + 2, :] = chars.CHAR_PAD_ID
    vocab_chars[num_lines + 3, :] = chars.CHAR_UNK_ID
    np.save(vocab_chars_output_file, vocab_chars)
    np.save(embedding_output_file, embedding)
    vocab_o_file.close()
    i_file.close()
    print("")
    print("Finished creating vocabulary and embedding file")

def split_vocab_and_embedding_fasttext(options, data_dir, download_dir):
    input_file = os.path.join(download_dir, constants.FASTTEXT_VECTOR_FILE)
    embedding_output_file = os.path.join(data_dir, constants.FASTTEXT_EMBEDDING_FILE)
    vocab_output_file = os.path.join(data_dir, constants.VOCAB_FILE)
    vocab_chars_output_file = os.path.join(data_dir, constants.VOCAB_CHARS_FILE)
    if all([os.path.exists(f) for f in
        [embedding_output_file, vocab_output_file, vocab_chars_output_file]]):
        print("Word embedding and vocab files already exist")
        return
    print("Creating NumPy word embedding file and vocab files")

    num_lines = _get_line_count_from_fasttext_vec(input_file)
    print("Vocab size: %d" % num_lines)
    # Include 4 entries for bos/eos/unk/pad (they will all be left as 0 vectors).
    embedding = np.zeros((num_lines + 4, constants.WORD_VEC_DIM_FASTTEXT), dtype=np.float32)
    vocab_o_file = open(vocab_output_file, "w", encoding="utf-8")
    # Get IDs for the total vocab, not just the words. This includes
    # the bos/eos/unk/pad.
    vocab_chars = np.zeros((num_lines + 4, constants.MAX_WORD_LEN), dtype=np.uint8)

    i_file = io.open(input_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    i = 0
    char_counts = {}
    vocab_list = []
    for line in i_file:
        if i in [0, 879129, 879130, 879131, 879132]:
            i += 1
            continue
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        vocab_list.append(word)
        for c in word:
            if c in char_counts:
                char_counts[c] += 1
            else:
                char_counts[c] = 1
        vocab_o_file.write(word + "\n")
        embedding[i] = np.fromiter(pad_or_truncate(tokens[1:], constants.WORD_VEC_DIM_FASTTEXT), dtype=np.float32)
        i += 1
        if i % 10000 == 0 or i == num_lines:
            print("Processed %d of %d (%f percent done)" % (i, num_lines, 100 * float(i) / float(num_lines)), end="\r")
    sorted_chars = sorted(char_counts.items(), key=operator.itemgetter(1),
        reverse=True)
    frequent_chars = dict((x[0], i) for i, x in enumerate(
        sorted_chars[:chars.MAX_CHARS]))
    print("")
    print("Creating word character data")
    for z in range(len(vocab_list)):
        word = vocab_list[z]
        vocab_chars[z, 0] = chars.CHAR_BOW_ID
        for zz in range(constants.MAX_WORD_LEN - 1):
            insert_index = zz + 1
            if zz >= len(word):
                vocab_chars[z, insert_index] = chars.CHAR_PAD_ID
            elif word[zz] not in frequent_chars:
                vocab_chars[z, insert_index] = chars.CHAR_UNK_ID
            else:
                vocab_chars[z, insert_index] = frequent_chars[word[zz]]
        vocab_chars[z, min(1 + len(word), constants.MAX_WORD_LEN - 1)] = \
            chars.CHAR_EOW_ID
    # The order of the following must match that of vocab.py
    vocab_chars[num_lines, :] = chars.CHAR_BOS_ID
    vocab_chars[num_lines + 1, :] = chars.CHAR_EOS_ID
    vocab_chars[num_lines + 2, :] = chars.CHAR_PAD_ID
    vocab_chars[num_lines + 3, :] = chars.CHAR_UNK_ID
    np.save(vocab_chars_output_file, vocab_chars)
    np.save(embedding_output_file, embedding)
    vocab_o_file.close()
    i_file.close()
    print("")
    print("Finished creating vocabulary and embedding file")