"""Defines constants used for data preprocessing.
"""

VOCAB_CHARS_FILE = "vocab.chars.npy"
TRAIN_SQUAD_FILE = "train-v2.0.json"
DEV_SQUAD_FILE = "dev-v2.0.json"
EXOBRAIN_KOREAN_FILE = "SQuAD_Korean_QA_dataset_wikipedia_339.xlsx"
TRAIN_SQUAD_KOREAN_FILE = "train-exobrain-korean.json"
DEV_SQUAD_KOREAN_FILE = "dev-exobrain-korean.json"

COVE_WEIGHTS_FOLDER = "cove_weights"
COVE_WEIGHT_NAMES = ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0',
    'bias_hh_l0', 'weight_ih_l0_reverse', 'weight_hh_l0_reverse',
    'bias_ih_l0_reverse', 'bias_hh_l0_reverse', 'weight_ih_l1',
    'weight_hh_l1', 'bias_ih_l1', 'bias_hh_l1', 'weight_ih_l1_reverse',
    'weight_hh_l1_reverse', 'bias_ih_l1_reverse', 'bias_hh_l1_reverse']

# Training data can be split into multiple batches of files in order to
# limit the size of data in memory at once. Adjust as necessary.
MAX_SAMPLES_PER_SPLIT = 100000
TRAIN_FOLDER_NAME = "train"
DEV_FOLDER_NAME = "dev"
CONTEXT_FILE_PATTERN = "context.%d.npy"
QUESTION_FILE_PATTERN = "question.%d.npy"
SPAN_FILE_PATTERN = "span.%d.npy"
WORD_IN_QUESTION_FILE_PATTERN = "word_in_question.%d.npy"
WORD_IN_CONTEXT_FILE_PATTERN = "word_in_context.%d.npy"
QUESTION_IDS_FILE_PATTERN = "question_ids.%d.npy"
QUESTION_IDS_TO_GND_TRUTHS_FILE_PATTERN = "question_ids_to_gnd_truths.%d"
CONTEXT_POS_FILE_PATTERN = "context.pos.%d.npy"
CONTEXT_NER_FILE_PATTERN = "context.ner.%d.npy"
QUESTION_POS_FILE_PATTERN = "question.pos.%d.npy"
QUESTION_NER_FILE_PATTERN = "question.ner.%d.npy"
QUESTION_IDS_TO_SQUAD_QUESTION_ID_FILE_PATTERN = "question_ids_to_squad_question_id.%d"
QUESTION_IDS_TO_PASSAGE_CONTEXT_FILE_PATTERN = "passage_context.%d"

VECTORS_URL = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
FASTTEXT_VECTORS_URL = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ko.vec"
WORD_VEC_DIM = 300
WORD_VEC_DIM_FASTTEXT = 300
MAX_WORD_LEN = 25
VECTOR_FILE = "glove.840B.300d.txt"
FASTTEXT_VECTOR_FILE = "fasttext.300d.txt"
VECTOR_ZIP_FILE = "glove.840B.300d.zip"
SQUAD_TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
SQUAD_TRAIN_FILE = "train-v2.0.json"
SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
SQUAD_DEV_FILE = "dev-v2.0.json"
SQUAD_KOREAN_TRAIN_URL = "https://drive.google.com/file/d/1koFAvff_RKKivF8UZTTAlQJ-Qz5aUtnw"
SQUAD_KOREAN_TRAIN_FILE = "train-exobrain-korean.json"
SQUAD_KOREAN_DEV_URL = "https://drive.google.com/file/d/1onXV-y15YqLsoFbMQlrS21OsZyh3TcKZ"
SQUAD_KOREAN_DEV_FILE = "dev-exobrain-korean.json"


EMBEDDING_FILE = "glove.embedding.npy"
FASTTEXT_EMBEDDING_FILE = "fasttext.embedding.npy"

VOCAB_FILE = "vocab.txt"

