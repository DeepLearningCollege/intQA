import os
import tensorflow as tf

from preprocessing.save_cove_weights import save_cove_weights
from preprocessing.create_train_data import DataParser
from preprocessing.download_data import download_data
from preprocessing.download_fasttext_data import download_fasttext_data
from preprocessing.embedding_util import split_vocab_and_embedding
from preprocessing.s3_util import maybe_upload_data_files_to_s3
from flags import get_options_from_flags

def main(_):
    options = get_options_from_flags()
    data_dir = options.data_dir
    download_dir = options.download_dir

    # 디렉토리 생성
    # data_dir = Pretrained 된 Word-Embedding Vector를 다운로드 받을 디렉토리
    # download_dir = SQuAD DataSet를 다운로드 받을 디렉토리
    for d in [data_dir, download_dir]:
        # 디렉토리를 재귀적으로 생성하며 디렉토리가 이미 존재하면 예외를 발생시키지 않음.
        os.makedirs(d, exist_ok=True)

    # 데이터 다운로드 단계
    # 첫째, GloVe vectors를 https://nlp.stanford.edu/projects/glove/ 로 부터 다운로드 받음.
    # 둘째, SQuAD Dataset 다운로드 받음
    if options.word_embedding_model_type == "fasttext":
        download_fasttext_data(download_dir)
    else:
        download_data(download_dir)
    #
    split_vocab_and_embedding(options, data_dir, download_dir)
    if options.exobrain_korean_dataset:
        DataParser(data_dir, download_dir).exobrain_korean_dataset_excel_to_squad_json(options)
    DataParser(data_dir, download_dir).create_train_data(options)
    if options.use_cove_vectors:
        save_cove_weights(options)
    maybe_upload_data_files_to_s3(options)

if __name__ == "__main__":
    tf.app.run()
