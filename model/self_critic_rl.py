import tensorflow as tf

from train.evaluation_functions import get_best_start_and_end, get_sampled_start_and_end, f1_score
from train.sentence_util import find_question_sentence


def self_critic_rl(sess, options, sq_dataset, start_span_probs, end_span_probs, data_index_iterator, qst_iterator, ce_loss):
    """
    applies self-critical-sequence-training to optimize
    f1 between distribution sampled answer and greedy sampled answer. i.e) F1(SAMPLED,GT) - F1(GREEDY,GT)
    0.5 * ce_loss / delta_ce ** + 0.5 * rl_loss / delta_rl

    :param options:
    :param start_span_probs:
    :param end_span_probs:
    :return: loss, ce_loss, rl_loss
    """

    if not options.self_critic_type:
        return ce_loss, ce_loss, ce_loss
    # 각 context 단어별 distribution 에서 loss를 샘플한다
    sampled_start, sampled_end = get_sampled_start_and_end(sess, start_span_probs, end_span_probs, options)
    greedy_start, greedy_end = get_best_start_and_end(start_span_probs, end_span_probs, options)

    # 각 예측 샘플과 정답에서 문장을 가져온다.
    example_index = data_index_iterator
    question_word_ids = qst_iterator
    question = find_question_sentence(question_word_ids, sq_dataset.vocab)
    sampled_prediction_str = sq_dataset.get_sentence(example_index, sampled_start, sampled_end)
    greedy_prediction_str = sq_dataset.get_sentence(example_index, greedy_start, greedy_end)
    ground_truth_str = sq_dataset.get_sentences_for_all_gnd_truths(example_index)

    # f1 을 계산한다.
    sampled_f1 = f1_score(sampled_prediction_str, ground_truth_str)
    greedy_f1 = f1_score(greedy_prediction_str, ground_truth_str)

    # loss를 계산한다.
    rl_loss = None
    if options.self_critic_type == 'SCST':
        rl_loss = -(sampled_f1 - greedy_f1)
    elif options.self_critic_type == 'DCRL':
        rl_loss = -(abs(sampled_f1 - greedy_f1))

    with tf.variable_scope("self_critic_rl"):
        # ce_loss와 rl_loss의 조합을 위해 텐서플로우 variable로 convert한다
        tf_rl_loss = tf.variable([rl_loss])
        double = tf.constant(2, dtype=tf.float32)
        sigma_ce = tf.get_variable("sigma_a", shape=[1, 1], dtype=tf.float32)
        sigma_rl = tf.get_variable("sigma_b", shape=[1, 1], dtype=tf.float32)
        sq_sigma_ce = tf.square(sigma_ce)
        sq_sigma_rl = tf.square(sigma_rl)

        # loss를위 한 모든 term을 계산한다.
        norm_ce_loss = tf.multiply(tf.divide(1, tf.multiply(double, sq_sigma_ce)), ce_loss)
        norm_rl_loss = tf.multiply(tf.divide(1, tf.multiply(double, sq_sigma_rl)), tf_rl_loss)
        log_sigma_ce = tf.log(sigma_ce)
        log_sigma_rl = tf.log(sigma_rl)

        # loss를 조합한다.
        # tf.Variable(name="learning_rate", initial_value=
        #        self.options.learning_rate, trainable=False, dtype=tf.float32)
        all_loss_terms = tf.variable([norm_ce_loss, norm_rl_loss, log_sigma_ce, log_sigma_rl])
        loss = tf.reduce_sum(all_loss_terms)
        return loss, ce_loss, rl_loss
