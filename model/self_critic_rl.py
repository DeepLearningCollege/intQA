import tensorflow as tf

def self_critic_rl(options, ce_loss, start_pos_list, end_pos_list):
    """
    applies self-critical-sequence-training to optimize
    f1 between distribution sampled answer and greedy sampled answer. i.e) F1(SAMPLED,GT) - F1(GREEDY,GT)
    0.5 * ce_loss / delta_ce ** + 0.5 * rl_loss / delta_rl

    :param options:
    :param start_span_probs:
    :param end_span_probs:
    :return: loss, ce_loss, rl_loss
    """
    with tf.variable_scope("self_critic_rl"):
        # shape is [batch_size, num_iter_answer_pointer, num words in sentence]
        shape = [None, options.num_stochastic_answer_pointer_steps, options.max_ctx_length]
        # TODO return these so that model can return handles
        sampled_start_pos_list = \
            tf.placeholder(tf.int8, shape=shape, name='sampled_start_pos_list')
        sampled_end_pos_list = \
            tf.placeholder(tf.int8, shape=shape, name='sampled_end_pos_list')
        greedy_start_pos_list = \
            tf.placeholder(tf.int8, shape=shape, name='greedy_start_pos_list')
        greedy_end_pos_list = \
            tf.placeholder(tf.int8, shape=shape, name='greedy_start_pos_list')
        reward = \
            tf.placeholder(tf.int8, shape=shape[0:2], name='reward')

        sampled_start_pos_log_p = sampled_start_pos_list * tf.log(start_pos_list) * reward
        sampled_start_adv = tf.reduce_mean(tf.reduce_sum(-1 * sampled_start_pos_log_p, axis=1))
        sampled_end_pos_log_p = sampled_end_pos_list * tf.log(end_pos_list) * reward
        sampled_end_adv = tf.reduce_mean(tf.reduce_sum(-1 * sampled_end_pos_log_p, axis=1))
        sampled_adv = sampled_start_adv + sampled_end_adv

        greedy_start_pos_log_p = greedy_start_pos_list * tf.log(start_pos_list) * reward
        greedy_start_adv = tf.reduce_mean(tf.reduce_sum(-1 * greedy_start_pos_log_p, axis=1))
        greedy_end_pos_log_p = greedy_end_pos_list * tf.log(end_pos_list) * reward
        greedy_end_adv = tf.reduce_mean(tf.reduce_sum(-1 * greedy_end_pos_log_p, axis=1))
        greedy_adv = greedy_start_adv + greedy_end_adv

        # loss를 계산한다.
        # rl_loss = None
        # if options.self_critic_type == 'SCST':
        #     rl_loss = -(sampled_f1 - greedy_f1)
        # elif options.self_critic_type == 'DCRL':
        #     rl_loss = -(abs(sampled_f1 - greedy_f1))
        rl_loss = greedy_adv - sampled_adv

        # ce_loss와 rl_loss의 조합을 위해 텐서플로우 variable로 convert한다
        tf_rl_loss = tf.Variable([rl_loss])

        sigma_ce = tf.get_variable('sigma_a', shape=[1, 1], dtype=tf.float32)
        sigma_rl = tf.get_variable('sigma_b', shape=[1, 1], dtype=tf.float32)

        sq_sigma_ce = tf.square(sigma_ce)
        sq_sigma_rl = tf.square(sigma_rl)

        double = tf.constant(2, dtype=tf.float32)
        norm_ce_loss = tf.multiply(tf.divide(1, tf.multiply(double, sq_sigma_ce)), ce_loss)
        norm_rl_loss = tf.multiply(tf.divide(1, tf.multiply(double, sq_sigma_rl)), tf_rl_loss)

        # loss를위 한 모든 term을 계산한다.
        log_sigma_ce = tf.log(sigma_ce)
        log_sigma_rl = tf.log(sigma_rl)

        # loss를 조합한다.
        # tf.Variable(name="learning_rate", initial_value=
        #        self.options.learning_rate, trainable=False, dtype=tf.float32)
        loss = tf.add_n([norm_ce_loss, norm_rl_loss, log_sigma_ce, log_sigma_rl])
        return loss, sampled_start_pos_list, sampled_end_pos_list, greedy_start_pos_list, greedy_end_pos_list, reward