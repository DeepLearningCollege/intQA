import tensorflow as tf

_NUMERICAL_STABILITY_EPSILON = 1e-8


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

        # # shape is [batch_size, num_iter_answer_pointer, num words in sentence]
        shape = [None, options.num_stochastic_answer_pointer_steps, options.max_ctx_length]
        shape_int = [-1, options.num_stochastic_answer_pointer_steps, options.max_ctx_length]
        # # TODO return these so that model can return handles
        sampled_start_pos_list = tf.placeholder(tf.float32, shape=shape, name='sampled_start_pos_list')
        sampled_end_pos_list = tf.placeholder(tf.float32, shape=shape, name='sampled_end_pos_list')
        greedy_start_pos_list = tf.placeholder(tf.float32, shape=shape, name='greedy_start_pos_list')
        greedy_end_pos_list = tf.placeholder(tf.float32, shape=shape, name='greedy_start_pos_list')
        reward = tf.placeholder(tf.float32, shape=shape[0:2], name='reward')
        reward = tf.expand_dims(reward, 2)
        input_term = [sampled_start_pos_list,
                      sampled_end_pos_list,
                      greedy_start_pos_list,
                      greedy_end_pos_list,
                      reward]

        if options.ce_only:
            return ce_loss, [ce_loss], input_term

        start_pos_list = [start_pos + _NUMERICAL_STABILITY_EPSILON for start_pos in start_pos_list]
        # before transpose [num_iter_answer_pointer, batch_size, num words in sentence]
        start_pos_list = tf.log(start_pos_list)
        # after transpose [batch_size, num_iter_answer_pointer, num words in sentence]
        start_pos_list = tf.transpose(start_pos_list, [1, 0, 2])

        end_pos_list = [end_pos + _NUMERICAL_STABILITY_EPSILON for end_pos in end_pos_list]
        # before transpose [num_iter_answer_pointer, batch_size, num words in sentence]
        end_pos_list = tf.log(end_pos_list)
        # after transpose [batch_size, num_iter_answer_pointer, num words in sentence]
        end_pos_list = tf.transpose(end_pos_list, [1, 0, 2])

        if not options.dcrl:
            sampled_start_pos_log_p = sampled_start_pos_list * start_pos_list * reward
            start_adv = tf.reduce_mean(-1 * tf.reduce_sum(sampled_start_pos_log_p, axis=[1, 2]))

            sampled_end_pos_log_p = sampled_end_pos_list * end_pos_list * reward
            end_adv = tf.reduce_mean(-1 * tf.reduce_sum(sampled_end_pos_log_p, axis=[1, 2]))

            adv = start_adv + end_adv
        else:
            sampled_positive_reward = tf.clip_by_value(reward, 0, 1)
            sampled_start_pos_log_p = sampled_start_pos_list * start_pos_list * sampled_positive_reward
            sampled_end_pos_log_p = sampled_end_pos_list * end_pos_list * sampled_positive_reward

            greeedy_positive_reward = tf.clip_by_value(-reward, 0, 1)
            greedy_start_pos_log_p = greedy_start_pos_list * start_pos_list * greeedy_positive_reward
            greedy_end_pos_log_p = greedy_end_pos_list * end_pos_list * greeedy_positive_reward

            start_adv = tf.reduce_mean(-1 * tf.reduce_sum(
                sampled_start_pos_log_p + greedy_start_pos_log_p, axis=[1, 2]))
            end_adv = tf.reduce_mean(-1 * tf.reduce_sum(
                sampled_end_pos_log_p + greedy_end_pos_log_p, axis=[1, 2]))

            adv = start_adv + end_adv

        # loss를 계산한다.
        rl_loss = adv

        # # ce_loss와 rl_loss의 조합을 위해 텐서플로우 variable로 convert한다
        sigma_ce = tf.get_variable('sigma_a', shape=(), initializer=tf.random_normal_initializer, dtype=tf.float32)
        sigma_rl = tf.get_variable('sigma_b', shape=(), initializer=tf.random_normal_initializer, dtype=tf.float32)
        sq_sigma_ce = tf.square(sigma_ce)
        sq_sigma_rl = tf.square(sigma_rl)

        norm_ce_loss = ce_loss / (2 * sq_sigma_ce)
        norm_rl_loss = rl_loss / (2 * sq_sigma_rl)

        # loss를위 한 모든 term을 계산한다.
        log_sigma_ce = tf.log(sq_sigma_ce + _NUMERICAL_STABILITY_EPSILON)
        log_sigma_rl = tf.log(sq_sigma_rl + _NUMERICAL_STABILITY_EPSILON)

        # loss를 조합한다.
        loss = norm_ce_loss + norm_rl_loss + log_sigma_ce + log_sigma_rl
        output_term = [rl_loss,
                       sampled_start_pos_log_p,
                       start_adv,
                       sampled_end_pos_log_p,
                       end_adv,
                       [norm_ce_loss, norm_rl_loss, log_sigma_ce, log_sigma_rl]
                       ]
        return loss, output_term, input_term
