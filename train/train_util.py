"""Functions to help with training and evaluation.
"""

import tensorflow as tf


def get_eval_feed_dict(squad_data, options, towers, is_train):
    feed_dict = get_feed_dict(squad_data, options, towers, is_train=is_train,
                              use_dropout=False)
    for i in range(len(towers)):
        tower = towers[i]
        feed_dict[tower.get_keep_prob_placeholder()] = 1
        feed_dict[tower.get_input_keep_prob_placeholder()] = 1
        feed_dict[tower.get_rnn_keep_prob_placeholder()] = 1
    return feed_dict


def get_ce_partial_run_args(squad_data, options, towers, common_run_ops):
    if len(towers) < 1:
        raise Exception("There are no models in the list of towers to train")
    towers_run_ops = []
    feed_dict_keys = []

    for tower in towers:
        tower_run_ops = [
            # tower.ce_loss,
            tf.constant(1.0),
            # TODO remove the following line
            tower.get_start_span_probs(),
            # TODO remove the following line
            tower.get_end_span_probs(),
            tower.start_pos_list,
            tower.end_pos_list,
            tower.get_data_index_iterator(),
            tower.get_qst()
        ]
        towers_run_ops.append(tower_run_ops)

        feed_dict_keys.append(tower.get_keep_prob_placeholder())
        feed_dict_keys.append(tower.get_input_keep_prob_placeholder())
        feed_dict_keys.append(tower.get_rnn_keep_prob_placeholder())
        feed_dict_keys.append(tower.get_use_dropout_placeholder())
    feed_dict_keys.append(squad_data.get_iterator_handle())
    # gradients_summary 는 scrl_loss 가 반영되어야 계산하기 때문에 못씀 ㅠㅠ.
    # common_run_ops = [loss_summary, gradients_summary]
    run_ops = [towers_run_ops, common_run_ops]
    return run_ops, feed_dict_keys


def get_feed_dict(squad_data, options, towers, is_train, use_dropout):
    if len(towers) < 1:
        raise Exception("There are no models in the list of towers to train")
    examples_per_tower = int(options.batch_size / len(towers))
    feed_dict = {}
    for i in range(len(towers)):
        tower = towers[i]
        feed_dict[tower.get_keep_prob_placeholder()] = 1 if not use_dropout \
            else 1 - options.dropout
        feed_dict[tower.get_input_keep_prob_placeholder()] = 1 if not use_dropout \
            else 1 - options.input_dropout
        feed_dict[tower.get_rnn_keep_prob_placeholder()] = 1 if not use_dropout \
            else 1 - options.rnn_dropout
        feed_dict[tower.get_use_dropout_placeholder()] = use_dropout
    train_handle = squad_data.get_train_handle()
    dev_handle = squad_data.get_dev_handle()
    tf_handle = squad_data.get_iterator_handle()
    feed_dict[tf_handle] = train_handle if is_train else dev_handle
    return feed_dict


def get_train_feed_dict(squad_data, options, towers):
    return get_feed_dict(squad_data, options, towers, is_train=True, use_dropout=True)


def get_scrl_partial_run_args(squad_data,
                              options,
                              towers,
                              common_ops):
    if len(towers) < 1:
        raise Exception("There are no models in the list of towers to train")

    towers_run_ops = []
    feed_dict_keys = []
    for tower_idx, tower in enumerate(towers):
        towers_run_ops.append([
            towers[tower_idx].loss,
            towers[tower_idx].rl_loss,
            towers[tower_idx].ce_loss
        ])
        # TODO unfold across batch index??
        feed_dict_keys.append(tower.sampled_start_pos_list)
        feed_dict_keys.append(tower.sampled_end_pos_list)
        feed_dict_keys.append(tower.greedy_start_pos_list)
        feed_dict_keys.append(tower.greedy_end_pos_list)
        feed_dict_keys.append(tower.reward)
    run_ops = [towers_run_ops, common_ops]
    return run_ops, feed_dict_keys


def get_scrl_train_feed_dict(squad_data, options, towers,
                             greedy_start_one_hots,
                             greedy_end_one_hots,
                             sampled_start_one_hots,
                             sampled_end_one_hots,
                             rewards):
    if len(towers) < 1:
        raise Exception("There are no models in the list of towers to train")
    examples_per_tower = int(options.batch_size / len(towers))
    feed_dict = {}
    for tower_idx, tower in enumerate(towers):
        # TODO unfold across batch index??
        feed_dict[tower.sampled_start_pos_list] = sampled_start_one_hots[tower_idx]
        feed_dict[tower.sampled_end_pos_list] = sampled_end_one_hots[tower_idx]
        feed_dict[tower.greedy_start_pos_list] = greedy_start_one_hots[tower_idx]
        feed_dict[tower.greedy_end_pos_list] = greedy_end_one_hots[tower_idx]
        feed_dict[tower.reward] = rewards[tower_idx]
    return feed_dict


def get_dev_feed_dict(squad_data, options, towers):
    return get_feed_dict(squad_data, options, towers, is_train=False, use_dropout=False)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, var in grad_and_vars:
            if g is None:
                print("g", g)
                print("var", var)
                raise Exception("Programmer error -- some variable isn't used towards"
                                + " the loss")
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
