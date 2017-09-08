import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import sys
import cv2
import random

sys.path.insert(0, './')


def get_init_restorer(bn=False, vc=False):
    """Returns a function run by the chief worker to warm-start the training."""
    #checkpoint_exclude_scopes = ['vgg_16/fc8']
    checkpoint_exclude_scopes = []
    if not bn:
        checkpoint_exclude_scopes.append('BatchNorm')  # restore bn params
    if not vc:
        checkpoint_exclude_scopes.append('vc_centers')  # restore bn params

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    # for var in slim.get_model_variables():
    for var in tf.global_variables():
        excluded = False
        for exclusion in exclusions:
            if exclusion in var.op.name:
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    # variables_to_restore += [var for var in tf.global_variables() if 'vc_centers' in var.op.name]  # always restore VC
    return tf.train.Saver(variables_to_restore)


# copied from tf tutorial
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
        is_none = False
        for g, _ in grad_and_vars:
            if g is None:
                is_none = True
                continue
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        if is_none:
            continue
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        # if 'conv1' not in v.op.name:  # TODO
        average_grads.append(grad_and_var)
    return average_grads


def process_image(img, path, augment):
    scale_size = 224
    vgg_mean = np.float32([[[104., 117., 124.]]])
    background = np.array([[[255., 255., 255.]]])

    assert img is not None
    if len(img.shape)==2:
        h,w=img.shape
        c=1
    else:
        h, w, c = img.shape
    if c == 4:
        alpha = np.reshape(img[:, :, 3], (h, w, 1)) / np.float32(255)
        mixed_img = np.float32(img[:, :, 0:3]) * alpha + np.float32(background) * (1 - alpha)
        gray = cv2.cvtColor(mixed_img, cv2.COLOR_BGR2GRAY)
        mask = img[:, :, 3]
        h_temp = np.nonzero(np.amax(mask, axis=1))[0]
        w_temp = np.nonzero(np.amax(mask, axis=0))[0]
        h_size = max(h_temp[-1] + 1 - h // 2, h // 2 - h_temp[0]) * 2
        w_size = max(w_temp[-1] + 1 - w // 2, w // 2 - w_temp[0]) * 2
        obj_size = max(h_size, w_size)
        base_scale = 1.1
    elif c == 3:
        gray=img
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        obj_size = max(h, w)
        base_scale = 1.0
    elif c == 1:
        gray = np.float32(img)
        obj_size = max(h, w)
        base_scale = 1.0
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # if img is None:
    #     print('Image is None: ' + path)
    #     img = np.zeros((540, 960, 4))
    # h, w, c = img.shape
    # if not c == 4:
    #     print('Not 4 channels: ' + path)
    # if np.amax(img[:, :, :]) == 0:
    #     print('All channels are zero: ' + path)
    #     mixed_img = np.float32(img[:, :, 0:3])
    #     obj_size = max(h, w)
    # else:
    #     alpha = np.reshape(img[:, :, 3], (h, w, 1)) / np.float32(255)
    #     mixed_img = np.float32(img[:, :, 0:3]) * alpha + np.float32(background) * (1 - alpha)
    #
    #     mask = np.amax(img, axis=2)
    #     h_temp = np.nonzero(np.amax(mask, axis=1))[0]
    #     w_temp = np.nonzero(np.amax(mask, axis=0))[0]
    #     h_size = max(h_temp[-1] + 1 - h // 2, h // 2 - h_temp[0]) * 2
    #     w_size = max(w_temp[-1] + 1 - w // 2, w // 2 - w_temp[0]) * 2
    #     obj_size = max(h_size, w_size)
    # gray = cv2.cvtColor(mixed_img, cv2.COLOR_BGR2GRAY)
    
    #gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if augment == 0:
        rotation_degree = random.random() * 0.0
        scale_border = random.random() * 0.0 + base_scale
        delta_x = (random.random() * 2 - 1) * 0.0
        delta_y = (random.random() * 2 - 1) * 0.0
        noise = np.random.randn(scale_size, scale_size, 3) * 0.0
    elif augment == 1:
        rotation_degree = random.random() * 360
        scale_border = random.random() * 0.2 + base_scale
        delta_x = (random.random() * 2 - 1) * 0.1
        delta_y = (random.random() * 2 - 1) * 0.1
        noise = np.random.randn(scale_size, scale_size, 3) * 0.0
    else:
        rotation_degree = random.random() * 360
        scale_border = random.random() * 0.2 + base_scale
        delta_x = (random.random() * 2 - 1) * 0.1
        delta_y = (random.random() * 2 - 1) * 0.1
        noise = np.random.randn(scale_size, scale_size, 3) * 0.0

    scale = scale_size / float(obj_size) / scale_border
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), rotation_degree, scale)
    matrix[0, 2] += scale_size // 2 - w // 2 + delta_x * scale_size
    matrix[1, 2] += scale_size // 2 - h // 2 + delta_y * scale_size
    affine_img = cv2.warpAffine(gray, matrix, (scale_size, scale_size), borderMode=cv2.BORDER_CONSTANT,
                                borderValue=np.squeeze(background))
    return affine_img - vgg_mean + noise,scale_size // 2-int(scale*w//2),scale_size // 2-int(scale*h//2)


def dis_cosine(features):
    """Calculate the distance between two tensor along the first dimension."""
    inner_product = tf.matmul(features, tf.transpose(features))
    norm_vector = tf.reshape(tf.norm(features, axis=1), [-1, 1])
    norm_matrix = tf.matmul(norm_vector, tf.transpose(norm_vector))
    norm_matrix = tf.maximum(norm_matrix, 0.001)
    cos = tf.truediv(inner_product, norm_matrix)
    dis_pairwise = tf.reshape(tf.ones(cos.shape) - cos, [-1])
    return dis_pairwise


def layer_rank_hard_loss(features, t1_idx, t2_idx, param_m, param_neg_num, param_hard_ratio, param_rand_ratio):
    with tf.name_scope('loss_layer') as scope:
        # pairwise distance metric
        dis_pairwise = dis_cosine(features)

        # pick positive pairs and negative pairs
        t1 = tf.gather(dis_pairwise, t1_idx)
        t2 = tf.gather(dis_pairwise, t2_idx)

        # triplet loss of a batch
        triplet_list = tf.maximum(tf.zeros(tf.shape(t1)), t1 - t2 + param_m * tf.ones(tf.shape(t1)))
        loss = tf.reduce_mean(tf.reshape(triplet_list, [-1]))

        # monitor accuracy
        accuracy = tf.reduce_mean(tf.to_float(tf.greater(t2, t1)))
        return loss, accuracy


def generate_ts(labels):
    batch_size = len(labels)
    # init t1 and t2
    t1_idx = np.empty([batch_size, batch_size - 2], dtype=np.int32)
    t2_idx = np.empty([batch_size, batch_size - 2], dtype=np.int32)
    for i in range(batch_size):
        if i % 2 == 0:
            t1_idx[i] = [i * batch_size + i + 1] * (batch_size - 2)
            t2_idx[i] = np.ones([batch_size - 2]) * i * batch_size + np.concatenate(
                [np.arange(0, i), np.arange(i + 2, batch_size)])
        else:
            t1_idx[i] = np.copy(t1_idx[i - 1])
            t2_idx[i] = np.ones([batch_size - 2]) * batch_size + np.copy(t2_idx[i - 1])
    # delete invalid pairs, though there should not be any
    for i in range(batch_size):
        if i % 2 == 0:
            for j in range(i):
                if labels[j] == labels[i]:
                    t1_idx[i][j] = 0
                    t2_idx[i][j] = 0
            for j in range(i + 2, batch_size):
                if labels[j] == labels[i]:
                    t1_idx[i][j - 2] = 0
                    t2_idx[i][j - 2] = 0
        else:
            for j in range(i - 1):
                if labels[j] == labels[i]:
                    t1_idx[i][j] = 0
                    t2_idx[i][j] = 0
            for j in range(i + 1, batch_size):
                if labels[j] == labels[i]:
                    t1_idx[i][j - 2] = 0
                    t2_idx[i][j - 2] = 0
    t1_idx = t1_idx[np.nonzero(t1_idx)]
    t2_idx = t2_idx[np.nonzero(t2_idx)]
    return t1_idx, t2_idx


def sub2ind(N, i, j):
    return i * N + j


def generate_ts(P, Q):
    t1_idx = np.empty([Q * (Q - 1) * P * (P - 1) * P, ], dtype=np.int32)
    t2_idx = np.empty([Q * (Q - 1) * P * (P - 1) * P, ], dtype=np.int32)
    idx = 0
    for c1 in range(Q):
        for c2 in range(Q):
            if c1 == c2:
                continue
            # two classes chosen
            for o1 in range(P):
                for o2 in range(P):
                    if o1 == o2:
                        continue
                    # two objects chosen from c1
                    for o3 in range(P):
                        # one object chosen from c2
                        t1_idx[idx] = sub2ind(P * Q, c1 * P + o1, c1 * P + o2)
                        t2_idx[idx] = sub2ind(P * Q, c1 * P + o1, c2 * P + o3)
                        idx += 1
    return t1_idx, t2_idx


def online_clustering(features, num_vc):
    '''Do online clustering'''

    normed_features = tf.nn.l2_normalize(features, dim=3, name='normed_features')
    vc_centers = tf.get_variable('vc_centers', initializer=tf.truncated_normal([1, 1, features.shape[3].value, num_vc]))
    normed_vc_centers = tf.nn.l2_normalize(vc_centers, dim=2, name='normed_vc_centers')
    vc_cos = tf.nn.conv2d(normed_features, normed_vc_centers, [1, 1, 1, 1], "SAME", name='vc_cos')
    vc_softmax = tf.nn.softmax(vc_cos, dim=-1, name='vc_softmax')
    sparse_loss_mean = -tf.reduce_mean(tf.reduce_sum(tf.multiply(vc_softmax, vc_cos, name='softmax_cos'), axis=3, name='sum_softmax_cos'), name='mean_sum_softmax_cos')
    # sparse_loss_mean = tf.reduce_mean(tf.reduce_sum(-tf.multiply(vc_softmax, tf.log(vc_softmax), name='plogp'), axis=3, name='entropy'), name='mean_entropy')

    tf.add_to_collection(tf.get_variable_scope().original_name_scope, normed_features)
    tf.add_to_collection(tf.get_variable_scope().original_name_scope, normed_vc_centers)
    tf.add_to_collection(tf.get_variable_scope().original_name_scope, vc_cos)
    tf.add_to_collection(tf.get_variable_scope().original_name_scope, vc_softmax)
    tf.add_to_collection(tf.get_variable_scope().original_name_scope, sparse_loss_mean)
    end_points = slim.utils.convert_collection_to_dict(tf.get_variable_scope().original_name_scope)
    return sparse_loss_mean, end_points
