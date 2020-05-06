from FLAGS import *
import os
from tensorflow.contrib import layers
from lib import nn_Ops
def balance_parameter(Jgen2_pre, Jgen2, can=5):
    #for tri *100 can=1
    if FLAGS.LossType=='Triplet':
        x = (tf.abs(Jgen2_pre - Jgen2) / Jgen2) * 100
    else:
        x = (tf.abs(Jgen2_pre - Jgen2) / Jgen2) * 40
    parameter = 1.0 / (1 + tf.exp(1 - x))
    parameter = tf.cond(tf.greater(Jgen2_pre, 0), lambda: parameter, lambda: tf.constant(1.0))
    return parameter


def spand_npair(embedding, label=False):
    if label:
        label_pre = tf.slice(
            input_=embedding, begin=[0],
            size=[int(FLAGS.batch_size / 2)])
        label_after = tf.slice(
            input_=embedding, begin=[int(FLAGS.batch_size / 2)],
            size=[int(FLAGS.batch_size / 2)])
        label_pos = tf.reshape(label_after, [int(FLAGS.batch_size / 2), 1])
        label_neg_tile = tf.tile(label_pos, [int(FLAGS.batch_size / 2), 1])
        label_neg_tile_2 = tf.reshape(label_neg_tile, [-1])
        label_anc_2 = tf.reshape(label_pre, [-1])
        label_pos_2 = tf.reshape(label_pos, [-1])
        label_l = tf.concat([label_anc_2, label_pos_2, label_neg_tile_2], axis=0)
        return label_l
    else:
        embedding_split = tf.split(embedding, 2, axis=0)
        anc = embedding_split[0]
        pos = embedding_split[1]
        neg = pos
        neg_tile = tf.tile(neg, [int(FLAGS.batch_size / 2), 1])
        embedding_l = tf.concat([anc, pos, neg_tile], axis=0)
        return embedding_l


def spand_npair2(embedding, labels, indx=None):
    dim = embedding.shape[1].value
    label_pre = tf.slice(
        input_=labels, begin=[0],
        size=[int(FLAGS.batch_size / 2)])
    label_after = tf.slice(
        input_=labels, begin=[int(FLAGS.batch_size / 2)],
        size=[int(FLAGS.batch_size / 2)])
    label_anc_tile = tf.reshape(tf.tile(tf.reshape(label_pre, [-1, 1]), [1, int(FLAGS.batch_size / 2)]), [-1])
    label_pos_tile = tf.reshape(tf.tile(tf.reshape(label_after, [-1, 1]), [1, int(FLAGS.batch_size / 2)]), [-1])
    label_neg_tile = tf.reshape(tf.tile(tf.reshape(label_after, [-1, 1]), [int(FLAGS.batch_size / 2), 1]), [-1])

    embedding_split = tf.split(embedding, 2, axis=0)
    anc = embedding_split[0]
    pos = embedding_split[1]
    anc_tile = tf.reshape(tf.tile(anc, [1, int(FLAGS.batch_size / 2)]), [-1, int(dim)])
    pos_tile = tf.reshape(tf.tile(pos, [1, int(FLAGS.batch_size / 2)]), [-1, int(dim)])
    neg_tile = tf.tile(pos, [int(FLAGS.batch_size / 2), 1])

    if indx!=None:
        indx=tf.reshape(tf.tile(tf.reshape(indx,[-1,1]),[1,int(FLAGS.batch_size/2)]),[-1])
        ind = tf.where(indx > 0)
        ind = tf.cast(ind, tf.int32)
        anc_tile = tf.gather_nd(anc_tile, ind)
        pos_tile = tf.gather_nd(pos_tile, ind)
        neg_tile = tf.gather_nd(neg_tile, ind)

        label_anc_tile = tf.gather_nd(label_anc_tile, ind)
        label_pos_tile = tf.gather_nd(label_pos_tile, ind)
        label_neg_tile = tf.gather_nd(label_neg_tile, ind)

    # dis_pn = tf.reshape(distance(pos_tile, neg_tile), [-1])
    # ind2 = tf.where(dis_pn > 0)
    # ind2 = tf.cast(ind2, tf.int32)
    dis_pn = tf.not_equal(label_pos_tile, label_neg_tile)
    dis_pn = tf.cast(dis_pn, tf.int32)
    ind2 = tf.where(dis_pn > 0)
    ind2 = tf.cast(ind2, tf.int32)

    anc_tile = tf.gather_nd(anc_tile, ind2)
    pos_tile = tf.gather_nd(pos_tile, ind2)
    neg_tile = tf.gather_nd(neg_tile, ind2)
    embedding_l = tf.concat([anc_tile, pos_tile, neg_tile], axis=0)

    label_anc_tile = tf.gather_nd(label_anc_tile, ind2)
    label_pos_tile = tf.gather_nd(label_pos_tile, ind2)
    label_neg_tile = tf.gather_nd(label_neg_tile, ind2)
    label_embedding_l = tf.concat([label_anc_tile, label_pos_tile, label_neg_tile], axis=0)
    return embedding_l, label_embedding_l

def spand_npair2(embedding, labels, indx=None, axis=0):
    dim = embedding.shape[1].value
    label_pre = tf.slice(
        input_=labels, begin=[0],
        size=[int(FLAGS.batch_size / 2)])
    label_after = tf.slice(
        input_=labels, begin=[int(FLAGS.batch_size / 2)],
        size=[int(FLAGS.batch_size / 2)])
    label_anc_tile = tf.reshape(tf.tile(tf.reshape(label_pre, [-1, 1]), [1, int(FLAGS.batch_size / 2)]), [-1])
    label_pos_tile = tf.reshape(tf.tile(tf.reshape(label_after, [-1, 1]), [1, int(FLAGS.batch_size / 2)]), [-1])
    label_neg_tile = tf.reshape(tf.tile(tf.reshape(label_after, [-1, 1]), [int(FLAGS.batch_size / 2), 1]), [-1])

    embedding_split = tf.split(embedding, 2, axis=0)
    anc = embedding_split[0]
    pos = embedding_split[1]
    anc_tile = tf.reshape(tf.tile(anc, [1, int(FLAGS.batch_size / 2)]), [-1, int(dim)])
    pos_tile = tf.reshape(tf.tile(pos, [1, int(FLAGS.batch_size / 2)]), [-1, int(dim)])
    neg_tile = tf.tile(pos, [int(FLAGS.batch_size / 2), 1])

    if indx!=None:
        indx=tf.reshape(tf.tile(tf.reshape(indx,[-1,1]),[1,int(FLAGS.batch_size/2)]),[-1])
        ind = tf.where(indx > 0)
        ind = tf.cast(ind, tf.int32)
        anc_tile = tf.gather_nd(anc_tile, ind)
        pos_tile = tf.gather_nd(pos_tile, ind)
        neg_tile = tf.gather_nd(neg_tile, ind)

        label_anc_tile = tf.gather_nd(label_anc_tile, ind)
        label_pos_tile = tf.gather_nd(label_pos_tile, ind)
        label_neg_tile = tf.gather_nd(label_neg_tile, ind)

    # dis_pn = tf.reshape(distance(pos_tile, neg_tile), [-1])
    # ind2 = tf.where(dis_pn > 0)
    # ind2 = tf.cast(ind2, tf.int32)
    dis_pn = tf.not_equal(label_pos_tile, label_neg_tile)
    dis_pn = tf.cast(dis_pn, tf.int32)
    ind2 = tf.where(dis_pn > 0)
    ind2 = tf.cast(ind2, tf.int32)

    anc_tile = tf.gather_nd(anc_tile, ind2)
    pos_tile = tf.gather_nd(pos_tile, ind2)
    neg_tile = tf.gather_nd(neg_tile, ind2)
    embedding_l = tf.concat([anc_tile, pos_tile, neg_tile], axis=axis)

    label_anc_tile = tf.gather_nd(label_anc_tile, ind2)
    label_pos_tile = tf.gather_nd(label_pos_tile, ind2)
    label_neg_tile = tf.gather_nd(label_neg_tile, ind2)
    label_embedding_l = tf.concat([label_anc_tile, label_pos_tile, label_neg_tile], axis=0)
    return embedding_l, label_embedding_l



def spand_npair_indx(labels, indx=None, axis=0):
    label_pre = tf.slice(
        input_=labels, begin=[0],
        size=[int(FLAGS.batch_size / 2)])
    label_after = tf.slice(
        input_=labels, begin=[int(FLAGS.batch_size / 2)],
        size=[int(FLAGS.batch_size / 2)])
    label_pos_tile = tf.reshape(tf.tile(tf.reshape(label_after, [-1, 1]), [1, int(FLAGS.batch_size / 2)]), [-1])
    label_neg_tile = tf.reshape(tf.tile(tf.reshape(label_after, [-1, 1]), [int(FLAGS.batch_size / 2), 1]), [-1])

    indx = tf.reshape(tf.tile(tf.reshape(indx, [-1, 1]), [1, int(FLAGS.batch_size / 2)]), [-1])

    dis_pn = tf.not_equal(label_pos_tile, label_neg_tile)
    dis_pn = tf.cast(dis_pn, tf.int32)
    ind2 = tf.where(dis_pn > 0)
    ind2 = tf.cast(ind2, tf.int32)

    indx = tf.gather_nd(indx, ind2)
    return indx


def slice_ap_n(embedding, label=False, embedding_size=1024, loss_tyte=FLAGS.LossType):
    """
    extract the anc and pos from the embedding vector
    """
    if loss_tyte == "Triplet":
        # if label:
        #     embedding_ap = tf.slice(
        #         input_=embedding, begin=[0],
        #         size=[int(FLAGS.batch_size / 3) * 2])
        # else:
            embedding_split = tf.split(embedding, 3, axis=0)
            anc = embedding_split[0]
            pos = embedding_split[1]
            embedding_ap = tf.concat([anc, pos], axis=0)
    elif loss_tyte == "NpairLoss":
        embedding_size = embedding.shape[1].value
        embedding_ap = embedding
    return embedding_ap


def distance(emb1, emb2):
    """
    Calculate the sqrt l2 distance between two embedding
    :param emb1: embedding 1
    :param emb2: embedding 2
    :return: The distance
    """
    return tf.sqrt(tf.reduce_sum(tf.square(emb1-emb2), axis=1, keep_dims=True))


def Pulling_Positivate_Adapt(Loss_type, embedding, Pull_dis_mean):
    if Loss_type == 'Triplet':
        embedding_split = tf.split(embedding, 3, axis=0)
        anc = embedding_split[0]
        pos = embedding_split[1]
        neg = embedding_split[2]
        disap = distance(anc, pos)
        # disap = tf.Print(disap,[disap, anc, pos])
        # lam = tf.exp(can-disap)
        neg_mask = tf.greater_equal(disap, Pull_dis_mean)
        op_neg_mask = tf.logical_not(neg_mask)
        neg_mask = tf.cast(neg_mask, tf.float32)
        op_neg_mask = tf.cast(op_neg_mask, tf.float32)
        lam = tf.multiply(tf.exp(Pull_dis_mean-disap), neg_mask) + tf.multiply(1+(Pull_dis_mean-disap)*0.3, op_neg_mask)

        anc = anc + lam * (anc - pos)
        pos = pos + lam * (pos - anc)
        disap_after = distance(anc, pos)
        # anc = tf.Print(anc, [disap_after,anc, pos,lam])
        embedding_l = tf.concat([anc, pos, neg], axis=0)
        return embedding_l, disap
    else:
        print("Your loss type is not suit for two_stage")
        os._exit()


def Pulling_Positivate(Loss_type, embedding, can=20):
    if Loss_type == 'Triplet':
        embedding_split = tf.split(embedding, 3, axis=0)
        anc = embedding_split[0]
        pos = embedding_split[1]
        neg = embedding_split[2]
        disap = distance(anc, pos)
        # disap = tf.Print(disap,[disap, anc, pos])
        # lam = tf.exp(can-disap)
        neg_mask = tf.greater_equal(disap, can)
        op_neg_mask = tf.logical_not(neg_mask)
        neg_mask = tf.cast(neg_mask, tf.float32)
        op_neg_mask = tf.cast(op_neg_mask, tf.float32)
        lam = tf.multiply(tf.exp(can-disap), neg_mask) + tf.multiply(1+(can-disap)*0.3, op_neg_mask)

        anc = anc + lam * (anc - pos)
        pos = pos + lam * (pos - anc)
        disap_after = distance(anc, pos)
        # anc = tf.Print(anc, [disap_after,anc, pos,lam])
        embedding_l = tf.concat([anc, pos, neg], axis=0)
        return embedding_l, disap
    elif Loss_type == 'NpairLoss':
        embedding_split = tf.split(embedding, 2, axis=0)
        anc = embedding_split[0]
        pos = embedding_split[1]
        disap = distance(anc, pos)
        # disap = tf.Print(disap,[disap, anc, pos])
        # lam = tf.exp(can-disap)
        neg_mask = tf.greater_equal(disap, can)
        op_neg_mask = tf.logical_not(neg_mask)
        neg_mask = tf.cast(neg_mask, tf.float32)
        op_neg_mask = tf.cast(op_neg_mask, tf.float32)
        # lam = tf.multiply(tf.exp(can - disap), neg_mask) + tf.multiply(1 + (can - disap) * FLAGS.alpha2, op_neg_mask)
        lam = tf.multiply(FLAGS.alpha2 * tf.exp(can - disap), neg_mask) + tf.multiply(FLAGS.alpha2 + (can - disap) * FLAGS.alpha2, op_neg_mask)
        anc = anc + lam * (anc - pos)
        pos = pos + lam * (pos - anc)
        # disap_after = distance(anc, pos)
        # anc = tf.Print(anc, [disap_after,anc, pos,lam])
        embedding_l = tf.concat([anc, pos], axis=0)
        return embedding_l, disap
    else:
        print("Your loss type is not suit for two_stage")
        os._exit()


def Pulling_Positivate_linear(Loss_type, embedding, can=20):
    if Loss_type == 'Triplet':
        embedding_split = tf.split(embedding, 3, axis=0)
        anc = embedding_split[0]
        pos = embedding_split[1]
        neg = embedding_split[2]
        disap = distance(anc, pos)
        # disap = tf.Print(disap,[disap, anc, pos])
        # lam = tf.exp(can-disap)
        neg_mask = tf.greater_equal(disap, 0)
        neg_mask = tf.cast(neg_mask, tf.float32)
        lam = tf.multiply(1+(can-disap)*0.3, neg_mask)
        #将lam小于0的值全部取成0
        neg_mask2 = tf.greater_equal(lam, 0)
        neg_mask2 = tf.cast(neg_mask2, tf.float32)
        lam = tf.multiply(lam, neg_mask2)

        anc = anc + lam * (anc - pos)
        pos = pos + lam * (pos - anc)
        disap_after = distance(anc, pos)
        # anc = tf.Print(anc, [disap_after,anc, pos,lam])
        embedding_l = tf.concat([anc, pos, neg], axis=0)
        return embedding_l, disap
    else:
        print("Your loss type is not suit for two_stage")
        os._exit()


def Pulling_Positivate_exp(Loss_type, embedding, can=20):
    if Loss_type == 'Triplet':
        embedding_split = tf.split(embedding, 3, axis=0)
        anc = embedding_split[0]
        pos = embedding_split[1]
        neg = embedding_split[2]
        disap = distance(anc, pos)
        # disap = tf.Print(disap,[disap, anc, pos])
        # lam = tf.exp(can-disap)
        neg_mask = tf.greater_equal(disap, 0)
        neg_mask = tf.cast(neg_mask, tf.float32)
        lam = tf.multiply(tf.exp(0-disap), neg_mask)

        anc = anc + lam * (anc - pos)
        pos = pos + lam * (pos - anc)
        disap_after = distance(anc, pos)
        # anc = tf.Print(anc, [disap_after,anc, pos,lam])
        embedding_l = tf.concat([anc, pos, neg], axis=0)
        return embedding_l, disap
    else:
        print("Your loss type is not suit for two_stage")
        os._exit()

# def pooling_layer(embedding, Pooling_type='', is_Training = True):
#     if Pooling_type=='gem':
#         return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

def generator_ori(embedding, Loss_type='', is_Training = True):
    """
    :param Loss_type: if '' represent do generate for all embeddings
    and while not empty, represents do generate only for the anchor and positive embeddings.
    """
    indim = embedding.shape[1].value
    if Loss_type == 'Triplet':
        embedding_split = tf.split(embedding, 3, axis=0)
        anc = embedding_split[0]
        pos = embedding_split[1]
        neg = embedding_split[2]
        embedding = tf.concat([anc, pos], axis=0)
    elif Loss_type == 'NpairLoss':
        embedding = embedding
    elif Loss_type != '':
        print("Your loss type is not suit for two_stage")
        os._exit()
    if FLAGS.ADD_NOISE:
        noise_rand = tf.random_normal(shape=tf.shape(embedding), mean=0.0, stddev=(50) / (255), dtype=tf.float32)
        embedding_g = fc_block(
            embedding + noise_rand, in_d=indim, out_d=512,
            name='generator1', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
        )
    else:
        #add 原本是indim直接到512
        embedding_g = fc_block(
            embedding, in_d=indim, out_d=512,
            name='generator1', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
        )
    embedding_g = tf.nn.dropout(embedding_g, keep_prob=0.9)
    embedding_g = fc_block(
        embedding_g, in_d=512, out_d=256,
        name='generator2', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
    )
    embedding_g = fc_block(
        embedding_g, in_d=256, out_d=512,
        name='generator3', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
    )
    embedding_g = fc_block(
        embedding_g, in_d=512, out_d=indim,
        name='generator4', is_bn=False, is_relu=False, is_Training=is_Training
    )
    if Loss_type == 'Triplet':
        embedding_g = tf.concat([embedding_g, neg], axis=0)
    elif Loss_type == 'NpairLoss':
        embedding_g = embedding_g

    return embedding_g

def generator2(embedding, Loss_type='', is_Training = True):
    """
    :param Loss_type: if '' represent do generate for all embeddings
    and while not empty, represents do generate only for the anchor and positive embeddings.
    """
    indim = embedding.shape[1].value
    if Loss_type == 'NpairLoss':
        embedding = embedding
    elif Loss_type != '':
        print("Your loss type is not suit for two_stage")
        os._exit()#add 原本是indim直接到512
    embedding_g = fc_block(
        embedding, in_d=indim, out_d=512*3,
        name='generator1', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
    )
    embedding_g = fc_block(
        embedding_g, in_d=512*3, out_d=512*3,
        name='generator2', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
    )
    embedding_g = fc_block(
        embedding_g, in_d=512*3, out_d=512*3,
        name='generator3', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
    )
    embedding_g = fc_block(
        embedding_g, in_d=512*3, out_d=1024*3,
        name='generator4', is_bn=False, is_relu=False, is_Training=is_Training
    )
    if Loss_type == 'NpairLoss':
        embedding_g = embedding_g

    return embedding_g

def generator(embedding, Loss_type='', is_Training = True):
    """
    :param Loss_type: if '' represent do generate for all embeddings
    and while not empty, represents do generate only for the anchor and positive embeddings.
    """
    indim = embedding.shape[1].value
    if Loss_type == 'Triplet':
        embedding_split = tf.split(embedding, 3, axis=0)
        anc = embedding_split[0]
        pos = embedding_split[1]
        neg = embedding_split[2]
        embedding = tf.concat([anc, pos], axis=0)
    elif Loss_type == 'NpairLoss':
        embedding = embedding
    elif Loss_type != '':
        print("Your loss type is not suit for two_stage")
        os._exit()
    if FLAGS.ADD_NOISE:
        noise_rand = tf.random_normal(shape=tf.shape(embedding), mean=0.0, stddev=(50) / (255), dtype=tf.float32)
        embedding_g = fc_block(
            embedding + noise_rand, in_d=indim, out_d=512,
            name='generator1', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
        )
    else:
        #add 原本是indim直接到512
        embedding_g = fc_block(
            embedding, in_d=indim, out_d=512,
            name='generator1', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
        )
    embedding_g = fc_block(
        embedding_g, in_d=512, out_d=512,
        name='generator2', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
    )
    embedding_g = fc_block(
        embedding_g, in_d=512, out_d=512,
        name='generator3', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
    )
    embedding_g = fc_block(
        embedding_g, in_d=512, out_d=1024,
        name='generator4', is_bn=False, is_relu=False, is_Training=is_Training
    )
    if Loss_type == 'Triplet':
        embedding_g = tf.concat([embedding_g, neg], axis=0)
    elif Loss_type == 'NpairLoss':
        embedding_g = embedding_g

    return embedding_g


def generator_s(embedding, Loss_type='', is_Training = True):
    """
    :param Loss_type: if '' represent do generate for all embeddings
    and while not empty, represents do generate only for the anchor and positive embeddings.
    """
    indim = embedding.shape[1].value
    if Loss_type == 'Triplet':
        embedding_split = tf.split(embedding, 3, axis=0)
        anc = embedding_split[0]
        pos = embedding_split[1]
        neg = embedding_split[2]
        embedding = tf.concat([anc, pos], axis=0)
    elif Loss_type == 'NpairLoss':
        embedding = embedding
    elif Loss_type != '':
        print("Your loss type is not suit for two_stage")
        os._exit()
    if FLAGS.ADD_NOISE:
        noise_rand = tf.random_normal(shape=tf.shape(embedding), mean=0.0, stddev=(50) / (255), dtype=tf.float32)
        embedding_g = fc_block(
            embedding + noise_rand, in_d=indim, out_d=2048,
            name='generator1', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
        )
    else:
        #add 原本是indim直接到512
        embedding_g = fc_block(
            embedding, in_d=indim, out_d=2048,
            name='generator1', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
        )
    embedding_g = fc_block(
        embedding_g, in_d=2048, out_d=2048,
        name='generator2', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
    )
    embedding_g = fc_block(
        embedding_g, in_d=2048, out_d=1024,
        name='generator4', is_bn=False, is_relu=False, is_Training=is_Training
    )
    if Loss_type == 'Triplet':
        embedding_g = tf.concat([embedding_g, neg], axis=0)
    elif Loss_type == 'NpairLoss':
        embedding_g = embedding_g

    return embedding_g


def discriminator1(embedding, class_num=2, is_Training=True):
    indim = embedding.shape[1].value
    with tf.variable_scope('d1'):
        net = layers.fully_connected(embedding, 512, activation_fn=tf.nn.leaky_relu,
                                     weights_initializer=tf.glorot_uniform_initializer())
        net = tf.layers.batch_normalization(
            inputs=net, center=True,
            scale=True, training=is_Training, fused=True
        )

    # add d2 d3
    with tf.variable_scope('d2'):
        net = layers.fully_connected(net, 256, activation_fn=tf.nn.leaky_relu,
                                     weights_initializer=tf.glorot_uniform_initializer())
        net = tf.layers.batch_normalization(
            inputs=net, center=True,
            scale=True, training=is_Training, fused=True
        )
    with tf.variable_scope('d3'):
        net = tf.nn.dropout(net, keep_prob=0.8)
        net = layers.fully_connected(net, 256, activation_fn=tf.nn.leaky_relu,
                                     weights_initializer=tf.glorot_uniform_initializer())
        net = tf.layers.batch_normalization(
            inputs=net, center=True,
            scale=True, training=is_Training, fused=True
        )
    with tf.variable_scope('d4'):
        net = tf.nn.dropout(net, keep_prob=0.8)
        net = layers.fully_connected(net, class_num, activation_fn=None,
                                     weights_initializer=tf.glorot_uniform_initializer())
    return net


def discriminator2(embedding, is_Training=True):
    with tf.variable_scope('d1'):
        net = layers.fully_connected(embedding, 512, activation_fn=tf.nn.leaky_relu,
                                     weights_initializer=tf.glorot_uniform_initializer())
        net = tf.layers.batch_normalization(
            inputs=net, center=True,
            scale=True, training=is_Training, fused=True
        )
    with tf.variable_scope('d2'):
        net = layers.fully_connected(net, 256, activation_fn=tf.nn.leaky_relu,
                                     weights_initializer=tf.glorot_uniform_initializer())
        net = tf.layers.batch_normalization(
            inputs=net, center=True,
            scale=True, training=is_Training, fused=True
        )
    with tf.variable_scope('d3'):
        net = tf.nn.dropout(net, keep_prob=0.8)
        net = layers.fully_connected(net, 256, activation_fn=tf.nn.leaky_relu,
                                     weights_initializer=tf.glorot_uniform_initializer())
        net = tf.layers.batch_normalization(
            inputs=net, center=True,
            scale=True, training=is_Training, fused=True
        )
        net = tf.nn.dropout(net, keep_prob=0.8)
    with tf.variable_scope('d4'):
        net = layers.fully_connected(net, FLAGS.num_class+1, activation_fn=None,
                                     weights_initializer=tf.glorot_uniform_initializer())
    return net


def fc_block(embedding, in_d, out_d, name, is_bn, is_relu, is_Training=True, reuse=False, is_LeakyReLU=False, is_Tanh=False):
    with tf.variable_scope(name, reuse=reuse):
        W_fc = weight_variable([in_d, out_d], name + "w")
        b_fc = bias_variable([out_d], name + "b")
        embedding = tf.matmul(embedding, W_fc) + b_fc
        assert not (is_LeakyReLU and is_Tanh)

        if is_LeakyReLU:
            embedding = tf.nn.leaky_relu(embedding)
        elif is_Tanh:
            embedding = tf.nn.tanh(embedding)
        elif is_relu:
            embedding = tf.nn.relu(embedding)

        if is_bn:
            embedding = bn_block(embedding, normal=True, is_Training=is_Training, name=name + 'BN')
        return embedding


def bn_block(embedding, normal, is_Training, name, reuse=False):
    if normal:
        with tf.variable_scope(name, reuse=reuse):
            embedding = tf.layers.batch_normalization(
                    inputs=embedding, center=True,
                    scale=True, training=is_Training, fused=True
            )
        print("BN layer: "+name+" is applied")
    return embedding


def weight_variable(shape, name, wd=True):
    initializer = tf.glorot_uniform_initializer()#tf.contrib.layers.xavier_initializer()  # tf.truncated_normal_initializer(stddev=0.1)
    if wd:
        weight = tf.get_variable(name='weight'+name, shape=shape,
                                 initializer=initializer,
                                 collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
    else:
        weight = tf.get_variable(name='weight' + name, shape=shape,
                                 initializer=initializer,
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES])

    return weight


def bias_variable(shape, name):
    bias = tf.get_variable(name='bias'+name, shape=shape, initializer=tf.constant_initializer(0))
    return bias


def Crossloss_index(
    _sentinel=None,  # pylint: disable=invalid-name
    labels=None,
    logits=None,
    loss_type="Npair",
    name=None):
        sm = tf.nn.softmax(logits)
        sm = tf.argmax(sm, 1)
        sm = tf.cast(sm, tf.int32)
        indx = tf.equal(sm, labels)
        indx = tf.cast(indx, tf.float32)
        if loss_type=="Npair":
            indx = tf.split(indx,2)
            indx_a = indx[0]
            indx_p = indx[1]
            final = tf.multiply(indx_a, indx_p)
            return final
        elif loss_type=="Triplet":
            indx = tf.split(indx, 3)
            indx_a = indx[0]
            indx_p = indx[1]
            indx_t = indx[2]
            final = tf.multiply(indx_a, indx_p)
            final = tf.multiply(final, indx_t)
            return final
        else:
            assert 1==0

def classfier(embedding, is_Training):
    with tf.variable_scope('Classifier'):
        embedding_c = nn_Ops.bn_block(
            embedding, normal=FLAGS.normalize, is_Training=is_Training, name='BN1', reuse=True)

        embedding_c = nn_Ops.fc_block(
            embedding_c, in_d=1024, out_d=FLAGS.embedding_size,
            name='fc1', is_bn=False, is_relu=False, reuse=True, is_Training=is_Training
        )
        return embedding_c

def spand_npair_hdml(embedding):
    embedding_split = tf.split(embedding, 2, axis=0)
    anc = embedding_split[0]
    pos = embedding_split[1]
    neg = pos
    neg_tile = tf.tile(neg, [int(FLAGS.batch_size / 2), 1])
    embedding_z_quta = tf.concat([anc, neg_tile], axis=0)
    return embedding_z_quta