from datasets import data_provider
from lib import GoogleNet_Model, Loss_ops, nn_Ops, THSG, evaluation, Two_Stage_Model
import copy
from tqdm import tqdm
from tensorflow.contrib import layers
from FLAGS import *
import datetime
# Create the stream of datas from dataset
streams = data_provider.get_streams(FLAGS.batch_size, FLAGS.dataSet, method, crop_size=FLAGS.default_image_size)
stream_train, stream_train_eval, stream_test = streams

regularizer = layers.l2_regularizer(FLAGS.Regular_factor)
_time = (datetime.datetime.now()+datetime.timedelta(hours=8)).strftime("%m-%d-%H-%M")
LOGDIR = FLAGS.log_save_path+FLAGS.dataSet+'/'+FLAGS.LossType+'/'+_time+'/'

if FLAGS.SaveVal:
    nn_Ops.create_path(_time)


def main(_):
    if not FLAGS.LossType == 'Triplet':
        print("LossType triplet loss is required")
        return 0

    # placeholders
    x_raw = tf.placeholder(tf.float32, shape=[None, FLAGS.default_image_size, FLAGS.default_image_size, 3])
    label_raw = tf.placeholder(tf.int32, shape=[None, 1])
    with tf.name_scope('istraining'):
        is_Training = tf.placeholder(tf.bool)
    with tf.name_scope('learning_rate'):
        lr = tf.placeholder(tf.float32)

    with tf.variable_scope('Classifier'):
        google_net_model = GoogleNet_Model.GoogleNet_Model(pooling_type=FLAGS.pooling_type)
        embedding = google_net_model.forward(x_raw)
        embedding = nn_Ops.bn_block(
            embedding, normal=FLAGS.normalize, is_Training=is_Training, name='BN1')

        embedding = nn_Ops.fc_block(
            embedding, in_d=1024, out_d=FLAGS.embedding_size,
            name='fc1', is_bn=False, is_relu=False, is_Training=is_Training
        )
        with tf.name_scope('Loss'):
            # wdLoss = layers.apply_regularization(regularizer, weights_list=None)
            def exclude_batch_norm(name):
                return 'batch_normalization' not in name and 'Generator' not in name and 'Loss' not in name

            wdLoss = FLAGS.Regular_factor * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if exclude_batch_norm(v.name)]
            )
            # Get the Label
            label = tf.reduce_mean(label_raw, axis=1, keep_dims=False)
            # For some kinds of Losses, the embedding should be l2 normed
            #embedding_l = embedding_z
            J_m = Loss_ops.Loss(embedding, label, FLAGS.LossType)
            # J_m=tf.Print(J_m,[J_m])
            J_m = J_m + wdLoss
    with tf.name_scope('Jgen2'):
        Jgen2 = tf.placeholder(tf.float32)
    with tf.name_scope('Pull_dis_mean'):
        Pull_dis_mean = tf.placeholder(tf.float32)
    Pull_dis = nn_Ops.data_collector(tag='pulling_linear', init=25)

    embedding_l, disap = Two_Stage_Model.Pulling_Positivate(FLAGS.LossType, embedding, can=FLAGS.alpha)


    with tf.variable_scope('Generator1'):
        embedding_g = Two_Stage_Model.generator_ori(embedding_l, FLAGS.LossType)
        with tf.name_scope('Loss'):
            def exclude_batch_norm1(name):
                return 'batch_normalization' not in name and 'Generator1' in name and 'Loss' not in name

            wdLoss_g1 = FLAGS.Regular_factor * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if exclude_batch_norm1(v.name)]
            )
    # Generator1_S
    with tf.variable_scope('GeneratorS'):
        embedding_s = Two_Stage_Model.generator_ori(embedding_g, FLAGS.LossType)
        with tf.name_scope('Loss'):
            def exclude_batch_norm2(name):
                return 'batch_normalization' not in name and 'GeneratorS' in name and 'Loss' not in name

            wdLoss_g1s = FLAGS.Regular_factor * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if exclude_batch_norm2(v.name)]
            )

    # Discriminator1 only contains the anchor and positive message
    with tf.variable_scope('Discriminator1') as scope:
        embedding_dis1 = Two_Stage_Model.discriminator1(Two_Stage_Model.slice_ap_n(embedding),1)
        scope.reuse_variables()
        embedding_g_dis1 = Two_Stage_Model.discriminator1(Two_Stage_Model.slice_ap_n(embedding_g),1)
        # scope.reuse_variables()
        # embedding_s_dis1 = Two_Stage_Model.discriminator1(Two_Stage_Model.slice_ap_n(embedding_s))


    with tf.variable_scope('DiscriminatorS') as scope:
        embedding_dis1S = Two_Stage_Model.discriminator1(Two_Stage_Model.slice_ap_n(embedding),1)
        scope.reuse_variables()
        embedding_s_dis1S = Two_Stage_Model.discriminator1(Two_Stage_Model.slice_ap_n(embedding_s),1)

    with tf.variable_scope('Generator2'):
        embedding_h = Two_Stage_Model.generator_ori(embedding_g)
    with tf.variable_scope('Discriminator2') as scope:
        embedding_dis2 = Two_Stage_Model.discriminator2(embedding)
        scope.reuse_variables()
        embedding_g_dis2 = Two_Stage_Model.discriminator2(embedding_g)
        scope.reuse_variables()
        embedding_h_dis2 = Two_Stage_Model.discriminator2(embedding_h)

    embedding_h_cli = embedding_h

    '''
        using binary_crossentropy replace sparse_softmax_cross_entropy_with_logits
    '''
    with tf.name_scope('Loss'):
        J_syn = Loss_ops.Loss(embedding_h_cli, label, _lossType=FLAGS.LossType, hard_ori=FLAGS.HARD_ORI)
        # J_syn = tf.constant(0.)
        J_m = J_m
        para1 = tf.exp(-FLAGS.beta / Jgen2)
        J_metric = para1 * J_m + (1. - para1) * J_syn

        real_loss_d1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.zeros([FLAGS.batch_size*2/3], dtype=tf.int32), logits=embedding_dis1))
        generated1_loss_d1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.ones([FLAGS.batch_size*2/3], dtype=tf.int32), logits=embedding_g_dis1))
        J_LD1 = FLAGS.Softmax_factor * -(tf.reduce_mean(tf.log(1. - tf.nn.sigmoid(embedding_dis1)) + tf.log(tf.nn.sigmoid(embedding_g_dis1))))

        J_LG1 = FLAGS.Softmax_factor * (-tf.reduce_mean(tf.log(1. - tf.nn.sigmoid(embedding_g_dis1)))) + wdLoss_g1
        embedding_g_split = tf.split(embedding_g, 3, axis=0)
        embedding_g_split_anc = embedding_g_split[0]
        embedding_g_split_pos = embedding_g_split[1]
        dis_g1 = tf.reduce_mean(
            Two_Stage_Model.distance(embedding_g_split_anc, embedding_g_split_pos))
        dis_g1 = tf.maximum(-dis_g1 + 1000, 0.)*10
        J_LG1 = J_LG1

        '''add for D1S'''
        J_LD1S = FLAGS.Softmax_factor * (-tf.reduce_mean(tf.log(1. - tf.nn.sigmoid(embedding_dis1S)) + tf.log(tf.nn.sigmoid(embedding_s_dis1S))))

        J_LG1_S_cross = FLAGS.Softmax_factor *(-tf.reduce_mean(tf.log(1. - tf.nn.sigmoid(embedding_s_dis1S))))
        recon_ori_s = FLAGS.Recon_factor * tf.reduce_mean(Two_Stage_Model.distance(embedding_s, embedding))
        J_LG1_S = J_LG1_S_cross + recon_ori_s + wdLoss_g1s

        # label_onehot = tf.one_hot(label, FLAGS.num_class+1)
        real_loss_d2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=embedding_dis2))
        generated2_loss_d2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=embedding_g_dis2))
        generated2_h_loss_d2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                         labels=tf.zeros([FLAGS.batch_size] ,dtype=tf.int32)+FLAGS.num_class, logits=embedding_h_dis2))
        J_LD2 = FLAGS.Softmax_factor * (real_loss_d2 + generated2_loss_d2 + generated2_h_loss_d2) / 3
        J_LD2 = J_LD2 + J_syn

        cross_entropy, W_fc, b_fc = THSG.cross_entropy(embedding=embedding, label=label)
        Logits_q = tf.matmul(embedding_h, W_fc) + b_fc
        J_LG2_cross_entropy = FLAGS.Softmax_factor * FLAGS._lambda * tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=Logits_q))

        J_LG2C_cross_GAN = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=embedding_h_dis2))
        J_LG2C_cross = FLAGS.Softmax_factor * (J_LG2C_cross_GAN + J_LG2_cross_entropy) / 2
        recon_g_h_ancpos = FLAGS.Recon_factor * tf.reduce_mean(Two_Stage_Model.distance(Two_Stage_Model.slice_ap_n(embedding_g),
                                                                   Two_Stage_Model.slice_ap_n(embedding_h)))

        J_fan = FLAGS.Softmax_factor * Loss_ops.Loss_fan(embedding_h, label, _lossType=FLAGS.LossType,
                                                          param=2 - tf.exp(-FLAGS.beta / Jgen2),
                                                          hard_ori=FLAGS.HARD_ORI)
        J_LG2 = J_LG2C_cross + recon_g_h_ancpos + J_fan
        # J_LG2 = tf.Print(J_LG2, [J_LG2])

        J_F = J_metric
    c_train_step = nn_Ops.training(loss=J_F, lr=lr, var_scope='Classifier')
    d1_train_step = nn_Ops.training(loss=J_LD1, lr=FLAGS.lr_dis, var_scope='Discriminator1')
    g1_train_step = nn_Ops.training(loss=J_LG1, lr=FLAGS.lr_gen, var_scope='Generator1')
    d1s_train_step = nn_Ops.training(loss=J_LD1S, lr=FLAGS.lr_dis, var_scope='DiscriminatorS')
    g1s_train_step = nn_Ops.training(loss=J_LG1_S, lr=FLAGS.lr_gen, var_scope='GeneratorS*Generator1')

    d2_train_step = nn_Ops.training(loss=J_LD2, lr=FLAGS.lr_dis, var_scope='Discriminator2')
    g2_train_step = nn_Ops.training(loss=J_LG2, lr=FLAGS.lr_gen, var_scope='Generator2')

    s_train_step = nn_Ops.training(loss=cross_entropy, lr=FLAGS.s_lr, var_scope='Softmax_classifier')

    # initialise the session
    with tf.Session(config=config) as sess:
        summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
        # Initial all the variables with the sess
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # learning rate
        _lr = FLAGS.init_learning_rate

        # Restore a checkpoint
        if FLAGS.load_formalVal:
            saver.restore(sess, FLAGS.log_save_path+FLAGS.dataSet+'/'+FLAGS.LossType+'/'+FLAGS.formerTimer)

        # Training
        epoch_iterator = stream_train.get_epoch_iterator()

        # collectors
        J_m_loss = nn_Ops.data_collector(tag='J_m', init=1e+6)
        J_syn_loss = nn_Ops.data_collector(tag='J_syn', init=1e+6)
        J_metric_loss = nn_Ops.data_collector(tag='J_metric', init=1e+6)

        real_loss_d1_loss = nn_Ops.data_collector(tag='real_loss_d1', init=1e+6)
        generated1_loss_d1_loss = nn_Ops.data_collector(tag='generated1_loss_d1', init=1e+6)
        J_LD1S_loss = nn_Ops.data_collector(tag='L_D1S', init=1e+6)
        J_LG1_loss = nn_Ops.data_collector(tag='J_LG1', init=1e+6)

        J_LG1_S_cross_loss = nn_Ops.data_collector(tag='J_LG1_S_cross', init=1e+6)
        recon_ori_s_loss = nn_Ops.data_collector(tag='recon_ori_s', init=1e+6)

        real_loss_d2_loss = nn_Ops.data_collector(tag='real_loss_d2', init=1e+6)
        generated2_loss_d2_loss = nn_Ops.data_collector(tag='generated2_loss_d2', init=1e+6)
        generated2_h_loss_d2_loss = nn_Ops.data_collector(tag='generated2_h_loss_d2', init=1e+6)

        J_LG2C_cross_loss = nn_Ops.data_collector(tag='J_LG2C_cross', init=1e+6)
        recon_g_h_ancpos_loss = nn_Ops.data_collector(tag='recon_g_h_ancpos', init=1e+6)
        J_LG2_loss = nn_Ops.data_collector(tag='J_LG2', init=1e+6)
        J_fan_loss = nn_Ops.data_collector(tag='J_fan', init=1e+6)

        J_LD1_loss = nn_Ops.data_collector(tag='J_LD1', init=1e+6)
        J_LD2_loss = nn_Ops.data_collector(tag='J_LD2', init=1e+6)
        J_F_loss = nn_Ops.data_collector(tag='J_F', init=1e+6)
        cross_entropy_loss = nn_Ops.data_collector(tag='cross_entropy', init=1e+6)
        dis_g1_loss = nn_Ops.data_collector(tag='dis_g1', init=1e+6)

        max_nmi = 0
        step = 0

        bp_epoch = FLAGS.init_batch_per_epoch
        with tqdm(total=FLAGS.max_steps) as pbar:
            for batch in copy.copy(epoch_iterator):
                # get images and labels from batch
                x_batch_data, Label_raw = nn_Ops.batch_data(batch)
                pbar.update(1)
                _, _, _, disap_var = sess.run([d1_train_step, d1s_train_step, d2_train_step,disap], feed_dict={x_raw: x_batch_data,
                                                                   label_raw: Label_raw,
                                                                   is_Training: True, lr: _lr,
                                                                   Pull_dis_mean : Pull_dis.read()})
                Pull_dis.update(var=disap_var.mean()*0.8)
                _, _, _, disap_var = sess.run([d1_train_step, d1s_train_step, d2_train_step,disap], feed_dict={x_raw: x_batch_data,
                                                                    label_raw: Label_raw,
                                                                    is_Training: True, lr: _lr,
                                                                    Pull_dis_mean : Pull_dis.read()})
                Pull_dis.update(var=disap_var.mean()*0.8)
                c_train, s_train, d1_train, g1_train, d1s_train, g1s_train, d2_train, g2_train, real_loss_d2_var, J_metric_var, J_m_var, \
                    J_syn_var, real_loss_d1_var, generated1_loss_d1_var, J_LD1_var, J_LD2_var, J_LD1S_var, \
                    J_LG1_var, J_LG1_S_cross_var, recon_ori_s_var,  real_loss_d2_var, generated2_loss_d2_var, cross_entropy_var, \
                    generated2_h_loss_d2_var, J_LG2C_cross_var, recon_g_h_ancpos_var, J_LG2_var, J_fan_var, J_F_var, dis_g1_var,disap_var\
                    = sess.run(
                        [c_train_step, s_train_step, d1_train_step, g1_train_step, d1s_train_step, g1s_train_step, d2_train_step, g2_train_step,
                         real_loss_d2, J_metric, J_m, J_syn, real_loss_d1, generated1_loss_d1, J_LD1, J_LD2,
                         J_LD1S, J_LG1, J_LG1_S_cross, recon_ori_s, real_loss_d2,
                         generated2_loss_d2, cross_entropy, generated2_h_loss_d2, J_LG2C_cross, recon_g_h_ancpos, J_LG2, J_fan, J_F, dis_g1,disap],
                        feed_dict={x_raw: x_batch_data,
                                   label_raw: Label_raw,
                                   is_Training: True, lr: _lr, Jgen2: J_LG2_loss.read(), Pull_dis_mean : Pull_dis.read()})
                Pull_dis.update(var=disap_var.mean()*0.8)
                real_loss_d2_loss.update(var=real_loss_d2_var)
                J_metric_loss.update(var=J_metric_var)
                J_m_loss.update(var=J_m_var)
                J_syn_loss.update(var=J_syn_var)
                real_loss_d1_loss.update(var=real_loss_d1_var)
                generated1_loss_d1_loss.update(var=generated1_loss_d1_var)
                J_LD1S_loss.update(var=J_LD1S_var)
                dis_g1_loss.update(var=dis_g1_var)
                J_LG1_loss.update(var=J_LG1_var)
                J_LG1_S_cross_loss.update(var=J_LG1_S_cross_var)
                recon_ori_s_loss.update(var=recon_ori_s_var)
                real_loss_d2_loss.update(var=real_loss_d2_var)
                generated2_loss_d2_loss.update(var=generated2_loss_d2_var)
                generated2_h_loss_d2_loss.update(var=generated2_h_loss_d2_var)
                J_LG2C_cross_loss.update(var=J_LG2C_cross_var)
                recon_g_h_ancpos_loss.update(var=recon_g_h_ancpos_var)
                J_LG2_loss.update(var=J_LG2_var)
                J_fan_loss.update(var=J_fan_var)
                J_LD1_loss.update(var=J_LD1_var)
                J_LD2_loss.update(var=J_LD2_var)
                J_F_loss.update(var=J_F_var)
                cross_entropy_loss.update(var=cross_entropy_var)
                step += 1
                # print('learning rate %f' % _lr)

                # evaluation
                if step % bp_epoch == 0:
                    print('only eval eval')
                    # nmi_te_cli, f1_te_cli, recalls_te_cli, map_cli = evaluation.Evaluation(
                    #     stream_test, image_mean, sess, x_raw, label_raw, is_Training, embedding, 98, neighbours)
                    recalls_te_cli, map_cli = evaluation.Evaluation(
                            stream_test, image_mean, sess, x_raw, label_raw, is_Training, embedding, 98, neighbours)
                    # Summary
                    eval_summary = tf.Summary()
                    # eval_summary.value.add(tag='test nmi', simple_value=nmi_te_cli)
                    # eval_summary.value.add(tag='test f1', simple_value=f1_te_cli)
                    eval_summary.value.add(tag='test map', simple_value=map_cli)
                    for i in range(0, np.shape(neighbours)[0]):
                        eval_summary.value.add(tag='Recall@%d test' % neighbours[i], simple_value=recalls_te_cli[i])

                    # Embedding_Visualization.embedding_Visual("./", embedding_var, summary_writer)

                    real_loss_d2_loss.write_to_tfboard(eval_summary)
                    J_metric_loss.write_to_tfboard(eval_summary)
                    J_m_loss.write_to_tfboard(eval_summary)
                    J_syn_loss.write_to_tfboard(eval_summary)
                    real_loss_d1_loss.write_to_tfboard(eval_summary)
                    generated1_loss_d1_loss.write_to_tfboard(eval_summary)
                    J_LD1S_loss.write_to_tfboard(eval_summary)
                    J_LG1_loss.write_to_tfboard(eval_summary)
                    dis_g1_loss.write_to_tfboard(eval_summary)
                    J_LD1_loss.write_to_tfboard(eval_summary)
                    J_LD2_loss.write_to_tfboard(eval_summary)
                    J_F_loss.write_to_tfboard(eval_summary)
                    J_LG1_S_cross_loss.write_to_tfboard(eval_summary)
                    recon_ori_s_loss.write_to_tfboard(eval_summary)
                    real_loss_d2_loss.write_to_tfboard(eval_summary)
                    generated2_loss_d2_loss.write_to_tfboard(eval_summary)
                    generated2_h_loss_d2_loss.write_to_tfboard(eval_summary)
                    J_LG2C_cross_loss.write_to_tfboard(eval_summary)
                    recon_g_h_ancpos_loss.write_to_tfboard(eval_summary)
                    J_LG2_loss.write_to_tfboard(eval_summary)
                    J_fan_loss.write_to_tfboard(eval_summary)
                    cross_entropy_loss.write_to_tfboard(eval_summary)
                    summary_writer.add_summary(eval_summary, step)
                    print('Summary written')
                    if map_cli > max_nmi:
                        max_nmi = map_cli
                        print("Saved")
                        saver.save(sess, os.path.join(LOGDIR, "model.ckpt"))
                    # saver.save(sess, os.path.join(LOGDIR, "model.ckpt"))
                    summary_writer.flush()
                    if step in [5632, 6848]:
                        _lr = _lr * 0.5

                    if step >= 5000:
                        bp_epoch = FLAGS.batch_per_epoch
                    if step >= FLAGS.max_steps:
                        os._exit(0)


if __name__ == '__main__':
    tf.app.run()
