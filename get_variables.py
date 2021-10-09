import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow
model_dir = './albert_model/albert_tiny2'
ckpt = tf.train.get_checkpoint_state(model_dir)
# reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(model_dir, 'albet_model.ckpt'))
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print("tensor_name: ", key)
# if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#     print('Reloading model parameters..')
#     ckpt_file = os.path.basename(ckpt.model_checkpoint_path)

# # 先构建网络结构* build_model() *
# # 初始化变量*
# sess.run(tf.global_variables_initializer())
# # 最后从checkpoint中加载已训练好的参数
# saver = tf.train.Saver() saver.restore(self.sess, init_checkpoint)


# print('load_graph...')
# ckpt = tf.train.get_checkpoint_state(self.config["ckpt_model_path"])
# if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
# if ckpt:
#     print('Reloading model parameters..')
#     self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
#
# else:
#     raise ValueError('No such file:[{}]'.format(self.config["ckpt_model_path"]))


def use_saver():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./albert_model/albert_tiny/albert_model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./albert_model/albert_tiny'))
    print(sess.run('w1:0'))
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("w1:0")
    w2 = graph.get_tensor_by_name("w2:0")
    print(saver)


# use_saver()


def pywrap():
    global reader, var_to_shape_map, key
    # 不用创建模型直接加载
    from tensorflow.python import pywrap_tensorflow
    checkpoint_path = os.path.join(model_dir, "albert_model.ckpt")
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        # print(reader.get_tensor(key)) # Remove this is you want to print only variable names
pywrap()


# def in_sess():
#     # 需要构建网络结构 model
#     model = None
#     sess = tf.Session()
#     tvars = tf.trainable_variables()
#     (assignment_map, initialized_variable_names) = model.get_assignment_map_from_checkpoint(tvars, config[
#         "ckpt_model_path"])
#     tf.train.init_from_checkpoint(config["ckpt_model_path"], assignment_map)
#     sess.run(tf.global_variables_initializer())
#
#     for i in tf.global_variables():
#         print(i)

'''
#使用inspect_checkpoint来查看ckpt里的内容
from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file(file_name="/tmp/model.ckpt", 
                                      tensor_name, # 如果为None,则默认为ckpt里的所有变量
                                      all_tensors, # bool 是否打印所有的tensor，这里打印出的是tensor的值，一般不推荐这里设置为False
                                      all_tensor_names) # bool 是否打印所有的tensor的name

#上print_tensors_in_checkpoint_file其实是用NewCheckpointReader实现的。



from tensorflow.python.tools import freeze_graph

freeze_graph(input_graph, #=some_graph_def.pb
             input_saver, 
             input_binary, 
             input_checkpoint, #=model.ckpt
             output_node_names, #=softmax
             restore_op_name, 
             filename_tensor_name, 
             output_graph, #='./tmp/frozen_graph.pb'
             clear_devices, 
             initializer_nodes, 
             variable_names_whitelist='', 
             variable_names_blacklist='', 
             input_meta_graph=None, 
             input_saved_model_dir=None, 
             saved_model_tags='serve', 
             checkpoint_version=2)
#freeze_graph_test.py讲述了怎么使用freeze_grapg。
'''