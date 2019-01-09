import tensorflow as tf

a = tf.constant([1.0, 2], name="a")
b = tf.constant([3.0, 4], name="b")
ret = a + b

# 计算图
print(a.graph)
print(a.graph is tf.get_default_graph())

# new 一个 grpah
g1 = tf.Graph()
with g1.as_default():
    tf.zeros_initializer()
    v = tf.get_variable("v", initializer=tf.zeros_initializer(), shape=[1])

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", initializer=tf.ones_initializer(), shape=[1])

# 在计算图g1中读取变量v的取值
with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        # 在计算图g1中v取值应该为0
        print(sess.run(tf.get_variable("v")))

with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

# 创建会话
sess = tf.Session()
with sess.as_default():
    print(ret.eval())


sess = tf.InteractiveSession() # InterativeSession自动注册为默认会话
print(ret.eval())
sess.close()
