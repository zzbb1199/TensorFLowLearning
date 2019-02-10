"""
读取保存的常量文件
"""
import tensorflow as tf

with tf.Session() as sess:
    model_filename = "./save_path/combined_model.pb"
    # 读取保存的模型
    with tf.gfile.GFile(model_filename,"rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 将graph_def 中保存的图加载到当前图中
    ret = tf.import_graph_def(graph_def,return_elements=["add:0"])
    print(sess.run(ret))