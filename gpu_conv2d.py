import tensorflow.compat.v1 as tf


def main():
    tf.disable_eager_execution()

    with tf.device('/gpu:0'):
        t1 = tf.random.uniform(shape=[32, 56, 56, 64], dtype=tf.half)
        t2 = tf.random.uniform(shape=[3, 3, 64, 64], dtype=tf.half)
        t = tf.nn.conv2d(
                input=t1,
                filters=t2,
                strides=[2, 2],
                padding='SAME',
                data_format='NHWC',
                name='Conv2D')

    run_options = tf.RunOptions()
    run_options.trace_level = run_options.FULL_TRACE
    run_metadata = tf.RunMetadata()

    options = tf.GraphOptions(build_cost_model=1)
    cfg = tf.ConfigProto(graph_options=options)
    with tf.Session(config=cfg) as sess:
        sess.run(tf.global_variables_initializer())
        _ = sess.run([t], options=run_options, run_metadata=run_metadata)

    for node in run_metadata.cost_graph.node:
        if node.name == 'Conv2D':
            print(node.name, ':', node.compute_cost * 1000, 'ns.')


if __name__ == '__main__':
    main()
