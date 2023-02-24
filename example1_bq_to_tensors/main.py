import tensorflow as tf

prefix = "gs://mybucket/pool"
path = f"{prefix}/train/output-00000-of-00001.tfrecord"


def decode_fn(record_bytes):
    return tf.io.parse_example(
        # Data
        record_bytes,
        # Schema
        {"label": tf.io.FixedLenFeature([], dtype=tf.int64),
         "userId": tf.io.FixedLenFeature([], dtype=tf.int64),
         "postId": tf.io.FixedLenFeature([], dtype=tf.int64)}
    )


dataset = tf.data.TFRecordDataset([path]).map(decode_fn)
for elem in dataset.batch(16):
    break
# should print a dictionary: userId -> tensor, postId -> tensor, label -> tensor
print(elem)
