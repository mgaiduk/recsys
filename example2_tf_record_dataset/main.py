import tensorflow as tf

prefix = "gs://mybucket/pool"
path = f"{prefix}/train/output-00000-of-00001.tfrecord"
pattern = f"{prefix}/train/output-*"


class DatasetReader:
    def __init__(self, input_path, batch_size=1024, epochs=2):
        self.input_path = input_path
        self.batch_size = batch_size
        self.epochs = epochs

    def __call__(self, ctx: tf.distribute.InputContext):
        @tf.function
        def decode_fn(record_bytes):
            return tf.io.parse_example(
                # Data
                record_bytes,
                # Schema
                {"label_vplay98": tf.io.FixedLenFeature([], dtype=tf.int64),
                 "userId": tf.io.FixedLenFeature([], dtype=tf.int64),
                 "postId": tf.io.FixedLenFeature([], dtype=tf.int64)}
            )

        def make_dataset_fn(path):
            dataset = tf.data.TFRecordDataset([path]).batch(
                self.batch_size).repeat(self.epochs).map(decode_fn)
            return dataset
        dataset = tf.data.Dataset.list_files(
            self.input_path, shuffle=True, seed=42)
        dataset = dataset.interleave(
            make_dataset_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset


# simple 1-host strategy, can be replaced with tpu strategy later
strategy = tf.distribute.get_strategy()
dataset_callable = DatasetReader(
    input_path=pattern
)
dist_dataset = strategy.distribute_datasets_from_function(
    dataset_fn=dataset_callable
)

for elem in dist_dataset:
    break
elem  # get batched tensor!
