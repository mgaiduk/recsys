# recsys
Examples to set up ML training of recommendation models

## Plan
1. Get data from bq into tensors
    Lets assume we have the following BQ table:
    userId,postId,label
    We can export it to Gcs with google-provided dataflow template: https://cloud.google.com/dataflow/docs/guides/templates/provided-batch
    Example of how this can be parsed: https://github.com/mgaiduk/recsys/blob/main/example1_bq_to_tensors/main.py
    One downside of such "TFRecord" export is that labels are saved for each entry, so for our case we end up with 3x more data (which should also slow down training)
2. Proper tensorflow dataset from a file pattern on gcs
2. Train simple model on one machine
3. simple model on tpus + tfrs
4. add user history
    BigQuery windowed functions!