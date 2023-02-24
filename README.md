# recsys
Examples to set up ML training of recommendation models

## Plan
1. Get data from bq into tensors
    Lets assume we have the following BQ table:
    userId,postId,label
    We can export it to Gcs with google-provided dataflow template: https://cloud.google.com/dataflow/docs/guides/templates/provided-batch
    
2. Train simple model on one machine
3. simple model on tpus + tfrs
4. add user history
    BigQuery windowed functions!