import h2o
from h2o.estimators.isolation_forest import H2OIsolationForestEstimator
h2o.init(ip="h2o-h2o-3.h2o-system.svc.cluster.local", port=54321)

# import the ecg discord datasets:
train = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/anomaly/ecg_discord_train.csv")
test = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/anomaly/ecg_discord_test.csv")

# build a model using the `sample_size` parameter:
isofor_model = H2OIsolationForestEstimator(sample_size = 5, ntrees = 7, seed = 12345)
isofor_model.train(training_frame = train)

# test the predictions and retrieve the mean_length.
# mean_length is the average number of splits it took to isolate
# the record across all the decision trees in the forest. Records
# with a smaller mean_length are more likely to be anomalous
# because it takes fewer partitions of the data to isolate them.
pred = isofor_model.predict(test)
pred