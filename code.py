import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
h2o.init()

# import the boston dataset:
# this dataset looks at features of the boston suburbs and predicts median housing prices
# the original dataset can be found at https://archive.ics.uci.edu/ml/datasets/Housing
boston = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/BostonHousing.csv")

# set the predictor columns
predictors = boston.columns[:-1]

# this example will predict the medv column
# you can run the following to see that medv is indeed a numeric value
boston["medv"].isnumeric()
[True]
# set the response column to "medv", which is the median value of owner-occupied homes in $1000's
response = "medv"

# convert the `chas` column to a factor
# `chas` = Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
boston['chas'] = boston['chas'].asfactor()

# split into train and testing sets
train, test = boston.split_frame(ratios = [0.8], seed = 1234)

# set the `alpha` parameter to 0.25
# then initialize the estimator then train the model
boston_glm = H2OGeneralizedLinearEstimator(alpha = 0.25)
boston_glm.train(x = predictors,
                 y = response,
                 training_frame = train)

# predict using the model and the testing dataset
predict = boston_glm.predict(test)

# View a summary of the prediction
predict.head()