from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Fetch dataset from UCIML repository
def load_data():
    data = fetch_ucirepo(id=942)
    return data

# This function calculates the accuracy score of a given
# regularization model using the provided validation data.
def get_accuracy_score(regularisation, xVal, yVal):
    regularisation.fit(xVal, yVal.values.ravel())
    yPredictionVal = regularisation.predict(xVal)
    score = accuracy_score(yVal, yPredictionVal)
    return score

#Calculate PCA Accuracy
def get_pca_accuracy_score():
    # Principal Component Analysis (PCA) with 50 components on the
    # training data XTrain and applies the same transformation to the test data XTest.
    pca = PCA(n_components=50)  # Choose the number of principal components
    X_trained_data_pca = pca.fit_transform(XTrain)
    X_test_data_pca = pca.transform(XTest)

    # initializes and trains a logistic regression model with
    # a maximum of 1000 iterations using the principal components of
    # the training data X_trained_data_pca and corresponding target values yTrain.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_trained_data_pca, yTrain.values.ravel())

    # generates predictions using a trained model on the test data
    # after applying PCA transformation.
    y_pred = model.predict(X_test_data_pca)

    # Calculate accuracy score for PCA
    accuracy_result = accuracy_score(yTest, y_pred)
    return accuracy_result;

# Fetch dataset from UCIML repository
rt_iot2022 = load_data()

# Separate the dataset into features variables (X) and target variable (y).
X = rt_iot2022.data.features
y = rt_iot2022.data.targets

# Obtain a info about the dataset.
X.info();
y.info();

# Separates numerical and categorical attributes from the dataset X by
# removing the specified columns ("proto" and "service") and storing them in separate lists.
numerical_X = X.drop(["proto","service"],axis=1)
numerical_attributes = list(numerical_X)
categorical_attributes = ["proto","service"]

# pipeline for preprocessing numerical data,
# including imputation of missing values using the median and standardization of feature scaling.
pipeline_numdata = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

# a column transformer pipeline for preprocessing both numerical
# and categorical data, including one-hot encoding for categorical
# variables and passing through any remaining columns.
transformed_pipelinedata = ColumnTransformer([
             ("num", pipeline_numdata, numerical_attributes),
             ("cat", OneHotEncoder(), categorical_attributes),
         ],remainder='passthrough')

#  transforms the input data X using the specified pipeline for preprocessing
X_data_transformed = transformed_pipelinedata.fit_transform(X)

#  splits the transformed data into training and testing sets,
#  along with their corresponding labels, with a specified test size and random state.
XTrain, XTest, yTrain, yTest = train_test_split(X_data_transformed, y, test_size=0.25, random_state=50)

#Logistic regression starts

# Calculate accuracy score for No regularization with no solver
print("################NO REGULARIZATION - NO - SOLVER ###################")
none_regularization = LogisticRegression(penalty=None, max_iter=1000);

none_reg_traindata_accuarcy = get_accuracy_score(none_regularization, XTrain, yTrain)
none_reg_testdata_accuarcy = get_accuracy_score(none_regularization,XTest, yTest)

print("No reg accuracy score for Training data set : {0}".format(none_reg_traindata_accuarcy))
print("No reg  accuracy score for Test data set : {0}".format(none_reg_testdata_accuarcy))
print("####################################################################")

# Calculate accuracy score for No regularization with saga solver
print("################NO REGULARIZATION - SAGA- SOLVER ###################")
none_reg_saga_solver = LogisticRegression(penalty=None, solver='saga', l1_ratio=0.1)
none_reg_saga_solver.fit(XTrain, yTrain.values.ravel())
none_reg_y_val= none_reg_saga_solver.predict(XTrain)
none_reg_saga_traindata_score = accuracy_score(yTrain, none_reg_y_val)

none_reg_saga_solver.fit(XTest, yTest.values.ravel())
none_reg_y_test_val = none_reg_saga_solver.predict(XTest)
none_reg_saga_testdata_score = accuracy_score(yTest, none_reg_y_test_val)

print("No reg accuracy score for Training data set {0}".format(none_reg_saga_traindata_score))
print("No reg  accuracy score for Test data set {0}".format(none_reg_saga_testdata_score))
print("####################################################################")

# Calculate accuracy score for L1 regularization with saga solver
print("################L1 REGULARIZATION - SAGA- SOLVER ###################")
l1_reg_saga_solver = LogisticRegression(penalty='l1', solver='saga',max_iter=1000)
l1_reg_saga_solver.fit(XTrain, yTrain.values.ravel())
l1_reg_y_val = l1_reg_saga_solver.predict(XTrain)
l1_reg_saga_traindata_score = accuracy_score(yTrain, l1_reg_y_val)

l1_reg_saga_solver.fit(XTest, yTest.values.ravel())
l1_reg_y_test_val = l1_reg_saga_solver.predict(XTest)
l1_reg_saga_testdata_score = accuracy_score(yTest, l1_reg_y_test_val)

print("L1 reg accuracy score for Training data set {0} ".format(l1_reg_saga_traindata_score))
print("L1 reg accuracy score for Test data set {0}".format(l1_reg_saga_testdata_score))
print("####################################################################")

# Calculate accuracy score for L2 regularization with saga solver
print("################L2 REGULARIZATION - SAGA- SOLVER ###################")
l2_reg_saga_solver = LogisticRegression(penalty='l2', solver='saga',max_iter=1000)
l2_reg_saga_solver.fit(XTrain, yTrain.values.ravel())
l2_reg_y_val = l2_reg_saga_solver.predict(XTrain)
l2_reg_saga_traindata_score= accuracy_score(yTrain, l2_reg_y_val)

l2_reg_saga_solver.fit(XTest, yTest.values.ravel())
l2_reg_y_test_val = l2_reg_saga_solver.predict(XTest)
l2_reg_saga_testdata_score = accuracy_score(yTest, l2_reg_y_test_val)

print("L2 reg accuracy score for Training data set {0}".format(l2_reg_saga_traindata_score))
print("L2 reg accuracy score for Test data set {0}".format(l2_reg_saga_testdata_score))
print("####################################################################")

# Calculate accuracy score for L2 regularization with lbfgs solver
print("################L2 REGULARIZATION - LBFGS- SOLVER ###################")
l2_reg_lbfgs_solver = LogisticRegression(penalty='l2', solver='lbfgs',max_iter=1000)
l2_reg_lbfgs_solver.fit(XTrain, yTrain.values.ravel())
l2_reg_lbfgs_y_val = l2_reg_lbfgs_solver.predict(XTrain)
l2_reg_lbfgs_traindata_score = accuracy_score(yTrain, l2_reg_lbfgs_y_val)

l2_reg_lbfgs_solver.fit(XTest, yTest.values.ravel())
l2_reg_y_test_val1 = l2_reg_lbfgs_solver.predict(XTest)
l2_reg_lbfgs_testdata_score = accuracy_score(yTest, l2_reg_y_test_val1)

print("L2 reg accuracy score for Training data set {0}".format(l2_reg_lbfgs_traindata_score))
print(" L2 reg accuracy score for Test data set {0}".format(l2_reg_lbfgs_testdata_score))
print("####################################################################")

# Calculate accuracy score for L2 regularization with liblinear solver
print("################L2 REGULARIZATION - LIBLINEAR- SOLVER ###################")
l2_reg_liblinear_solver = LogisticRegression(penalty='l2', solver='liblinear',max_iter=1000)
l2_reg_liblinear_solver.fit(XTrain, yTrain.values.ravel())
l2_reg_liblinear_y_val = l2_reg_liblinear_solver.predict(XTrain)
l2_reg_liblinear_traindata_score = accuracy_score(yTrain, l2_reg_liblinear_y_val)

l2_reg_liblinear_solver.fit(XTest, yTest.values.ravel())
l2_reg_y_test_val2 = l2_reg_liblinear_solver.predict(XTest)
l2_reg_liblinear_testdata_score = accuracy_score(yTest, l2_reg_y_test_val2)

print("L2 reg accuracy score for Training data set {0}".format(l2_reg_liblinear_traindata_score))
print("L2 reg accuracy score for Test data set{0}".format(l2_reg_liblinear_testdata_score))
print("####################################################################")

# Calculate accuracy score for L2 regularization with liblinear solver
print("################L2 REGULARIZATION - NEWTON-CHOLESKY- SOLVER ###################")
l2_reg_newton_solver = LogisticRegression(penalty='l2', solver='newton-cholesky',max_iter=1000)
l2_reg_newton_solver.fit(XTrain, yTrain.values.ravel())
l2_reg_newton_solver_y_val = l2_reg_newton_solver.predict(XTrain)
l2_reg_newton_traindata_score = accuracy_score(yTrain, l2_reg_newton_solver_y_val)

l2_reg_newton_solver.fit(XTest, yTest.values.ravel())
l2_reg_newton_y_test_val2 = l2_reg_newton_solver.predict(XTest)
l2_reg_newton_testdata_score = accuracy_score(yTest, l2_reg_newton_y_test_val2)

print("L2 reg accuracy score for Training data set {0}".format(l2_reg_newton_traindata_score))
print("L2 reg accuracy score for Test data set{0}".format(l2_reg_newton_testdata_score))
print("####################################################################")

# Calculate accuracy score for Elastic net  regularization with saga solver
print("################ELASTIC_NET REGULARIZATION - SAGA- SOLVER ###################")
elastic_net_saga_solver = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.1)
elastic_net_saga_solver.fit(XTrain, yTrain.values.ravel())
elastic_net_saga_solver_y_val = elastic_net_saga_solver.predict(XTrain)
elastic_net_saga_traindata_score = accuracy_score(yTrain, elastic_net_saga_solver_y_val)

elastic_net_saga_solver.fit(XTest, yTest.values.ravel())
elastic_net_saga_y_test_val = elastic_net_saga_solver.predict(XTest)
elastic_net_saga_testdata_score= accuracy_score(yTest, elastic_net_saga_y_test_val)

print("Elastic net accuracy score for Training data set {0}".format(elastic_net_saga_traindata_score))
print("Elastic net accuracy score for Test data set {0}".format(elastic_net_saga_testdata_score))
print("####################################################################")

# Calculate accuracy score for PCA
print("################ PCA  ###################")

pca_accuarcy = get_pca_accuracy_score()
print("PCA accuracy score :", pca_accuarcy)
print("####################################################################")

# Calculate accuracy score for L1 regularization with liblinear solver
print("################L1 REGULARIZATION - LIBLINEAR- SOLVER ###################")
l1_reg_liblinear_solver = LogisticRegression(penalty='l1', solver='liblinear',max_iter=1000)
l1_reg_liblinear_solver.fit(XTrain, yTrain.values.ravel())
l1_reg_y_val1 = l1_reg_liblinear_solver.predict(XTrain)
l1_reg_liblinear_traindata_score = accuracy_score(yTrain, l1_reg_y_val1)

l1_reg_liblinear_solver.fit(XTest, yTest.values.ravel())
l1_reg_y_test_val2 = l1_reg_liblinear_solver.predict(XTest)
l1_reg_liblinear_testdata_score = accuracy_score(yTest, l1_reg_y_test_val2)

print("L1 reg accuracy score for Training data set {0}".format(l1_reg_liblinear_traindata_score))
print("L1 reg accuracy score for Test data set {0}".format(l1_reg_liblinear_testdata_score))
print("####################################################################")
# regularization techniques, optimization algorithms, and model performance metrics.
regularization_methods = ['No Regularization',  'L1 ', 'L2','Elastic-net ', 'PCA']
accuracy_scores_test = [none_reg_saga_testdata_score,l1_reg_saga_testdata_score,  l2_reg_saga_testdata_score, elastic_net_saga_testdata_score,  pca_accuarcy]

print("PCA accuracy score test :", accuracy_scores_test)
print("####################################################################")