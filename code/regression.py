from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import os

DATA_FOLDER = "../data"
FILE_NAME = "adult.csv"
USE_SPLIT = 0.25


def end_line():
    print(
        "------------------------------------------------------------------------------------------------------------")


# Function to explore data and understand it
# Question 1.1
def understand_data():
    # using pandas to read csv as dataframe
    combined_df = pd.read_csv(os.path.join(DATA_FOLDER, FILE_NAME)).drop(['fnlwgt'], axis=1)
    label = combined_df['class']
    df = combined_df.drop(['class'], axis=1)
    num_of_instances = df.shape[0]
    num_of_null = df.isnull().sum().sum()
    total_values = num_of_instances * df.shape[1]
    instances_with_missing = (df.isnull().sum(axis=1) != 0).sum()
    print("There are %d instances.\nThere are %d missing values.\nFraction of missing values is %f.\nNumber of "
          "instances with missing values: is %d.\nFraction of instances with missing values over all instances is "
          "%f." % (num_of_instances, num_of_null, num_of_null / total_values, instances_with_missing,
                   instances_with_missing / num_of_instances))
    end_line()
    return [df, label, combined_df]


# Function to use OrdinalEncoder for representing attributes and values
# Question 1.2
def preprocess_data(df, label, isprint=False):
    oe = OrdinalEncoder()
    le = LabelEncoder()
    df = df.fillna('NaN')
    ordinal_array = oe.fit_transform(df, label)
    label_array = le.fit_transform(label)
    if isprint:
        print("The columns are")
        print(df.columns)
        print("The discrete values of classes are")
        print(oe.categories_)
        end_line()
    return [ordinal_array, label_array]


# Function to build Decision Tree Classiier
# Question 1.3
def build_decision_tree(df, drop=True, test=None, title=""):
    if drop:
        df = df.dropna()
    label = df['class']
    df = df.drop(['class'], axis=1)
    X, Y = preprocess_data(df, label)
    if test is None:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=USE_SPLIT, random_state=0)
    else:
        test_label = test['class']
        test = test.drop(['class'], axis=1)
        X_test, Y_test = preprocess_data(test, test_label)
        X_train, Y_train = [X, Y]
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)
    scores = cross_val_score(estimator=clf, X=X, y=Y, cv=10, n_jobs=4)
    print("The average score for 10 Fold Cross Validation is %f" % scores.mean())
    print("The average error rate for 10 Fold Cross Validation is %f" % (1 - scores.mean()))
    print("Classifier type is " + title)
    print("Training score is %f" % clf.score(X_train, Y_train))
    print("Test score is %f" % clf.score(X_test, Y_test))
    print("Training Error Rate is %f" % (1 - clf.score(X_train, Y_train)))
    print("Test Error Rate is %f" % (1 - clf.score(X_test, Y_test)))
    end_line()


def prepare_missing_data(df):
    null_instances = df[df.isna().any(axis=1)]
    remaining_instances = df.dropna()
    v = null_instances.shape[0]
    non_null_instances = remaining_instances.sample(n=v, replace=False)
    d = pd.concat([null_instances, non_null_instances])
    D_remaining = df.drop(d.index)
    D1 = d.fillna("missing")
    D2 = d
    for column in D2.columns:
        D2[column].fillna(D2[column].mode()[0], inplace=True)
    # This example assumes a 25% split for test class, so a third of samples in D1 are used
    separate_test_set = D_remaining.sample(n=int(2 * v * USE_SPLIT/ (1 - USE_SPLIT) ), replace=False)
    print(separate_test_set.shape[0])
    return [separate_test_set, D1, D2]


# Function to test effectiveness of handling missing values
# Question 1.4
def missing_values_experiment(df):
    test_set, D1, D2 = prepare_missing_data(df)
    build_decision_tree(D1, drop=False, title="Missing values padded with missing class label")
    build_decision_tree(D2, drop=False, title="Missing values padded with mode")


def main():
    df, label, combined_df = understand_data()
    data, labels = preprocess_data(df, label, isprint=True)
    build_decision_tree(combined_df, title="Decision Tree with Standard Split")
    missing_values_experiment(combined_df)


if __name__ == "__main__":
    main()
