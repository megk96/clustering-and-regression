regression.py

The regression.py consists of 5 different methods.

understand_data()
This method is to read the .csv file as a pandas DataFrame, remove the unwanted columns.
It also calculates the number of null instances and compares the fraction to total number of instances.
It also calculates a few other metrics and prints them.
Returns DataFrame, DataFrame without label, Label separately. Will be useful for future methods.

preprocess_data()
This uses OrdinalEncoder from sklearn to encode the data values into discrete values arranged ordinally.
It uses LabelEncoder to encode the labels: i.e 0 or 1 for the two classes present.
It outputs all possible values for the columns.
Returns the ordinal array and label array.

build_decision_tree(df, drop=True, test=None, title="")
This is a function used for both 1.3 and 1.4. It uses drop=True as default and for part 4, we explicitly put drop=False since we want to retain missing instances.
It performs a standard train_test_split, and the value of the split can be modified using constant USE_SPLIT in the code.
A 10 Fold Cross Validation is also done and the score is reported. This is to ensure that overfitting isn't occuring.

prepare_missing_data(df)
Accepts DataFrame as input and prepares D', D1' and D2' accordingly.
The new DataFrames are returned

missing_values_experiment(df)
This is the 1.4 part of the assigment and calls the relevant functions to carry over the experiment.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------

clustering.py

The clustering.py file consists of 3 different methods.

understand_data()
This method is to read the .csv file as a pandas DataFrame, remove the unwanted columns, and to output the mean, minimum and maximum ranges for each attribute.
It does not take any arguments and it returns the modified dataframe df, which will be used in further steps.

k_means_combos(df, K)
This method is to apply k means with value K applied on the DataFrame df (taken from arguments).
It uses the sklearn KMeans implementation and uses matplotlib scatterplot to plot the 15 different values (6C2, 6 attributes taken 2 at a time)
It stores the 15 plots in the folder plots, after added title, legend.

k_means_experiment(df, kset)
It implements the kmeans algorithm from sklearn from the list argument kset on the DataFrame df.
It uses Scipy and Numpy libraries to store matrices and computations, for euclidean distance.
The WC and BC metrics are calculated using the formula and these libraries.
The WC value is also inertia, obtained from the kmeans model sklearn. This value is verified by calculating it separately using within_cluster_verfication(dist)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------