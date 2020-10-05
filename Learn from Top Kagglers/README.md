## Kaggle
It is all about data and making things work, not about algorithms itself.
- everyone can and will tune classic approaches
- we need some insights to win

There is no "silver bullet" algorithm.

## Week 1
One tool is [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit), designed to provide blazing speed and handle really large data sets, which don't fit into memory.
`libfm` and `libffm` implement different types of optimization machines, and often used for sparse data like click-through rate prediction.


### numeric features
- Tree-based model
- Non-tree-based model: KNN, linear model, NN.
	- Need to scale the features in non-tree-based models.
	- For example, in KNN, slightly difference in some features will result in very big impact on predictions, making the KNN favouring the features over all others.
	- Also, we want the regularization to be applied to linear models coefficients for features in equal amount. But in fact, regularization impact turns out to be proportional to feature scale. Same reason for NN.
	- GD can go crazy with different feature scaling.

#### scaling
- To `[0,1]` by applying `sklearn.preprocessing.MinMaxScaler`
$X = (X - X_{\text{min}})/(X_{\text{max}} - X_{\text{min}})$
Distributions do not get changed!

- To mean = 0 and std = 1 by `sklearn.preprocessing.StandardScaler`

#### outliers
- Clip feature values between two chosen values such as 1% and 99% percentiles. Well-known in financial data and it is called `winsorization`
- Rank transformation. `scipy.stats.rankdata`. Note you need to store the mapping from features values to their rank values when applying to testing data.
#### often help non-tree-based methods, especially neural networks
Drive big values to the features' average value. Values near zero becomes more distinguishable.
- `np.log(x+1)`
- `np.sqrt(x+2/3)`

#### feature generation
Although gradient within decision tree is a very powerful model, it still experiences difficulties with approximation of multiplications and divisions. And adding those features explicitly can lead to a more robust model with less amount of trees.
Based on what? Domain knowledge.
*Examples:*
- price per square feet
- If we have prices of products as a feature, we can add new feature indicating fractional part of these prices. For example, if some product costs 2.49, the fractional part of its price is 0.49. 0.99 -> 0.99. 1.0 -> 0.0.
This feature can help the model utilize the differences in people's perception of these prices.

### categorical and ordinal features
#### label encoding
Not very effective in non-tree-based methods. Usually useful for tree-based methods.
1. Alphabetical (sorted) e.g. `[S,C,Q] -> [2,1,3]`, implemented by `sklearn.preprocessing.LabelEncoder`
2. Order of appearance, default behaviour of `pandas.factorize`
#### frequency encoding
Usually useful for tree-based methods.
```python
encoding = titanic.groupby('Embarked').size()
encoding = encoding/len(titanic)
titanic['enc'] = titanic['Embarked'].map(encoding)
```
e.g. `[S,C,Q] -> [0.5,0.3,0.2]`

*Challenge:* how to break the ties for features having the same frequency?

#### one-hot encoding
Already scaled. Usually useful for non-tree-based methods. Consider using the sparse matrices to store OHE features due to sparsity. So it means we only need to store non-zero values and locations.

#### feature generation
Interaction between several categorical features, usually useful for non-tree-based methods.

### datetime and coordinates
#### datetime
1. Periodicity: day number in week, month, season, year second, minute, hour
2. Time since:
	- row-independent moment for example since 00:00:00 UTC, 1 Jan, 1970
	- row-dependent important moment for example, number of days left until next holiday/time passed after last holiday/since the last campaign, number of days until expiration/next campaign
3. Difference between dates of two events

#### coordinates
1. Find interesting places or centres of clusters from training data, compute the distance from these clusters.
2. Aggregate stats such as the average house price in that area.

### missing values
- Hidden NaNs can be found by hist.
- Sometime we can treat outliers as missing values.
- Treating values which do not present in train data but appear in the testing data. Sometimes it's okay to concatenate training and testing (without knowing the labels) together and apply some preprocessing steps. Works in batch prediction like the Kaggle competition but not for online prediction. For example, in training + testing data, category D only appeared once in training set, category C only appeared once and only in testing set. Then if we apply the frequency encoding to this categorical features, we would treat category C same as category D.
 - Usual way to deal with missing values is to replace them with -999, mean or median.
 - In general, avoid filling NaNs before feature generation.
 - Create a flag feature indicating binary feature `isnull` or not can be beneficial.
 - XgBoost can handle missing values.

### feature extraction from texts
0. Text preprocessing
	- lowercase
	- lemmatization and stemming (e.g. have, had, car, cars, democracy, democratic, democratization), careful with stemming
	- remove stop words, e.g. in `sklearn.feature_extraction.text.CountVectorizer` there is an argument called `max_df` which is the threshold of words we can see, after which, the word will be removed from the text corpus.
1. Bag of words
	- `sklearn.feature_extraction.text.CountVectorizer`
	- TF: normalize sum of values in a row
	- IDF: to boost more important features by normalizing each feature by the inverse fraction of documents
	- `sklearn.feature_extraction.text.TfidfVectorizer`
	- N-grams to help to use local context  `sklearn.feature_extraction.text.CountVectorizer: Ngram_range, analyzer`
2. Embedding such as `word2vec` which uses the nearby words. Implementation: `Word2vec, Glove, FastText` or `Doc2Vec` -- all pre-trained models.

### feature extraction from images
1. train CNNs from scratch
2. fine tuning pre-trained CNNs
3. features can be extracted from different layers
4. images augmentation by orientation when you have small sample size

## Week 2

### EDA
- May help us to find *magical* features
- Get domain knowledge -> understand the data dictionary
- Check if the data is intuitive such as the 0 < age < 120, clicks <= impression
- Understand how the data was generated so that we can set up a proper validation scheme

### Anonymized data
- Hashing texts or hide column meaning
- Guess the meaning and types of the columns
e.g. Run a baseline model and print out the feature importance, find out some interesting features. Print out the statistics, could be standardized scaled. Now trying to find the scaling parameter and shift parameter. How? we can first take a look at the difference between neighbouring values.
```python
x_unique = train_data.x.unique()
x_unique_sorted = np.sort(x_unique)
np.diff(x_unique_sorted)
```
Let's say the the most of the diffs are `0.05`. If we _guess_ the diff value for the original feature was `1.0` so let's divide all the numbers by `0.05` to get the right scaling.
Observe the fractional part of the numbers and subtract something to shift back.

### Visualizations
- Histograms: `plt.hist(x)`. Watch for the peak, could it happen to be the mean value? So it means that it contains some missing values originally.
- Plot where x axis is the row index and y axis is the feature value. `plt.plot(x, '.')` or `plt.scatter(range(len(x)), x, c=y)` to see the data is properly shuffled.
- Stats: `df.describe()`
- Relationship between x1 and x2 `plt.scatter(x1, x2)`  group by class label (0,1, test)
-  `df.corr()` or `plt.matshow()` or `pd.scatter_matrix(df)`
- Plot: index v.s. feature statistics such as feature mean to group features and sort by the statistics `df.mean().sort_values().plot(style='.')`

### Dataset cleaning and other things to check
- Remove constant features: `train.nunique(axis=1) == 1` also apply the same check on the train+test merged dataset
- Drop duplicates: `traintest.T.drop_duplicates()`. Check if the all feature values are the same but have the different labels. Understand why.
- Interesting to compare two dates and calculate the difference

### Validate
#### Data splitting strategies
- Holdout: `sklearn.model_selection.ShuffleSplit` where `ngroups=1`
- K-fold: `sklearn.model_selection.Kfold` where `ngroups=k`
- Leave-one-left: `sklearn.model_selection.LeaveOneOut` where `ngroups=df.shape[0]`
#### Stratification
preserves the same target distribution over different folds, useful for imbalanced datasets.
### Time-based split
- One-timestamp random split
- Moving window validation. e.g. week1 ~ 3 to train, week4 to test. week1 ~ 4 to train, week5 to test. week1 ~ 5 to train, week6 to test.
- ID-based split. e.g. For the same user, reserve historical data as training and future actions as the testing.
### Things to consider during validation stages
- Different scores and optimal parameters: should average scores from different K-fold splits (using different random seeds)
- Tune parameters using one fold and test the model on another fold.
### Things to consider during submission stages
- Training and testing have different distributions: set up the validation set which mimics the testing distribution.
### Metrics
#### Regression metrics
- MSE
- $\text{RMSE} = \sqrt{\text{MSE}}$ same scale as the target variable. So even though RMSE and MSE are really similar in terms of models scoring, they can be not immediately interchangeable for gradient based methods. We will probably need to adjust some parameters like the learning rate.
- $R^2$: absolutely equivalent that to optimize $R^2$ is to optimize MSE.
- MAE: used when there are outliers
- MSPE/MAPE: Mean Square/Absolute Percentage Error, a relative metric, error divided by the target variable value
- RMSLE: Root Mean Square Logarithmic Error =
$$\text{RMSE}(log(y_i + 1), log(\hat{y}_i + 1) ) = \sqrt{\text{MSE}(log(y_i + 1), log(\hat{y}_i + 1) )}$$
cares more about relative errors than absolute errors
#### How to optimize regression metrics
- MSE: most common one; you can use most libraries out of the box
- MAE: absolute value difference
- Huber loss: balance between the MAE and the Mean Squared Error. Robust to outliers.
- MSPE is just a weighted version of MSE so we just need to assign proper sample weights, $w_i = \frac{1/y^2_i}{\sum_{i=1}^N 1/y^2_i}$, to optimize for MSPE
- MAPE is just a weighted version of MAE so we just need to assign proper sample weights, $w_i = \frac{1/y_i}{\sum_{i=1}^N 1/y_i}$, to optimize for MAPE
- If the library does not support `sample_weights` then we can do `df.sample(weights=sample_weights)` and use any model that optimize MSE or MAE. Usually we do the resampling multiple times and take the average.

#### Classification metrics
- Accuracy: how frequent our class prediction is correct
- Log loss in binary classification setting. What's the best constant as prediction? The answer is the class frequency.
- AUC: only in binary classification setting. Care only about the ordering not the absolute values. *X-axis*: False Positive. *Y-axis*: True Positive.  **Interpretation:** $\frac{\text{\# of correctly ordered pairs}}{\text{\# of all pairs}}$ such that a pair is made up with one from the class 0 and the other one is from class 1.

#### How to optimize classification metrics
- Log loss: implemented everywhere but be careful about `sklearn.RandomForestClassifier` turns out to be quite bad in terms of log loss. If the classifier doesn't directly optimize log loss, we should calibrate the probabilities. How?
- Probabilities calibration
	- Platt scaling: just fit logistic regression to your predictions
	- Isotonic regression: just fit isotonic regression to your predictions
	- Stacking: just fit XgBoost or NN to your predictions
- Accuracy: hard to optimize directly
- AUC: possible to directly optimize AUC but not trivial

### Target (mean) encoding
One of the drawbacks of Gradient Boosting Tree-based methods such as XgBoost or LightGBM is the inability to handle high cardinality categorical features.
#### How
- Encode each categorical feature with target mean, i.e. group by the categorical feature and compute the mean by group.
```python
means = X_train.groupby(col).target.mean()
X_train[col+'_mean_target'] = X_train[col].map(means)
validation[col+'_mena_target'] = validation[col].map(means)
```
- Another way to use the target variable is through the weight of evidence $\text{WoE} = ln \frac{Goods}{Bads} * 100$. Goods = number of 1s
#### Regularization
Target encoding easily gets overfitting as it may reveal the target information.
- Using CV
```python
from sklearn.model_selection import StratifiedKFold
y_tr = df_train['target'].values # target variable
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

for tr_ind, val_ind in skf.split(df_train, y_tr):
  X_tr, X_val = df_train.iloc[tr_ind], df_train.iloc[val_ind]
    for col in cols:
    # iterate through the columns we want to encode
	means = X_val[col].map(X_tr.groupby(col).target.mean())
	X_val[col+'_mean_target'] = means

 df_train_new.iloc[val_ind] = X_val
 # add encoded features into another new training data frame

prior = df_train['target'].mean() # global mean
df_train_new.fillna(prior, inplace=True) # fill NaNs with global mean
```
- Smoothing: $\alpha$ controls the amount of regularization. When = 0 there is no regularization. When it goes to infinity, it approaches to global mean.
$$
\frac{\text{mean(target) * nrows + globalmean}*\alpha}{\text{nrows+} \alpha}
$$
## Week 3
### Hyperparameter tuning
| XGBoost | LightGBM |
|--|--|
| `max_depth` | `max_depth/num_leaves` |
|`subsample`|`bagging_fraction`|
|`colsample_bytree` or `colsample_bylevel`| `feature_fraction` |
|`eta`|`learning_rate`|
|`num_round`|`num_iterations`|
|`min_child_weight`|`min_data_in_leaf`|
|`lambda`|`lambda_l1`|
|`alpha`|`lambda_l2`|

#### Neural Network (fully-connected layers)
- L2/L1 for weights
- Dropout
- Static drop connect (use a very large neural network for the first hidden layer but randomly drop 99% of connection between the input layer and the first hidden layer)
#### Tips
- Average everything
	- Over different random seeds
	- Over small deviations from optimal parameters. For example average `max_depth=4,5,6` for an optimal 5

### Practical guides
- Sort the samples by individual loss in descending order ... always find something interesting (fault examples etc...)
- Data loading:
	- convert csv/txt files to hdf5 or npy for much faster loading
	- Pandas dataframe defaults to 64-bit arrays, you may downcast it to 32-bits safely
	- Read and process data in chunks
- Start with fastest models - Light GBM, use early stopping so you don't need to tune the number of iterations
- One notebook per submission
- Before creating a submission restart the kernel by "restart and run all"

### Competition pipeline
- EDA
	- Plot histograms of variables. Check that a feature looks similar between train and test
	- Plot features v.s the target variable v.s time (e.g age band v.s. probability of buy)
	- Consider univariate predictability metrics (IV, $R^2$, AUC)
	- Binning numerical features
- Decide a CV strategy
	- Is time important? Time-baed validation!
	- Different entities than the train? Stratified validation!
	- Is it completely random? Random K-fold!
	- Combination of all the above
### Advanced feature engineering
- For example in a CTR task, we can create the user-page pair statistics such as the `min_price`, `max_price`
```python
gb = df.groupby(['user_id', 'page_id'], as_index=False).agg({'ad_price': {'max_price': np.max, 'min_price': np.min}})
gb.columns = ['user_id', 'page_id', 'min_price', 'max_price']
```
- Or we can add features such as
	- how many pages a user has visited
	- standard deviation or CV of prices
	- most visited page
- What if we don't have explicit group? Try to use nearest neighbours
- Feature interactions
	- For example there are `f1` and `f2` we can do join -> creating a new variable `f_join` and then do OHE, it's the same as we OHE `f1` and `f2` separately and do a pairwise multiplications
	- Sum/Diff/Division
	- Fit a RF and select the most important features
	- Extract high-order interactions from decision trees, i.e. map each leaf into a binary feature --> build new categorical features
	```python
	# scikit-learn
	tree_model.apply()
	# Xgboost
	booster.predict(pred_leaf=True)
	```
	- [reference](https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html)
	- [FB paper](https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf)

### [t-SNE](https://distill.pub/2016/misread-tsne/)
- t-SNE is one of manifold learning methods which projects points from the high dimensional space to low dimensional space so that the distances between points are approximately preserved. Other [manifold learning](https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html) methods
- Important to select the proper perplexities! choose different ones (5 -100). Due to stochastic nature, tSNE provides different projections even for the same data/hyper parameters.
- We can also create new features using tSNE by concatenating the transformers coordinates to original feature matrix.
- library `tSNE` tends to faster than implementations in scikit-learn

### Ensemble methods
- Simple average
- Weighted average
- Bagging: *Random Forest* `BaggingClassifier` and `BaggingRegressor`
- Boosting: a form of weighted averaging of models where each model is build sequentially via taking into account the past model performance
	- adaptively adjusting weights which is proportional to absolute value of residuals. e.g. `AdaBoost`
	-  most dominant one: residual based boosting. Predict the residuals (care about the direction) using the original features and add *all* predictions together. Fully gradient based or [DART](https://xgboost.readthedocs.io/en/latest/tutorials/dart.html)
- Stacking
	- Let's say we have 3 datasets, A (training), B (validation) and C (testing).
	- We have the labels to A and B but not for C
	- Train algorithm **0** on A and make predictions for B and C, save to B1 (columns: `pred0` and `y`) and C1 (columns: `pred0`)
	- Train algorithm **1** on A and make predictions for B and C, save to B1 (columns: `pred0` , `pred1` and `y`) and C1 (columns: `pred0`, `pred1`)
	- Train algorithm **2** on A and make predictions for B and C, save to B1 (columns: `pred0` , `pred1` , `pred2` and `y`) and C1 (columns: `pred0`, `pred1`, `pred2`)
	- Train a **meta learner** on B1 and make predictions for C1
	- Diversity comes from different algorithms or different input features
	- Performance plateauing after certain number of models
	- Meta model is normally modest
- StackingNet: a scalable meta modelling methodology that utilizes stacking to combine multiple models in a neural network architecture of multiple levels
- Tips in 1st level stacking
	- diversity based on algorithms:
		- 2-3 gradient boosted trees (lightGBM, XGBoost, H2O, catboost)
		- 2-3 NN
		- 1-2 ExtraTrees/RandomForest
		- 1-2 Linear models (LR, Ridge regression, Linear SVM)
		- 1-2 KNN models
		- 1 Factorization machine (`libfm`)
		- 1 SVM with non-linear kernel
	- diversity based on input data
		- categorical features: OHE, Label encoding, Target encoding
		- numerical features: outliers, binning, percentiles
		- interactions: col1 +-*/ col2, `groupby`, unsupervised learning such as PCA
- Tips in 2nd level stacking
	- simpler algorithms
		- gradient boosted trees with small depth like 2 or 3
		- linear models with high regularization
		- extra trees
		- shallow networks as 1 hidden layer
		- KNN with Bray Curtis distance
		- Brute forcing a search for best linear weights based on CV
	- feature engineering
		- pairwise differences between meta features
		- row-wise statistics like average or std
		- standard feature selection
- [Parameter tuning](https://github.com/kaz-Anova/StackNet/blob/master/parameters/PARAMETERS.MD#xgboostclassifier)
### CatBoost
#### Categorical data
- OHE
- Number of appearances
- Statistics with label usage on a random permutation of the data (resulting the best performance but may be overfitting)
#### What's new
- Ordered boosting: calculate leaf values based only on past objects in the permutations
- There are some ways to speed up training in CPU (`rsm, max_ctr_complexity, boosting_type`)
- GPU supported
- Overfitting detector
- Evaluating custom metrics during training
- User defined metrics and loss functions
- `NaN` features support (categorical: treat NaNs as a separate category; numeric: substitute with the variable that is either greater or less than all other values so it is guaranteed that when using internal split selection procedures the algorithm will consider putting the objects with NaNs in a separate leaf)
- Feature importance
- Hyper-parameters tuning
	- Number of trees + learning rate
	- Tree depth
	- L2 regularization
	- Bagging temperature (higher the temperature, the more aggressive the sampling is)
	- Random strength
## Week 5
Kaggle [past solutions](http://ndres.me/kaggle-past-solutions/)
