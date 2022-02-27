# Credit_Risk_Analysis
Using Python to build and evaluate several machine learning models to predict credit risk. Being able to predict credit risk with machine learning algorithms can help banks and financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.

## Overview
'Fast Lending' a peer-to-peer lending services company wants to use machine learning to predict credit risk. The management believes that this will provide a quicker and more reliable known experience. It also believes that machine learning will lead to a more accurate identification of good candidates for loans. The company wants me to assist the lead data scientist in implementing this plan. In my role I will build and evaluate several machine learning models and algorithms to predict credit risk. I will be using techniques such as resampling and boosting to make the most of the models and data. Once I have designed and implemented these algorithms, I will evaluate their performance and see how well these models predict the data. 

## Background

#### Machine Learning - 
Machine learning is the use of statistical algorithms to perform tasks such as learning from data patterns and making predictions. There are many different models—a model is a mathematical representation of something that happens in the real world. Broadly speaking, machine learning can be divided into three learning categories: supervised, unsupervised, and deep. 

![data-17-2-2-1-supervised-learning](https://user-images.githubusercontent.com/23488019/153807337-033b6c6e-df24-4a6c-a1f7-59a1120a2f33.png)

## Purpose 
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we need to employ different techniques to train and evaluate models with unbalanced classes. Jill asks me to use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I will oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, will use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, will compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once done, I will evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## What I am Creating
This project consists of three technical analysis deliverables and a written report. You will submit the following:

1. Deliverable 1: Use Resampling Models to Predict Credit Risk
2. Deliverable 2: Use the SMOTEENN Algorithm to Predict Credit Risk
3. Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk
4. Deliverable 4: A Written Report on the Credit Risk Analysis (README.md)


## Procedure: : 
There are multiple algorithms that are being used to learn and make their predictions based on the data. The final prediction is based on the accumulated predictions from each algorithm.
![data-17-8-1-1-multiple-algorithms-to-make-individual-predictions](https://user-images.githubusercontent.com/23488019/155899831-bd68c80c-83ea-4c5b-ac69-206792c42c00.png)

### Deliverable 1 - Use Resampling Models to Predict Credit Risk

Using  knowledge of the imbalanced-learn and scikit-learn libraries, here we will evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. First, we will use the oversampling RandomOverSampler and SMOTE algorithms, and then will use the undersampling ClusterCentroids algorithm. Using these algorithms, we will resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report. The steps followed are shown below:
- Create the training variables by converting the string values into numerical ones using the get_dummies() method. A dataset is split into training and testing sets in supervised learning. The model uses the training dataset to learn from it. It then uses the testing dataset to assess its performance. If one uses the entire dataset to train the model, one won't know how well the model will perform when it encounters unseen data. That is why it's important to set aside a portion of the dataset to evaluate our model.
- Create the target variables.
- Check the balance of the target variables.
Next, begin resampling the training data. First, use the oversampling RandomOverSampler and SMOTE algorithms to resample the data, then use the undersampling ClusterCentroids algorithm to resample the data. For each resampling algorithm, do the following:
- Use the LogisticRegression classifier to make predictions and evaluate the model’s performance.
- Calculate the accuracy score of the model.
- Generate a confusion matrix.
- Print out the imbalanced classification report.
Hence, For all three algorithms, an accuracy score for the model is calculated, a confusion matrix has been generated and an imbalanced classification report has been generated.


### Deliverable 2 - Use the SMOTEENN Algorithm to Predict Credit Risk
Using  the imbalanced-learn and scikit-learn libraries, we will use a combinatorial approach of over- and undersampling with the SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from Deliverable 1. Using the SMOTEENN algorithm, we will resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report. We will continue using the credit_risk_resampling.ipynb file where I have already created the training and target variables.
- Resample the training data using the SMOTEENN algorithm.
- After the data is resampled, use the LogisticRegression classifier to make predictions and evaluate the model’s performance.
- Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
Hence, in this deliverable, the combinatorial SMOTEENN algorithm does the following:
- An accuracy score for the model is calculated 
- A confusion matrix has been generated 
- An imbalanced classification report has been generated


### Deliverable 3 - Use Ensemble Classifiers to Predict Credit Risk
Using  the imblearn.ensemble library, we will train and compare two different ensemble classifiers, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk and evaluate each model. Using both algorithms, we will resample the dataset, view the count of the target classes, train the ensemble classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

- Creation of training and target variables by creating the training variables by converting the string values into numerical ones using the get_dummies() method, creating the target variables and checking the balance of the target variables.
Resample the training data using the BalancedRandomForestClassifier algorithm with 100 estimators.

- After the data is resampled using the BalancedRandomForestClassifier algorithm with 100 estimators, we will now calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.

- Print the feature importance sorted in descending order (from most to least important feature), along with the feature score.
- Next, resample the training data using the EasyEnsembleClassifier algorithm with 100 estimators.
- After the data is resampled, calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.

Hence for this deliverable,The BalancedRandomForestClassifier algorithm does the following:
- An accuracy score for the model is calculated 
- A confusion matrix has been generated 
- An imbalanced classification report has been generated 
- The features are sorted in descending order by feature importance 

Also, the EasyEnsembleClassifier algorithm does the following:
- An accuracy score of the model is calculated 
- A confusion matrix has been generated 
- An imbalanced classification report has been generated 

## Results
#### 1. Naive Random Oversampling 
In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.
![4](https://user-images.githubusercontent.com/23488019/155899401-426a1a4f-0e0a-4f09-99a4-27f9370da88d.PNG)

Here it can be noted that the balancd accuracy score is approximately 62.5%. This means that the model predicted the credit risk accurately 62.5% times. Althogh its a good score, it is not excellent. The high risk precision is about 1% only with 60% sensitivity. Due to the high number of low_risk population, its precision is almost 100% with a sensitivity of 65%. This shows that the precision scores for this type of model are greatly skewed towards the low risk loans. The low risk loans were predicted accurately however the high risk loans were not.Hence this model is not a good choice for identifying the high risk loans. 

#### 2. Smote Oversampling
The synthetic minority oversampling technique (SMOTE) is another oversampling approach to deal with unbalanced datasets. In SMOTE, like random oversampling, the size of the minority is increased. The key difference between the two lies in how the minority class is increased in size. As we have seen, in random oversampling, instances from the minority class are randomly selected and added to the minority class. In SMOTE, by contrast, new instances are interpolated. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.

![data-17-10-1-4-smote-generates-synthetic-data-points](https://user-images.githubusercontent.com/23488019/155899954-f6565c9d-3a02-45f7-b3f7-d525e26817da.png)
We use the SMOTE module from the imblearn library to oversample the minority class. The sampling_strategy argument specifies how the dataset is resampled. By default, it increases the minority class size to equal the majority class's size.
The fit_resample() method is used on the training data to train the SMOTE model and to oversample in a single step. The metrics of the minority class (precision, recall, and F1 score) are slightly improved over those of random oversampling.

![3](https://user-images.githubusercontent.com/23488019/155899395-9bf4eb79-119b-4c51-a6e9-79aaf7e7c116.PNG)
It's important to note that although SMOTE reduces the risk of oversampling, it does not always outperform random oversampling. Another deficiency of SMOTE is its vulnerability to outliers. Oversampling addresses class imbalance by duplicating or mimicking existing data.
It can be noted that the balanced accuracy score for this model is 65.12%. There is not much difference between this model and the previous model. Although the score is good its not excellent. The high risk precision is about 1% with 64% sensitivity. This model is not a good choice for trying to identify high risk loans just like the previous model. 



#### 3. Undersampling
Undersampling is another technique to address class imbalance. Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased. Undersampling only uses actual data. involves loss of data from the majority class. Furthermore, undersampling is practical only when there is enough data in the training set. There must be enough usable data in the undersampled majority class for a model to be useful.
![data-17-10-2-1-undersampling-diminishes-the-size-of-majority-class](https://user-images.githubusercontent.com/23488019/155900073-28a44e75-41cf-411e-b9c7-0a606422fd66.png)
The undersampling uses the cluster centroids to resample the data and reduce the majority class of training data to use in a logistic regression model. 
![data-17-10-2-4-cluster-centroid-undersampling-synthesizes-new-data-points](https://user-images.githubusercontent.com/23488019/155900839-e7ba456a-07a2-4ddd-9a5e-88725d20f131.png)
Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.

![2](https://user-images.githubusercontent.com/23488019/155901128-9a689b38-f879-4474-a763-aa96bbbb41f8.PNG)

Here the balanced accuracy score is 65%. The high risk precision here is still 1% with 60% sensitivity.  The low risk precision is still more accurate at 100% with sensitivity of 43%. This may be because of high number of false positives. 

Hence, this cannot be used as a great model for our predictions.

#### 4. Over and Under Sampling(SMOTEENN)
a downside of oversampling with SMOTE is its reliance on the immediate neighbors of a data point. Because the algorithm doesn't see the overall distribution of data, the new data points it creates can be heavily influenced by outliers. This can lead to noisy data. With downsampling, the downsides are that it involves loss of data and is not an option when the dataset is small. One way to deal with these challenges is to use a sampling strategy that is a combination of oversampling and undersampling.

SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN is a two-step process:

- Oversample the minority class with SMOTE.
- Clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.
![1](https://user-images.githubusercontent.com/23488019/155899384-0e96b8cc-a9e3-4447-8f9a-dccfce148af7.PNG)

Resampling with SMOTEENN did not work miracles, but some of the metrics show an improvement over undersampling. The balanced accuracy score is 64%.  The high risk precision is still 1% with only 70% sensitivity. Herer too due to high number of false positives, th elow risk sensitivity is 58%. Compared to th eprevious models this model is a little better but still not the best option.


#### 5. BalancedRandomForestClassifier Model
Instead of having a single, complex tree like the ones created by decision trees, a random forest algorithm will sample the data and build several smaller, simpler decision trees. 

![data-17-8-1-2-random-forest-with-three-decision-trees](https://user-images.githubusercontent.com/23488019/155901436-947884a8-90fa-44da-9e3d-fb93949f9352.png)

Each tree is simpler because it is built from a random subset of features:The Balanced Random Forest Classifier was used to create 100 decision trees to classify the testing data.

![e 2](https://user-images.githubusercontent.com/23488019/155899378-84b7bdc6-8569-4f88-acb2-345d42c0d465.PNG)

The balanced accuracy matrix is approximately 79%. This shows a lot of imprvement as compared to the previous models. The high risk precision is still low at 4% but a little more than the other ones that we have seen so far. The sensitivity for high risk precision is 67%. Here the value of F1 is also 7%. Since there are lower false positives, the low risk sensitivity is 91% with a precision of 100%.

#### 6. EasyEnsembleClassifier
The Easy Ensemble AdaBoost Classifier was used to train and evaluate models to classify the testing data.
![e 1](https://user-images.githubusercontent.com/23488019/155899371-a0c09059-98dd-4d61-90de-af5fa67d9d1a.PNG)
For this model the balanced accuracy score is 92.5% with a sensitivity of 91% and the F1 value is 14%. Here due to th elow number of false positives, the low risk sensitivity is 94% with a precision of 100%. This shows that there was a high rate of true positives in this model. 

## Summary
The results of the machine learning models conducted so far show that most of the models showed a much weak precision in determining if a credit risk is high. All of the models had low precision scores for the high-risk loans in accurately predicting positives. The balanced accuracy score for the models varied with the lowest score for the undersampling method and a high score with the AdaBoost classifier. It was noted that the EasyEnsembleClassifier model shows a recall of 94% so it detects almost all high risk credit. On the other hand, since most of the models had a low precision, a lot of low risk credits are  falsely detected as high risk which would penalize the bank's credit strategy and infer on its revenue by missing those business opportunities. The false positives helped in no way. 

Recommendation on the model to be used is the Easy Ensemble AdaBoost Classifier model because brought a lot more improvment specially on the sensitivity of the high risk credits. Although this model has the best numbers, it still has scope for improvement as the model is not perfect. This can be because there still can be a number of false positives which need to be verified. Hence lot more training and testing is required for the data so that a clear cut decision can be made. For those reasons I would not recommend the bank to use any of these models to predict credit risk because of the imperfections.



## Resources - 
1. https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html
2. https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html
3. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
4. https://www.khanacademy.org/math/algebra2/x2ec2f6f830c9fb89:logs/x2ec2f6f830c9fb89:log-intro/v/logarithms
5. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
6. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
7. https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html
8. https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html
 
