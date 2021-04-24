# Credit_Risk_Analysis
Predicting credit risk using machine learning models and algorithms like resampling and boosting
# Project Overview 
Jill commends you for all your hard work. Piece by piece, you’ve been building up your skills in data preparation, statistical reasoning, and machine learning. You are now ready to apply machine learning to solve a real-world challenge: credit card risk.

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, you’ll need to employ different techniques to train and evaluate models with unbalanced classes. Jill asks you to use `imbalanced-learn` and `scikit-learn` libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, you’ll oversample the data using the `RandomOverSampler` and `SMOTE` algorithms, and undersample the data using the `ClusterCentroids` algorithm. Then, you’ll use a combinatorial approach of over- and undersampling using the `SMOTEENN` algorithm. Next, you’ll compare two new machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, to predict credit risk. Once you’re done, you’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk. This project have the following deliverables;
* Deliverable 1: Use Resampling Models to Predict Credit Risk
* Deliverable 2: Use the SMOTEENN Algorithm to Predict Credit Risk
* Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk
# Resources
* Data Source: `LoanStats_201901.csv`
* Data Tools/Software: Python 3.8.5, Jupyter Notebook 6.1.4, Visual Studio Code 1.55.2, Anaconda 4.10.1

# Deliverable 1: Use Resampling Models to Predict Credit Risk
> Using your knowledge of the `imbalanced-learn` and `scikit-learn` libraries, you’ll evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. First, you’ll use the oversampling `RandomOverSampler` and `SMOTE` algorithms, and then you’ll use the undersampling `ClusterCentroids` algorithm. Using these algorithms, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.
## Requirements/Results
Using the information we’ve provided in the starter code, create your training and target variables by completing the following steps:
* Create the training variables by converting the string values into numerical ones using the `get_dummies()` method.
* Create the target variables.
 
 ![1](https://user-images.githubusercontent.com/76136277/115974711-885ef280-a52c-11eb-86c4-08c5bd61b470.PNG)

* Check the balance of the target variables.
![2](https://user-images.githubusercontent.com/76136277/115974718-99a7ff00-a52c-11eb-8a6f-3d2770bcf7ed.PNG)

Next, begin resampling the training data. First, use the oversampling `RandomOverSampler` and `SMOTE` algorithms to resample the data, then use the undersampling `ClusterCentroids` algorithm to resample the data. For each resampling algorithm, do the following:

### RandomOverSampling Algorithm
* Use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
![3](https://user-images.githubusercontent.com/76136277/115974755-fdcac300-a52c-11eb-86bd-47e83dab1c82.PNG)

* Calculate the accuracy score of the model.
![4](https://user-images.githubusercontent.com/76136277/115974757-03280d80-a52d-11eb-90d2-f292dd99ab6c.PNG)

* Generate a confusion matrix.
 
 ![5](https://user-images.githubusercontent.com/76136277/115974761-08855800-a52d-11eb-8320-2c0aa49848b9.PNG)

* Print out the imbalanced classification report.
![7](https://user-images.githubusercontent.com/76136277/115974765-0c18df00-a52d-11eb-8aa5-4aa19ab3d0de.PNG)

### Undersampling - `ClusterCentroids` Algorithm
* Resample the data using the `ClusterCentroids` resampler
![cc](https://user-images.githubusercontent.com/76136277/115975268-45534e00-a531-11eb-9f40-29db839f0e15.PNG)

* Use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
![cc1](https://user-images.githubusercontent.com/76136277/115975260-25bc2580-a531-11eb-83c2-8b874b8993d9.PNG)

* Calculate the accuracy score of the model.
![cc3](https://user-images.githubusercontent.com/76136277/115975252-150baf80-a531-11eb-99d3-39c6ba69aeed.PNG)

* Generate a confusion matrix.
 
 ![cc4](https://user-images.githubusercontent.com/76136277/115975247-00c7b280-a531-11eb-9b77-bc1d9cc8ca94.PNG)

* Print out the imbalanced classification report.
![cc5](https://user-images.githubusercontent.com/76136277/115975241-f6a5b400-a530-11eb-9a05-e4cc0f519d4e.PNG)


# Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk
> Using your knowledge of the `imbalanced-learn` and `scikit-learn` libraries, you’ll use a combinatorial approach of over- and undersampling with the `SMOTEENN` algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from Deliverable 1. Using the `SMOTEENN` algorithm, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.
## Requirements/Results
**SMOTE Oversampling**
1. Using the information we have provided in the starter code, resample the training data using the `SMOTEENN` algorithm.
![8](https://user-images.githubusercontent.com/76136277/115974920-459e1a00-a52e-11eb-8408-8bfd27f4674a.PNG)

2. After the data is resampled, use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
![9](https://user-images.githubusercontent.com/76136277/115974936-7a11d600-a52e-11eb-81b7-aeb397e5b647.PNG)

3. Calculate the accuracy score of the model, 
![10](https://user-images.githubusercontent.com/76136277/115974940-8302a780-a52e-11eb-9f02-5d806d069891.PNG)

4. Generate a confusion matrix

![11](https://user-images.githubusercontent.com/76136277/115974942-8a29b580-a52e-11eb-95f6-5b4a9e280dc0.PNG)

5. Print out the imbalanced classification report.
![12](https://user-images.githubusercontent.com/76136277/115974949-944bb400-a52e-11eb-890b-8d769c18fa35.PNG)

**Combination (Over and Under) Sampling**
1. Using the information we have provided in the starter code, resample the training data using the `SMOTEENN` algorithm.
![SM1](https://user-images.githubusercontent.com/76136277/115975300-9400e800-a531-11eb-8744-1079a5469f7a.PNG)

2. After the data is resampled, use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
![SM2](https://user-images.githubusercontent.com/76136277/115975302-96634200-a531-11eb-92b9-92f6c94dc79b.PNG)

3. Calculate the accuracy score of the model, 
![SM3](https://user-images.githubusercontent.com/76136277/115975304-995e3280-a531-11eb-859b-eece9f80bda5.PNG)

4. Generate a confusion matrix

![SM4](https://user-images.githubusercontent.com/76136277/115975306-9c592300-a531-11eb-8743-998297f97aa7.PNG)

5. Print out the imbalanced classification report.
![SM5](https://user-images.githubusercontent.com/76136277/115975308-a3803100-a531-11eb-86d8-8f2597f8d044.PNG)

# Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk
### Requirements
1. Using the information we have provided in the starter code, create your training and target variables by completing the following:
    * Create the training variables by converting the string values into numerical ones using the `get_dummies()` method.
    * Create the target variables.
     ![es1](https://user-images.githubusercontent.com/76136277/115975386-68cac880-a532-11eb-99be-e306db94f966.PNG)

    * Check the balance of the target variables.
    ![es2](https://user-images.githubusercontent.com/76136277/115975391-72ecc700-a532-11eb-89e8-3bf5803c95ec.PNG)

2. The `BalancedRandomForestClassifier` algorithm does the following:
    * An accuracy score for the model is calculated 
    ![es3](https://user-images.githubusercontent.com/76136277/115975476-0d4d0a80-a533-11eb-8940-724abcf7b473.PNG)

    * A confusion matrix has been generated 
    ![es4](https://user-images.githubusercontent.com/76136277/115975478-15a54580-a533-11eb-90e0-0174d6c34a5e.PNG)

    * An imbalanced classification report has been generated 
    ![es5](https://user-images.githubusercontent.com/76136277/115975482-1e961700-a533-11eb-884c-75b0d6736f0b.PNG)

    * The features are sorted in descending order by feature importance 
    ![es6](https://user-images.githubusercontent.com/76136277/115975486-25248e80-a533-11eb-94c5-332a6494d095.PNG)

3. The `EasyEnsembleClassifier` algorithm does the following:
    * An accuracy score of the model is calculated 
   ![es7](https://user-images.githubusercontent.com/76136277/115975501-50a77900-a533-11eb-8d45-8308c64f04fb.PNG)

    * A confusion matrix has been generated 
    
    ![es8](https://user-images.githubusercontent.com/76136277/115975502-5735f080-a533-11eb-9318-700059fce1ae.PNG)

    * An imbalanced classification report has been generated 
   ![es9](https://user-images.githubusercontent.com/76136277/115975503-5d2bd180-a533-11eb-9047-8955758fdd58.PNG)

# Summary
For all models, `EasyEnsembleClassifier` is the most effective, this is because it provides the highest score for all risk loans. In summary,  utilizing `EasyEnsembleClassifier` will perform a High-Risk loan precision as a great value for the overall analysis.
