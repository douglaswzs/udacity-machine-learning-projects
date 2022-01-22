# Machine Learning Engineer Nanodegree Udacity
## Capstone Project Proposal

Douglas Wong


The data required for this project is available in Udacity Capstone Workspace


# Table of Contents
1. [References](#References)
1. [Domain Background](#DomainBackground)
1. [Problem Statement](#ProblemStatement)
1. [Solution Statement](#SolutionStatement)
1. [Datasets and Input](#DatasetsAndInputs)
1. [Benchmark Model](#BenchmarkModel)
1. [Evaluation Metrics](#EvaluationMetrics)
1. [Dependencies](#Dependencies)


# Project Dependencies<a name="Dependencies"></a>
* sagemaker
* matplotlib
* datetime
* warnings
* sklearn
* pandas
* numpy
* joblib
* boto3
* time
* os
* re


# Domain Background <a name="DomainBackground"></a>
Starbucks is by far one of the world's largest franchise coffee shops. They are known for having implemented one of the most aggressive information technology strategies, which has allowed them to grow into industry leaders. Starbucks debuted its mobile order-ahead app feature in late 2014 and it quickly caught on with Starbucks Rewards members (a). Starbucks has also become a leading mobile payment app that competes with Google Pay, Apple Pay, and Samsung Pay(c). Starbuck is also well-known for its loyalty program My Starbucks® Rewards where it able to offer individualized offers to their member. The program grew by 16 percent year over year in the first quarter of 2020, reaching 18.9 million active users (b). The membership increase is correlated to the increase in sales growth (c).  

According to Starbucks, a quarter of their transactions will be completed over the phone by the end of 2020 (a). This suggests that the rewards app accounts for a sizable portion of their revenue. In this capstone project, we want to look at how the customers used the Starbucks rewards app so that we can improve earnings through targeted offers to drive sales.  

This project contains simulated data from the Starbucks reward mobile app. The Starbucks app rewards registered customers on its platform to entice them to make purchases. There are 3 main types of offers that are sent to the customer. 
* Buy one get one Offer (BOGO)
* Discount Offer
* Informational Offer

In a BOGO offer, a user needs to spend a certain amount to get the reward. In the discount offer, the user receives a reward equal to the fraction of the amount spent. In an information offer, there will be no reward nor minimum amount spend. 

However not all customer response to the same marketing campaign, some customers will response to campaign regardless of reward such as recurring customer, while certain customer such as new customer need to be attracted via discount. 

The data that has been collected by the app reward is a data mine which offers us insights on customer base spending habits, and thus based on this valuable data, we can utilize this using Machine Learning to increase the ROI of the marketing campaign. 

# Problem Statement <a name="ProblemStatement"></a>
Every company invests money in marketing campaigns expecting that it will be successful in bringing more profit as an outcome. Therefore, it is imperative that we can increase the return on investment (ROI) by identifying the most effective offer type to be offered to the different subgroups of our customer base 

# Solution Statement <a name="SolutionStatement"></a>
Based on the dataset that we have which we obtained from the Starbucks reward mobile app, we are proposing to utilize machine learning methodology to build a model to predict the success of campaign offer type. This can allow us to determine which offer should be targeted at different subgroups of customers as well. 

The proposed plan was to merge the portfolio of offer, the profile and transcript together into a big dataset where we can analyze the data. We will also determine if a particular offer is successful based on the record of the transactions. 

We will consider a successful offer as follows. It must satisfy two conditions, offers received must be viewed, and the transaction must occur during the duration of the offer. If an offer were received but not viewed, it would mean that the customer would have made the purchase regardless of the offer. 

```sequence
Offer Received -> Offer Viewed
```
(Within the duration of the offer)
```sequence
Transaction Occurs -> Offer Completed
```
We will be utilizing the Amazon SageMaker platform for its integrated development environments, creating a Sagemaker notebook instance. It is a machine learning compute instance running the Jupyter Notebook App. We will use this environment to do our data processing, then we will upload our training data to Amazon S3 Cloud Object Storage.  

As we have a labeled dataset, we will be using supervised learning algorithms to predict if an offer is going to be successful. We will be exploring and comparing three algorithms, scikit-learn library logistic regression, scikit-learn library random forest, and XGBoost algorithm provided by Amazon SageMaker. Logistic regression is a statistical method for predicting binary classes (e). Random forests classifier is an ensemble of decision trees trained on randomly selected data samples, then the best prediction from each tree and select the best solution by voting(f). XGBoost is an efficient implementation of gradient boosting. Gradient boosting is an algorithm that combines many weak learning models together to create a strong predictive model (g).   

We will also determine the `Feature Importance` to estimate feature importance to describe how important that a particular feature in a model at predicting the success of an offer.  

Beside looking into if an offer is successful, we will also be looking into the amount of profit that each offer brings in as another metric that we can investigate. 

The problem that we have is a Supervised learning is a type of ML where the model is provided with labeled training data, and how it is "quantifiable, measurable, and replicable".
It is measure by the accuracy metric and F1 score.
We ensured this is replicable by using random

# Datasets and Inputs <a name="DatasetsAndInputs"></a>
The dataset used in this project contains data from the Starbucks reward mobile app. It contains the event of receiving offers, opening offers, and making purchases. In this simplified dataset. Only the type of offer and the transaction and the purchase amount are available in this dataset but not the actual product contributed to the purchase. 

The data is contained in three files:
- `portfolio.json` - containing offer ids and meta data about each offer (duration, type, etc.)
- `profile.json` - demographic data for each customer
- `transcript.json` - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

`portfolio.json`
- id (string) - offer id
- offer_type (string) - the type of offer ie BOGO, discount, informational
- difficulty (int) - the minimum required to spend to complete an offer
- reward (int) - the reward is given for completing an offer
- duration (int) - time for the offer to be open, in days
- channels (list of strings)

`profile.json`
- age (int) - age of the customer
- became_member_on (int) - the date when customer created an app account
- gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
- id (str) - customer id
- income (float) - customer's income

`transcript.json`
- event (str) - record description (ie transaction, offer received, offer viewed, etc.)
- person (str) - customer id
- time (int) - time in hours since the start of the test. The data begins at time t=0
- value - (dict of strings) - either an offer id or transaction amount depending on the record

# Benchmark Model <a name="BenchmarkModel"></a>
We will calculate the accuracy and F1 score from a Logistic Regression Models to create a baseline model which will then be used to compare it with all other subsequent Models. This is because this model is simple to set up and provide a reasonable good results.

# Evaluation Metrics <a name="EvaluationMetrics"></a>
To see how well our classification model performs, we will assess and compare the accuracy score and the F1 score (weighted average of the precision and recall value). Since the accuracy of the True positive and True negative is important in our cases, ie to see if the data model can successfully predict if an offer campaign will be successful. Depending on the class distribution of our data, we will also compare the F1 Score as it will give a better measure of the incorrectly classified cases than the Accuracy Metric.  

We can also determine the `Feature Importance` via random forest classifier to estimate feature importance to describe how important that a particular feature in a model. 

# Project Design <a name="ProjectDesign"></a>
The planned workflow for this project is as follows. 
1. Data Preparation
    - Data Cleaning
1. Feature Engineering
    - Joining Datasets
1. Data Exploration Analysis (EDA)
1. Splitting Data
    - Training Data - The main data used for training our model.
    - Testing Data - Using the testing data to measure the performance of our model 
1. Data Modeling
    - Logistic Regression - Training of the benchmark model 
    - Random Forest Regression - Training of comparison model 
    - Gradient Boosting - Training of comparison model 
1. Evaluating and comparing different model performances
1. Feature Importance
1. Discussion / Conclusion


# References  <a name="References"></a>
1. Starbucks Industry AI Case Study (Domain Background) https://info.formation.ai/rs/435-BMS-371/images/Starbucks_CaseStudy_final.pdf  
1. Starbucks Industry (Domain Background) https://brainstation.io/magazine/digital-loyalty-lifts-starbucks-q1-to-record-7-1-billion 
1. Starbucks Industry (Domain Background) https://www.geekwire.com/2021/quarter-starbucks-orders-u-s-now-paid-smartphone/ 
1. AI in marketing https://www.sciencedirect.com/science/article/pii/S2667096820300021 
1. Logistic Regression Definition  https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python 
1. RandomForest Definition https://www.datacamp.com/community/tutorials/random-forests-classifier-python 
1. XGBoost 
1. Accuracy vs. F1-Score https://medium.com/analytics-vidhya/accuracy-vs-f1-score-6258237beca2 
1. Baseline Model from contemporary works done on the same dataset https://medium.com/@sha821/starbucks-capstone-project-4b1eb8015bee 
1. Udacity Starbucks Capstone Challenge Notebook and Dataset 

