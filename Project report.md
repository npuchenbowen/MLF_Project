# PHBS_MLF_2019_Project
## Catching the mouse_identify fraudulent credit card transactions

Yuchuan Xu (1801212958)
Yang Cao (1801212825)
Alimujiang (1801212778)
Bowen Chen (1801212827)


**1.	Research background**

In recent years, there have been a lot of credit card swiping incidents. According to research and statistics from Nielsen Consulting, in 2017, global credit card fraud losses amounted to 22.8 billion US dollars and are expected to continue to grow [1]. Although credit card swiping transactions account for only a very small portion of all credit card transactions, once it occurs, it will cause unnecessary losses to the credit card holder, and sometimes the amount of the loss is huge.
Specifically, credit card swiping refers to illegal or criminal acts in which criminals copy other people ’s credit cards in various illegal forms, thereby stealing cardholder ’s funds. In real life, the main manifestation is that the cardholder’s fund is stolen by an unfamiliar third party without the loss of the credit card and the payment password is not informed to others.

------------



**2.	Motivation**

Since credit card swiping would cause financial losses to the holders and affect the efficiency of financial institutions, it is necessary to design an automatic fraud detection system for high-precision fraud detection. This project hopes to use a set of credit card consumption data sets to train a classifier to distinguish whether the user's credit card use records are fraudulent information, which could help the credit card company effectively identify fraudulent credit card transactions.

------------



**3.	Data description**

1)	outline

The data set used in this project is a group of European cardholders ’credit card transaction data in September 2013 (which has been vectorized). This data set contains transactions that occurred within two days, a total of 284,807, of which 492 belong Piracy transactions, the rest belong to normal transactions, the data set is very unbalanced, piracy transactions only account for 0.172% of all transactions. The data set contains 31 descriptive indicators. The introduction and descriptive statistics of the indicators are shown in the following section.

2)	Variables

The data set contains a total of 31 indicators. The indicator Class is the response variable. If the transaction is a fraudulent transaction, the value is 1. If the transaction is a normal transaction, the value is 0. The indicator Time describes the time when the transaction occurs. Specifically, it is the interval between the time of each transaction and the time of the first transaction; the indicator Amount is the amount of the transaction; the indicators v1, v2-v28 are obtained through PCA. Due to the privacy issue, the meaning of v1-v28 are not available.

![](http://m.qpic.cn/psc?/V11zaUPV2EE2Gc/8YUQ4vKPKp.vxIKbDZcdtk2gilby.ETBA2v*SvSmiEv4iqZir4kBxVQZhRorC7eC5P.zzc8jLllD5LvhVtWkHw!!/b&bo=IgazAgAAAAADB7c!&rf=viewer_4&t=5)


(Descriptive statistics of primary indicators)

------------

<div align=left>


**4.	Feature selection**

**1)	Data cleaning**

We check the missing data by calling the Missingno tool in Python. As shown in the following figure, the data is very complete and does not require preprocessing operations such as data filling.


![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/yCLjTthScCcjc0qcPSGYBgdSPT36VG3HpWmcbhuXi16AgrNuCoTJr25m0U.ty6kmlPQ3xzQesdAu.Ney6*m5igYvWX9eqq741WcjbEzfuEo!/b&bo=BAb2AgAAAAARF9Y!&rf=viewer_4&t=5)

(Data loss)

**2)	Data scaling**

The unit of the indicator Time is seconds, which causes the value of the indicator to be much larger than other indicators, which is not convenient for subsequent analysis with other indicators, so it is converted into units of hours, corresponding to the time of day. The magnitude of other indicators are relatively close and no additional processing is required.

**3)	Comparison**

In order to better distinguish the features between positive and negative samples, we will separate the positive and negative samples and compare them separately to better extract the features of the data.

**4)	Correlation between positive and negative sample variables**

In the incident of the credit card swiping, the correlation between some variables is more obvious. The variables V1, V2, V3, V4, V5, V6, V7, V9, V10, V11, V12, V14, V16, V17, V18, and V19 showed certain patterns in the credit card swiping samples.

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/yCLjTthScCcjc0qcPSGYBq3A5ZTXhJwGHUrySywy2PTfHO4Hy*142R0zmlOKZoqwAvQxIDKArL4e17Nvm*n0BG8VZmyDZ12M6Nyo5EX0rMs!/b&bo=gALgAQAAAAARF0M!&rf=viewer_4&t=5)

(Description of the correlation of variables of swiping sample)

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/yCLjTthScCcjc0qcPSGYBt0JAuHHLz98RwLw0L0bV6drdXjo3Wenn8YMWUSHjkfZkUv7gcr77QHAJ.womya*t1x1g5HcsTDmEzRe.qkioAs!/b&bo=gALgAQAAAAARF0M!&rf=viewer_4&t=5)

(Description of the correlation of variables of swiping sample)

**5)	Positive and negative sample transaction amount**

The amount of credit card swiping transaction appears to be scattered and small compared with the amount of credit card normal transaction.
![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/yCLjTthScCcjc0qcPSGYBg5c0T*RB8HxVxumFDIC*GbeT9FO5IMG4rTkVhSTtjcEcw5kxZANziypDyJl7aMKpah*DziDnH*p4CAIYKPP.S4!/b&bo=4QRoAQAAAAARF6w!&rf=viewer_4&t=5)
(Amount of transaction of positive and negative samples)


**6)	Positive and negative sample trading time distribution**

As the normal transaction time distribution diagram shows, between there is a high-frequency period of credit card consumption during 9 am and 11 pm every day.
As the time distribution of fraudulent transactions show, the highest number of credit card swiping transactions reached 43 at 11 am on the first day, followed by second number at 2 am, indicating that credit card thieves did not want to attract the attention of credit card owners and prefer to choose the time when the credit card owner sleeps or the frequency of consumption is high.

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/yCLjTthScCcjc0qcPSGYBpHjFeOfW8W*8S92X1tuU4CoxUe05sMBPTUVvCWDzfmMvPsEww.CBdV0ZhseIB8Mn1VmItEZKd42cEps5h8PNt8!/b&bo=DQX1AgAAAAARF98!&rf=viewer_4&t=5)

(Distribution of transaction time of credit card swiping sample)



**7)	Relationship between positive and negative sample transaction amount and transaction time**

It can be seen from the figure that in the sample of credit card swiping, the outliers occurred during the period when the customer used the credit card to spend at a lower frequency. At the same time, the maximum amount of credit card swiping is only 2,125.87 US dollars.

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/yCLjTthScCcjc0qcPSGYBmsuzUMfNPIv7QCzvaLWE7NAwCaYDOCUZZnbn3AO245qNpdrtODNThd1tsPhPlaRNvlzlyNvYvIhoyEIrIUJM9c!/b&bo=2wQdAgAAAAARF.A!&rf=viewer_4&t=5)

(Relation between amount and time of fraudulent transaction)


**8)	Distribution of different variables on positive and negative samples**

The following figure lists some of the distributions of different variables in credit card swiping and normal samples. We will choose variables that have obvious differences in the distribution of different credit card states, excluding variables that distinguish poorly between positive and negative samples, such as V8, V13 and V24, after processing, the feature variable was reduced from 31 to 18.

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/yCLjTthScCcjc0qcPSGYBnmdpSIQCHI3SMzT2uh4t5N5jY8r8e7eh47obUPB*6CI3NwnI3y*dq7rOeLDM974Flr*6qGdB0nSUNTIC8zPH2U!/b&bo=igMFAQAAAAARF60!&rf=viewer_4&t=5)

(Distribution histogram of characteristic variable v1)

**9)	Data dimensionality reduction**

Next, we perform principal component analysis on the data and reduce the 28-dimensional data to a 2-dimensional space. The data distribution after dimensionality reduction is shown in the figure below, where red zone represents samples of normal transactions and green zone represents samples of fraudulent transactions.

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/yCLjTthScCcjc0qcPSGYBg.gtOIc2isU6R8*YX*hWTRnHagPMKSpUXyvlat3rBs.71XGZrsq.*0.*maSwzgDIASiFvmqIpdnh0ys.CVQaXY!/b&bo=gALgAQAAAAARF0M!&rf=viewer_4&t=5)

(Initial visual effect of PCA)

Due to the imbalance of the data set, there are 280,000 pieces of non-swiping data, which is a large amount of data, and there are less than 500 pieces of credit card swiping transaction data. Therefore, we randomly select 1,000 non-swiping data, and then perform principal component analysis to improve the discrimination effect.

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/yCLjTthScCcjc0qcPSGYBn9sT2.eGnK0Me9oj3*OAEjXKfOOHgY.NBFwkK74C7U.gFnRXnIZVMhSNBHji*eJa2nTcq4p25NgRUIC7qeRMnA!/b&bo=gALgAQAAAAADF1E!&rf=viewer_4&t=5)

(Visual effect of PCA after subsampled)

Then we try to use a better nonlinear dimensionality reduction method T-SNE, the results show that the distinction between positive and negative samples is more obvious.

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/yCLjTthScCcjc0qcPSGYBuyJSgirV5pGX9*xEC90sUZtUGPXEZtiKaIbl7alpP.TKJYuM2QG9MiaNDhYK*qZdDbCr6nQzLQ0KPIR4mpy82U!/b&bo=DgNJAgAAAAARF2Y!&rf=viewer_4&t=5)

(Visual effect of TSNE)

Finally, through 3D display, we can a better visual effect.

------------



**5.	Rat Clip -prediction of credit card fraud based on SMOTE and logistic regressionLogistic Regression**

Logistic regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary).  Like all regression analyses, the logistic regression is a predictive analysis.  Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.
The sigmoid function, also called the sigmoidal curve (von Seggern 2007, p. 148) or logistic function, is the function

![](http://m.qpic.cn/psc?/V11zaUPV2EE2Gc/8YUQ4vKPKp.vxIKbDZcdtoE.mqkpDrnr8oVv0FrwTO7L59FzliSEA8BRnmbvbi72oZq4aVjGn5gbRrUlJRYdDw!!/mnull&bo=kQAxAAAAAAADB4I!&rf=photolist&t=5)

Function image of Sigmoid function is:


![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdtmPdxfHJAHUjcTegHGqLXmsYw6sOj2WUvky3K4a018zKKCwYwwADLsLyMV1b6VS4vQ!!/mnull&bo=5gFKAQAAAAARB5w!&rf=photolist&t=5)


(Sigmoid function)

A sigmoid function is a type of activation function, and more specifically defined as a squashing function. Squashing functions limit the output to a range between 0 and 1, making these functions useful in the prediction of probabilities.
Sigmoidal functions are frequently used in machine learning, specifically in the testing of artificial neural networks, as a way of understanding the output of a node or “neuron.” For example, a neural network may attempt to find a desired solution given a set of inputs. A sigmoidal function will determine the output and that output will be used as the input for the following node. This process will repeat until the solution to the original problem is found.
Based on the sigmoid function, the assumed function form of the logic function is as follows:

![](http://m.qpic.cn/psc?/V11zaUPV2EE2Gc/8YUQ4vKPKp.vxIKbDZcdtoZoM9ogIVn2lonJNYKLbcEwvPNGxgEGzuVDu.kITSdvf9N0Cgc1dpw5eFceFSZIhw!!/mnull&bo=IAE8AAAAAAADBz8!&rf=photolist&t=5)

Where x is the input variable and Θ is the parameter vector. This function gives a probability value, that is, given x and Θ, the probability of y = 1. But logistic regression gives classification results, not just probability values, so here a decision function is needed.

![](http://m.qpic.cn/psc?/V11zaUPV2EE2Gc/8YUQ4vKPKp.vxIKbDZcdtsZhRSivKx*X659cXnB6jYZIyyoWzf2MpAC0Brkpcr21IMmlMtHgtMKDsQdELS7IcA!!/mnull&bo=1AA.AAAAAAADB8g!&rf=photolist&t=5)

As a general approach, 0.5 is used as an example of the threshold here, but in actual application, different thresholds will be selected depending on the situation.
Well, we have all the functions we want to use. The next step is to find the parameter Θ based on the given training set. To find the parameter  Θ, the first thing is to define the cost function, which is the objective function.
Assuming:

![](http://m.qpic.cn/psc?/V11zaUPV2EE2Gc/8YUQ4vKPKp.vxIKbDZcdtvqLUcgdf2RVmuO6b4fP.fMcVPL7NGFG3KmxZGJssrdY1094Soc5eiCIHtZON7mCGA!!/mnull&bo=wgBCAAAAAAADB6I!&rf=photolist&t=5)

Transfer the above two formulas in general form:

![](http://m.qpic.cn/psc?/V11zaUPV2EE2Gc/8YUQ4vKPKp.vxIKbDZcdtvLwmwm6Rg.XpgfDSE7dcbj77CJVvwrP*6TD4xcLVBph.FqbTw2j5UzcNdKCG1i6CQ!!/mnull&bo=AQEtAAAAAAADBw8!&rf=photolist&t=5)

Next, we will use maximum likelihood estimation to estimate the parameters based on the given training set.

![](http://m.qpic.cn/psc?/V11zaUPV2EE2Gc/8YUQ4vKPKp.vxIKbDZcdtkyW0WymMiAjEIWz09UYfGSd1M0yVjMgP**BkLA7gInD3YRH26MzkJuxsI8xyB.DTQ!!/mnull&bo=*AE.AAAAAAADB.E!&rf=photolist&t=5)

To simplify the operation, we take a logarithm on both sides of the above equation.

![](http://m.qpic.cn/psc?/V11zaUPV2EE2Gc/8YUQ4vKPKp.vxIKbDZcdtsB*cvo4tifaw5LlvXH1cBxlMpBmDBFacY1wc1mf6goFKeUkftmrAKcyEeO8lMq*lQ!!/mnull&bo=zAFCAAAAAAADB60!&rf=photolist&t=5)

In the actual solution, the cost function should be minimized, so the cost function is defined as:

![](http://m.qpic.cn/psc?/V11zaUPV2EE2Gc/8YUQ4vKPKp.vxIKbDZcdtvrXhRT6KeIiRxc1RwieBHKOp3KNEAQ6*8N3RzEvHjFKFfJKhQkvI0vaQKgKPxeq4A!!/mnull&bo=5AFMAAAAAAADB4s!&rf=photolist&t=5)

The next step is to find the extreme value. The methods commonly used in logistic regression learning are the gradient descent method and the Newton method, which are not outlined here.

(1)	Model evaluation
The evaluation of machine learning model effects mainly includes the following indicators: Accuracy, Precision, and Recall are common basic indicators.
To understand the meaning of these indicators, you first need to understand two samples: (1) Positive sample: that is, a sample that belongs to a certain category (generally the one requested). In this case, it was a credit card that was stolen; (2) Negative samples: samples that do not fall into this category. In this case, it is a credit card that has not been stolen.
So we can get the following table:

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdtkF4iS5jktYG*6M0K*LfjccMZUWvWlNsyk2DTyNWCzOWE3g10y*B5.6CSd31y1*HbA!!/mnull&bo=YgMbAQAAAAARB0s!&rf=photolist&t=5)


True Positives, TP：The prediction is a positive sample, and the actual number of features is also a positive sample;
False Positives,FP：The prediction is a positive sample, but the actual number of features is a negative sample;
True Negatives,TN：The prediction is a negative sample, and the actual number of features is also a negative sample;
False Negatives,FN：The prediction is a negative sample, but the actual number of features is a positive sample;
According to the above definitions of four types of situations, relevant model evaluation indicators can be calculated:

Accuracy：(TP + TN) / ( ALL )

Precision：TP / (TP + FP)

Recall：TP / (TP + FN)

The calculation and meaning of Precision and Recall can also be intuitively understood through the following picture.

(Schematic diagram of model evaluation index calculation)
![](http://m.qpic.cn/psc?/V11zaUPV2EE2Gc/8YUQ4vKPKp.vxIKbDZcdtq.EZugXycvz98LG7yOETXVOh8YwVfWUZ27uPiQyXqZXBycUqbcvPnRStwEVZjvYfg!!/mnull&bo=hAFMAQAAAAADB.o!&rf=photolist&t=5)

Different indicators have different emphases, and Accuracy is the most commonly used indicator, which can generally measure the performance of a prediction. Precision is concerned about the correct proportion of positive samples predicted by the model, that is, the accuracy of the model prediction. Recall pays attention to the proportion of positive samples (TP) predicted by the model in all positive samples, that is, whether the model can find all positive samples and find them incomplete.
The focus is different in different situations. In the scenario of recognizing spam, it may be biased towards Precision, because we don’t want many normal emails to be killed by mistake, which will cause serious problems. In the field of financial risk control, most of them prefer Recall. We hope that the system can screen out all risky behaviors or users, and then hand it over to human identification to omit one that may cause disastrous consequences.Two other commonly used indicators to evaluate the pros and cons of a Binary Classifier are ROC (Receiver Operating Characteristic) curve and AUC (Area Under Curve).

The abscissa of the ROC curve is False Positive Rate (FPR), and the ordinate is True Positive Rate (TPR). The definitions of FPR and TPR are as follows:
TPR (True Postive Rate): TP/(TP+FN), represents the ratio of actual positive instances to all positive instances in the positive class predicted by the classifier.
FPR(False Postive Rate): FP/(FP+TN)，represents the proportion of actual negative instances to all negative instances in the positive class predicted by the classifier.
For the logistic regression classifier used in this section, which gives the probability of being a positive class for each instance, then by setting a threshold such as 0.6, the probability is greater than or equal to 0.6 for the positive class, and less than 0.6 for the negative class. Correspondingly, a set of (FPR, TPR) can be calculated and corresponding coordinate points can be obtained in the plane. As the threshold value gradually decreases, more and more instances are classified as positive classes, but these positive classes are also doped with true negative instances, that is, TPR and FPR will increase at the same time. When the threshold is the largest, the corresponding coordinate point is (0,0), and when the threshold is the smallest, the corresponding coordinate point is (1,1).
As shown in the following figure, the solid line in the figure is the ROC curve, and each point on the line corresponds to a threshold.

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdtkaBrvN5ZD3C6o8p0WwtXyQOsl097nNKrLyDZllnU*d*v7WkC8N4VYsJ7QaAaZyMog!!/mnull&bo=awFlAQAAAAARBz4!&rf=photolist&t=5)

(ROC curve)


The ideal goal of the model is TPR = 1, FPR = 0, which is the (0,1) point in the figure, so the closer the ROC curve is to the (0,1) point, the better it deviates from the 45-degree diagonal.
AUC (Area under Curve) is the area under the ROC curve between 0.1 and 1. In terms of specific meaning, the AUC value is a probability value. When a positive sample and a negative sample are randomly selected, the probability that the current classification algorithm ranks this positive sample in front of the negative sample according to the calculated Score value is the AUC value. The current classification algorithm is more likely to rank positive samples in front of negative samples, so that it can better classify.

(2)	Logistic regression results and evaluation

We use 70% of the samples as the training set and 30% of the samples as the test set, use the training set to train the logistic regression, and pass the test set to test the model's performance indicators as follows:

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdto4qjbLXX6MTuOWat4*lbRIka.sIrc7F1Xe1j9dy3S.r6*DxkxMXHfzW1kTMalvQIQ!!/mnull&bo=YgMpAQAAAAARB3k!&rf=photolist&t=5)

The result of using logistic regression alone is not particularly good. Although Accuracy is relatively high, other indicators are relatively low, and the recall rate (Recall) is only 0.61, a large number of 1s are misjudged as 0, and it is predicted to shift to a large proportion of 0 for classification. This is mainly because the positive and negative samples in this article are extremely unbalanced, and it is precisely because of the existence of a large number of negative samples that the Accuracy indicator is artificially high, but the actual effect of the model is not good.
In the following part, we will focus on solving the problem of sample imbalance.

(3)	SMOTE oversampling + logistic regression results and evaluation

The default threshold for most models is the median output value. For example, the output range of logistic regression is [0,1]. When the output of a sample is greater than 0.5, it will be classified as a positive example, and the reverse is a negative example. When the categories of data are unbalanced, the use of the default threshold may cause all outputs to be counter-examples, producing falsely high accuracy and causing classification failure. The data set used in this article contains data submitted by European cardholders using credit cards in September 2013. This data set shows transactions that occurred within two days, of which 492 of 284,807 transactions were stolen. The data set is very unbalanced, with positive categories (stolen and brushed) accounting for 0.172% of all transactions.
In practical applications, there are four main methods to solve the problem of sample imbalance:

1)	Adjust the threshold to make the model more sensitive to fewer categories.
2)	Choose an appropriate evaluation standard, such as ROC, instead of accuracy.
3)	Undersampling: Undersampling discards a lot of data, but it also has the problem of overfitting, just like oversampling.
4)	Oversampling: The oversampling method is to repeat the proportional data. In fact, no more data is introduced for the model. Too much emphasis on the proportional data will amplify the impact of the proportional noise on the model.
5)	SMOTE（Synthetic Minority Oversampling Technique）: It is an improved scheme based on the random oversampling algorithm. The basic idea of the SMOTE algorithm is to analyze minority samples, and manually synthesize new samples based on the minority samples (equivalent to the large sample data) and add them to the data set Balanced data set.
Compared to simple oversampling, SMOTE reduces the risk of overfitting. SMOTE will generate new positive examples in the local area through K nearest neighbors. This method can be understood as a kind of centralized learning, which reduces the variance. And the method is more resistant to noise. Through SMOTE oversampling, 50% of positive and negative samples are obtained. Use the training set to train the Logistic model. By testing the model effect through the test set, the following confusion matrix is obtained.

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdtpjzWkiCyaLwFldS2JsbjmLH*wRVkWupwq6j9AbURSU6w1CA3k8KyMN9IiJ9etnmBA!!/mnull&bo=YgPtAAAAAAARB7w!&rf=photolist&t=5)

The effect index of the model can be calculated through the confusion matrix:

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdtnu9aZBXKK9pOBGd.pWaN6FxFKZTYxUqLgewRVW.BjfpovOv12uRrwCg51zqwjjCKA!!/mnull&bo=YgNbAQAAAAARBws!&rf=photolist&t=5)

After solving the problem of sample imbalance through SMOTE, ROC-AUC indicators, Recall (recall rate) and Precision (precision rate) have been greatly improved, and the model performs very well.
As mentioned earlier, thresholds also have an important effect on the results, so let's take a look at how much different thresholds will affect the results. The horizontal axis in the figure below is the Recall recall rate, and the vertical axis is the Precision recall rate. Different colored lines correspond to different thresholds. It can be seen that as the threshold decreases, the recall rate gradually increases, the precision rate gradually decreases, and the model's misjudgment also increases.

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdtnxcm1INVA0liWIkZBlR98.eXa216W5kgbNzcdEeVgRoWP7A*RZHIbl6cv4j1*HAKg!!/mnull&bo=sAS8AgAAAAARBzo!&rf=photolist&t=5)

(Precision-Recall Curve)

This phenomenon is very significant. In actual business, if the threshold is set high, the attack on theft may be too small. If the setting is too low, it may indeed be possible to find more cardholders whose credit cards have been stolen, but as the number of misjudgments increases, not only will the workload of the post-loan team increase, but it will also reduce the misjudgment of credit card theft Brush customer's consumption experience, which leads to a decrease in customer satisfaction. In reality, the company's choice of threshold should be a point where the marginal profit and marginal cost of the business can be balanced.


------------


**6.	Rat Poison-AutoEncoder combined with Logistic regression**

(1)	Introduction to AutoEncoder

AutoEncoder automatically encodes the input training set of the neural network as an unlabeled data set, so it is an unsupervised learning algorithm. It was proposed by the scientist Rumelhart in 1986 and is an important node in the development of neural networks. It is mainly used for complex high-level Dimensional data analysis. Autoencoder uses BP back-propagation algorithm to back-test to make the target output equal to the input value.
Autoencoder is a multi-layer neural network with the same meaning as the input layer and the output layer. The input layer and the output layer have the same number of nodes. The number of neurons in the two layers is exactly the same, but the number of neurons in the hidden layer must be less than the output. Floor. This model is often used for dimensionality reduction or feature learning, the structure is as follows:

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdtp6yWxZqy8kxyRNqqzF7phlvgMf1YKpuvcEt3U**4IUeovkJNY9euzKSUdpqjkpd4g!!/mnull&bo=agGOAQAAAAARB9Q!&rf=photolist&t=5)

(A simple self-encoder structure)


We can understand the self-encoder through an example: the chess master observes the board for 5 seconds and can remember the positions of all the pieces, which is impossible for ordinary people. However, the placement of the pieces must be the actual game (that is, there are rules for the pieces, just like the second set of numbers), and the randomly placed chessboards can't work (like the first set of numbers). Chess masters are not superior to ordinary people in memory, but are experienced and very good at recognizing chess patterns, so as to efficiently remember chess games.
Similar to the player's memory model, an autoencoder receives input, converts it into an efficient internal representation, and then outputs the analog of the input data. Self-encoders usually consist of two parts: Encoder (also known as recognition network) converts the input into an internal representation, and Decoder (also known as generation network) converts the internal representation into an output.

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdtriQPd4wo9ZNtHVDK*4DYegm0EVP0Gg7OcgdmEhXQZQoKM*1SpaWkyAkwQVDe0jHxA!!/mnull&bo=DQPaAQAAAAARB.U!&rf=photolist&t=5)

(Chess master's memory mode (left) and a simple self-encoder)


As shown in the figure above, the structure of the autoencoder is similar to the multilayer perceptron, except that the number of input neurons and output neurons are equal. In the example above, the autoencoder has only one hidden layer (Encoder) containing two neurons, and the output (Decoder) containing three neurons. The output is trying to reconstruct the input, and the loss function is the reconstruction loss. Since the dimension of the internal representation (that is, the output of the hidden layer) is smaller than the input data (2D replaces the original 3D), this is called an incomplete self-encoder.

(2)	The core concept of AutoEncoder algorithm

AutoEncoder is a neural network that is trained to copy the input to the output as much as possible. The neural network strives to make the output consistent with the input content. What it learns is an "identity function" where the input is equal to the output.
As shown in the figure below, AutoEncoder is mainly composed of three parts: Input Layer, Hidden Layer and Output Layer. The network structure is relatively simple. Among them, there is a hidden layer h inside the autoencoder, which can generate codes to represent the input. The network can be viewed as consisting of two parts: an encoder h = f (x) and a decoder for generating reconstruction r = g (h). Finally, make x approximately equal to g (f (x)). The network can be designed so that x = g (f (x)).

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdtuJbz7cfJ*ILmWHlEclKpPOeQKzgsWGy.yU4bqY8GneYxsQcxLV7sDjKxwFirYxPRw!!/mnull&bo=8ALqAAAAAAARByg!&rf=photolist&t=5)

(Self-encoder structure)


The core concept of the automatic encoder algorithm is: by establishing the Encode and Decode processes (actually encoder and decoder) in the hidden layer, the input and output are getting closer and closer, and the input data is encoded and decoded by the hidden layer, and the output a result consistent with the input data.
Although this concept is theoretically feasible, the actual operation will not be so ideal. In order to allow the automatic encoder to learn the useful features of the data, it cannot be designed as an encoder with 100% copy input (theoretically, it cannot be done because the number of layers is different), so some constraints must be specified in the hidden layer. The Autoencoder can only be copied approximately.
The common constraints of this model are as follows:

a)The dimension of the output layer is much larger than the hidden layer;

b)Use the output function to recover the input data by minimizing the error function of the input and output;

c)Because this is an unsupervised learning model, the data is unlabeled, so the source of the error is the difference between the reconstruction and the input.

The Autoencoder can be regarded as compressing the data, from the original "n-dimensional" to "m-dimensional", where m is the number of hidden layer neurons. Then, when necessary, recover the data with the least loss.

(3)	AutoEncoder Algorithm Process


a) The Autoencoder automatic encoding network is to restore the compressed data, that is to learn a set of h_(W,b) (x)≈x, which is the parameter to be learned by the algorithm.

b) Restoring the data should make the loss as small as possible. The objective function is specified as:

![](http://m.qpic.cn/psc?/V11zaUPV2EE2Gc/8YUQ4vKPKp.vxIKbDZcdtrNnElSWqvOun2FgC7rfwrownIf7Po*x40U*I*cRSsHOU.aQVX3XxraMhGWN0a98ZA!!/mnull&bo=lgA6AAAAAAADB44!&rf=photolist&t=5)


In order to determine the initial data of the weights in the medium weight matrix, the autoencoder needs a BP neural network to do a pre-training to minimize the error between the input and output values. At the beginning, a process of compressing and extracting features will be introduced. The compression of the input data (limiting the number of hidden layer neurons) is re-encoded, and the low-dimensional vector data is used to replace the high-dimensional input data, so that the compressed low-dimensional vector can retain the input data To help restore the original data later. Re-encode the input data with a weight matrix, then enter the activation function, and then use WT to decode, so that h (x) ≈x. The Encode and Decode processes in the hidden layer use the same parameter matrix. Through these constraints, the number of parameters can be reduced and the model complexity can be reduced.
The prerequisite for data compression is that there is some redundant information in the data, but we know that there is some redundant information in the data such as sound and images in the display. The self-encoding network continuously learns and Optimized to identify redundancy and remove redundancy.

(4)	Considerations for using AutoEncoder

In fact, the previous method of oversampling combined with Logistic regression has achieved good output in our experiments. The algorithm has high recognition accuracy for stolen brushes, but in order to optimize our model and improve the performance of the model, it is more accurate. To identify the stolen data and overcome the extremely unbalanced nature of the data sample itself, we used the AutoEncoder and Logistic regression to test again.
According to the properties we learned about the Autoencoder, this model is actually a neural network to learn the correlation representation of the input data. Considering that this model has the following advantages, we think that it may be a model for identifying stolen brushes. It has a certain improvement effect: 1) can extract the core features of the input data through more efficient and low-loss dimensionality reduction; 2) has a certain ability to filter noise, can better grasp the core features of the data; 3) can be better The interpretation of sparse attributes, most of the real scenes meet this constraint; 4) The algorithm is scalable and has relatively stable properties; 5) The algorithm has strong expressive ability and is more suitable for data visualization processing, making the model results more intuitive.
At the same time, according to previous experiments, we have noticed that Autoencoder also has some shortcomings: 1) The theoretical feasibility is greater than the actual usability, of course, this is highly related to the properties of the data itself and the scene; 2) The compression ability It is suitable for samples similar to training samples; 3) The learning ability of the encoder and decoder of the hidden layer cannot be too high, otherwise the function expression of the model is poor.

(5)	AutoEncoder + Logistic regression experiment process and result analysis

We used the data from the previous oversampling experiment to reduce the dimension to 3 dimensions using TSNE and visualize it. The results show that the distinction between positive and negative samples is obvious.

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdtuX5wEHPiPkDzt*FuQZ1tq.nazFqWhycdPnTuh53MjL5JlWLv9o*kh3LklThU3gWRA!!/mnull&bo=DANnAQAAAAARB1k!&rf=photolist&t=5)

(TSNE dimension reduction results)

The TSNE method used here is currently the most effective data dimensionality reduction and visualization method. When we want to classify high-dimensional data, it is not clear whether this data set has good separability (that is, between similar The interval is small, the interval between different types is large), you can use TSNE to project into a 2D or 3D space to observe. If it is separable in the low-dimensional space, the data is separable; if it is not separable in the high-dimensional space, it may be that the data is inseparable, or it may be simply because it cannot be projected into the low-dimensional space.
Next we build AutoEncoder and visualize the expressed features. The data characteristics before AutoEncoder expression are as follows:


![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdtnvnUrhoMtW7ovcafFZl0hJ6iXpPRAzJygutuLc0*k6u9biM3X8MKH5P17xyYe6scQ!!/mnull&bo=IAO1AQAAAAARB6c!&rf=photolist&t=5)

(Feature distribution without self-encoder)

The data characteristics after AutoEncoder expression are as follows:
![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdtkWWHfwAZH.4Yv6IxKjvCTdWHS9RhKzeWoMqrX.5PNnJfxgtRQ1OfOgdZ4xAatc5ZQ!!/mnull&bo=DANnAQAAAAARB1k!&rf=photolist&t=5)

(Feature distribution via autoencoder)

In comparison, the distinction between positive and negative samples is slightly improved.
Entering the data into Logistic regression, we get Accuracy score of 0.99391407, Roc_auc score of 0.87153955, Recall score of 0.74324324, Precision score of 0.8870967, the remaining parameters are as follows:

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdtip1SQvfVQmtzixsMJxM66zB7bXhL9Eqbskh9vO6buYEWTr3OTsgf67WnFvzsA3BDQ!!/mnull&bo=hgNPAQAAAAARB*s!&rf=photolist&t=5)

(Model performance indicators)

And get the following picture:

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdtuO.9Gsi2xx4jAcdvS2NVnKA9xcEcamNDSpmRNsxqxgXQWxSoIYjDZ7hXCjNsopu9A!!/mnull&bo=ngKRAQAAAAARBzw!&rf=photolist&t=5)

(The result of auto-encoder combined with Logistic regression)

Through this image comparison, it is found that the effect obtained by AutoEncoder + Logistic regression is significantly better than pure Logistic regression, but the effect is not as good as oversampling combined with Logistic regression method (because the area under the curve becomes significantly smaller).


**7. Ultra weapon——Two-pronged**

In order to better identify fraud, we use a two-pronged weapon. We combined SMOTE oversampling with AutoEncoder and Logistic Regression. The results are shown in the following table: 

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdti0eKy4iGs0wW0NKoRw.JAQsfoYS9v.SGa3w*EytJxeAKaMd61JgkSk8GXEHIO2R4w!!/mnull&bo=egN*AQAAAAARBzc!&rf=photolist&t=5)

(Performance score of our ultra weapon)

From the table, we can see that when we combined the three methods, we got better results in terms of Precision, Recall and F1-score. Under the test sample set containing 30% data, all indicators exceeded 0.99. 

![](http://m.qpic.cn/psc?/V11zaUPV24qAQK/8YUQ4vKPKp.vxIKbDZcdtgzj8mLqqb9KB*xROfALov0Brt8R8gSJyDfpe*o1kUm*RvA28BmH.VAhSwf9YjrVzg!!/mnull&bo=AgWlAgAAAAARB5A!&rf=photolist&t=5)

(Precision-Recall Curve of the ultra weapon)

From the figure, we can see that when the threshold changes from 0.1 to 0.9, the AUC is always above 0.98. It can be seen that our results are very little affected by the threshold and have strong robustness. 

**8. Conclusion and further improvement**

(1) Conclusion

In this project, we build a Machine Learning model to detect credit card fraud. We first used SMOTE oversampling to deal with the data imbalance. Then we used the unsupervised learning method AutoEncoder to extract important features of the data set to improve the robustness of the model. Finally, the extracted features were input into the Logistic Regression model for training to obtain our final credit card fraud detection model. Finally, the model we obtained has a high AUC and f1-score in both the training set and the test set, and is robust to different thresholds.

(2) Further improvement

In the process of model building, we manually process the data features. But in practice, the data may contain millions of features, so automated feature engineering is important.

In this project, we did not obtain enough information dimensions. However, in practice, more types of data, such as credit investigation data of the PBC (People's Bank of China) and user behavior data in Internet companies, can be used as important data for credit card fraud detection model.

More advanced algorithms such as relational network, streaming active learning strategy and deep learning can be applied to credit card fraud detection.

------------



**References**


[1]HSN Consultants, Inc.: The Nilson report (consulted on 2018-10-23) (2017).
https://nilsonreport.com/upload/content promo/The Nilson Report Issue 1118.pdf

[2]Carcillo, F., Dal Pozzolo, A., Le Borgne, Y.A., Caelen, O., Mazzer, Y., Bontempi, 	G.: Scar_: a scalable framework for streaming credit card fraud detection with 	spark. Information fusion 41, 182{194 (2018)

[3]Pozzolo, A.D., Boracchi, G., Caelen, O., Alippi, C., Bontempi, G.: Credit card 	fraud detection: A realistic modeling and a novel learning strategy. IEEE 	Transactions on Neural Networks and Learning Systems PP(99), 1{14 (2018). D	OI 10.1109/TNNLS.2017.2736643

[4] A.C. Bahnsen, D. Aouada, B. Ottersten, Example-dependent cost-sensitive 	decision trees, Expert Syst. Appl. 42 (19) (2015) 6609–6619.

[5] F. Carcillo, A. Dal Pozzolo, Y.A. Le Borgne, O. Caelen, Y. Mazzer, G. Bontempi, 	Scarff: a scalable framework for streaming credit card fraud detection with spark, 	Inf. Fusion 41 (2018) 182–194.

[6] A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, G. Bontempi, Credit card fraud 	detection: a realistic modeling and a novel learning strategy, IEEE Trans. Neural 	Netw. Learn. Syst. 29 (2017) 3784–3797.

[7] N. Sethi, A. Gera, A revived survey of various credit card fraud detection 	techniques, Int. J. Comput. Sci. Mobile Comput. 3 (4) (2014) 780–791.

[8] P.R. Shimpi, V. Kadroli, Survey on credit card fraud detection techniques, Int. J. 	Eng. Comput. Sci. 4 (11) (2015) 15010–15015.

[9]Lebichot B., Le Borgne YA., He-Guelton L., Oblé F., Bontempi G. (2020) 	Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection. 	In: Oneto L., Navarin N., Sperduti A., Anguita D. (eds) Recent Advances in Big 	Data and Deep Learning. INNSBDDL 2019. Proceedings of the International 	Neural Networks Society, vol 1. Springer, Cham

