# PHBS_MLF_2019_Project  
## Members 
[Yuchuan Xu (1801212958)](https://github.com/YuchuanXu-1801212958/YuchuanXu-1801212958-PHBS_MLF_2019)  
[Yang Cao (1801212825)](https://github.com/crobiny/MLF)  
[Alimujiang (1801212778)](https://github.com/Alimurestart/PHBS_MLF_2019)  
[Bowen Chen (1801212827)](https://github.com/npuchenbowen/PHBS_MLF_2019) 


## Goal
Because credit card theft will cause economic losses to the holder and affect the work efficiency of financial institutions, it is necessary to design an automatic fraud detection system for high-precision detection of fraudulent transactions. This project hopes to use a set of credit card consumption data sets to train a classifier to distinguish whether a user's credit card usage record is fraudulent information, so that credit card companies can effectively identify fraudulent credit card transactions.


## Description of dataset
The data set used in this project is the credit card transaction data of a group of European cardholders in September 2013. This data set contains 284,807 transactions that occurred in two days, of which 492 were stolen transactions and the rest were normal transactions. The data set is very unbalanced, with stolen transactions accounting for only 0.172% of all transactions. The data set contains 31 descriptive indicators.The index Class is the response variable. If the transaction belongs to the theft transaction, the value is 1; if the transaction belongs to the normal transaction, the value is 0. Indicator Time describes the Time at which a transaction occurs. Specifically, the number of seconds between the time of each transaction and the time of the first transaction; The indicator Amount is the Amount of the transaction; Indicators v1 and v2-v28 were obtained by PCA. Due to security and confidentiality reasons, the meaning of indicators themselves was not given.
![](data/description.png)
