

# Google Store's Customers Revenue Analytics

## Introduction
This project will investigate data integration and model building in IPython. It is based on the ![Google Analytics Customer Revenue Prediction](https://www.kaggle.com/c/ga-customer-revenue-prediction) Kaggle challenge, revolving around predicting how much the GStore customer will spend based on observed past data of customer's spending.


## Preprocessing data
Including following steps:
- Omitting columns that have constant values across all samples
- Filling NaN in "totals.transactionRevenue" column
- Processing timestamp data from POSIX format to date-time format. And then compute useful features from them.
- Turning category features to numbers by factorizing them. (this step will be introduced more in details below)

## Exploring data
### Generating heatmap of the session "Visiting Time" (in terms of month and hour) and "Transaction Revenue":
We find that there are a great correlation in the visit time and the revenue in terms of hour. Most of the revenue are created in the afternoon to midnight.This is quite easy to understand, since nobody visit GStore in the middle of the night (1am - 5am), and also, in the morning since they have to go to work.
In terms of month, we find there are a great demand in December, which must be because of Christmas. Indeed, at this time of the year, people still buy from GStore up to 2-3am in the morning.

<p align="center">
  <img  src="../master/illustrations/heatmap.png">
</p>

### Data clustering under variety of categories:
We cluster data under many category (browser, device category, OS system,...) to see how these information affect the buying decision of customers.
The below figure illustrates data clustering by Page Views.
![Pic2](../master/illustrations/behaviors.png)

## Extending dataset
We look for outside dataset to see if there is any extra insight into our problem. For extending dataset, we use two dataset that we think might affect the customer's behaviors in GStore, they are:
### The stock market performance of GStore:
A company's stock price reflects its performance as well as opinions of customers/investors about it and therefore, might reflect their intention to buy products from the company. For example, if the new released product is a good product, having many good review from critics and journalists, then the price of the stock will rise, and customers also would like to buy new products, too. In contrast, if there is something wrong with the new product, the stock price will drop and customers would not like to buy it. For example, when the Samsung Galaxy Note 7 is announced to be very dangerous, the Samsung stock price decrease drastically, the customers also postponed their decision to buy that product.
 
### The price of USD overtime:
For countries in Asia, such as Vietnam (my country), we do care allot about this exhange rates, since it greatly affects the price of the product. Indeed, many of the products on GStore are sold in USD, and we then multiply them with the exchange rate (USD->VND) to have the amount of VND we have to pay. Many times, if the USD is strong, the exchange rate will be high, and therefore we have to pay more. Many times we consider buying or not because of this. 


## Building model for revenue prediction
Building model for our regressor. The technique used is LightGBM. With tuned parameters. 
The features chosen are: 

*['visitNumber','totals.pageviews','visitNumber','totals.hits', 'visit_hour']*

The results are shown in the iPython Notebook report file.

## Permutation test 
We then conducted permutation test to investigate the contribution of each feature to the model's accuracy. We conduct this experiment to analyze the features before choooing out the above-mentioned set of features for the optimized model.

<p align="center">
  <img  src="../master/illustrations/All_permutation_test.jpg">
</p>








