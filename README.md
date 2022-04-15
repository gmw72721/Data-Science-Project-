# Final_Project

## File Structure

1. Scraping_Script.ipynb - Scrapes the test data for the 2021 Olympic Marathon results online. Also compiles and cleans the training data. 
2. EDA.ipynb - Runs a simple analysis and visualization on the data collected.
3. model_tuning.ipynb - Prepared data and ran through hyperparameter tuning. 
4. Final_model.ipynb - Predicted on test set and ran analyssis on results. 


## Introduction
Running as a sport is classically under-analyzed. Unlike baseball, football, and basketball, running doesn't have a high demand for analytics. This is largely due to the fact that it is generally not as well funded as the more viewer-friendly sports. As such, the data available in larger sports, like movement, relative position, etc., is not available for running because data collection is expensive. Perhaps the most criminally understudied event in athletics is the marathon, as it has a high rate of participation with an almost absent viewership. However, as the longest Olympic event, there are likely several interesting aspects that haven't been analized as of yet. We therefore propose using a generalized boosting tree ensemble (GBM) to model large amounts of sparse men's marathon data. We will attempt to augment the sparsity of the matrix by adding new columns. During this analysis we hope to model the data from the top 1000 men's marathon performances with data found online. Additionally, we will use the model fit to that data to predict the 2021 Olympic men's marathon. Finally, we will use partial effect plots to find how diffferent features might be correlated with marathon time.
## Data
### Data Through 2016
The marathon training data was found through links this article https://towardsdatascience.com/analysing-elite-running-performance-with-historical-data-b9c6bdd9c5d8 in the reference section (marked [1] and [3]). These datasets were scraped from sites that record running results and were wrangled by Kaggle users. The first dataset was scraped from sports-reference.com and contains the height and weight of athletes, as well as the years they competed in the Olympic games. The second was scraped from alltime-athletics.com and contains the top 1000 times recorded for various Olympic distances, including the athletes' names and country of origin. This should not be confused with the top 1000 times of Olympic running events, which is a mistake that results in bad prediction, as will be discussed later. The final dataset we used can be found at https://www.kaggle.com/tunguz/country-regional-and-world-gdp. It contains the gross domestic product (GDP) of most countries, as well as the associated country code. It should be noted that we only used the GDP from 2016.
### 2021 Olympic Results
In addition to these datasets, we scraped data from olympedia.org, since the results moved from sports-reference.com to olympedia.org. This dataset contains similar columns to those contained in the references in the towardsdatascience.com article, but contained data from the men's Olympic marathon from 2021 instead of the top 1000 men's marathon performances. This dataset is to be used as the test dataset.
## Data Wrangling
The main points of data wrangling were joining datasets together and adding useful columns not previously contained in the datasets. To begin, we joined the top 1000 men's marathon performances with the Olympic athlete attribute data on the athlete name. We then joined the resulting dataset with the gdp dataset on country code. After that, we added the total number of previous marathon performances, since that could indicate how well a given athlete has competed in the past. Additionally, if an athlete had previously competed in a marathon in our dataset, we kept that previous performance time as a column. It should be noted that any missing values in our final dataset was imputed with a -1 because tree ensembles (such as GBM) account for such differences and can potentially use them for better prediction than if we had simply imputed the mean. Additionally, all marathon times are measured in seconds, all heights are in centimeters, and all weights are in kilograms.

## EDA

Here are some aspects of the EDA that were considered when analyzing the marathon data set. 

|       |   Results |       Year |       Age |   Previous marathons |           GDP |    Weight |   Height |   Last Time |
|:------|----------:|-----------:|----------:|---------------------:|--------------:|----------:|---------:|------------:|
| count | 1001      | 1001       | 1001      |           1001       | 926           | 129       |  129     |    585      |
| mean  | 7629.44   | 2009.34    |   39.5614 |              4.17283 |   3.69753e+11 |  59.4419  |  173.093 |   7615.59   |
| std   |   64.5683 |    6.42968 |    7.9716 |              3.0866  |   1.55435e+12 |   4.28461 |    6.725 |     70.1434 |
| min   | 7377      | 1981       |   25      |              1       |   3.4099e+08  |  49       |  155     |   7377      |
| 25%   | 7595      | 2007       |   34      |              2       |   3.58952e+10 |  56       |  167     |   7576      |
| 50%   | 7648      | 2011       |   38      |              3       |   5.04128e+10 |  60       |  174     |   7631      |
| 75%   | 7680      | 2014       |   45      |              6       |   6.37675e+10 |  62       |  177     |   7673      |
| max   | 7705      | 2017       |   74      |             14       |   1.61553e+13 |  70       |  191     |   7705      |

The pairplot of our data set. We can see some correlations in the data such as age and year the individual ran the marathon. Also, note the extreme skewness in the GDP. 

![image](https://user-images.githubusercontent.com/55562380/145143261-0fe73f6f-e210-4b8f-9751-159a1424e22e.png)

Given the relative simplicity of our data set, these seemed to be the most important aspects of the EDA. Other plots can be found in the EDA.ipynb file for more information. 

## Model and Tuning

A gradient boosted regressor was trained on the top 1000 men's marathon results data set. We chose to use this model for a variety of reasons as outlined below:


1. Gradient boosted models are flexible for working with many types of data sets without the excessive computational requirements of a deep learning model
2. Many gradient boosting models can account for nan values in the dataset without much imputation
3. Gradient boosted models have shown to be strong at predicting on data, but at the cost of less interpretability. Our project focuses on prediction. 

However, gradient boosting regressors tend to overfit if cross validation is not performed. Therefore, repeated K-fold cross-validation was performed to prevent overfitting by maximizing out-of-sample performance.

On the hyperparameter tuning step, we chose five key parameters to tune our model:

	- learning_rate: Changes the contribution of each tree in the model
	- n_estimators: Considers the number of boosting stages to perform on the model
	- max_depth: Limits the number of nodes in trees
	- min_samples_split: Minimum number of samples needed to split a node. 
	- min_samples_leaf: Minimum number samples required to be a leaf node. 

Each of these hyperparameters were indicated to have the greatest effect on modifying the model for optimal tuning on the data set. 

The model was tuned using a repeated K-fold cross-validation using random search on the hyperparameters. This allowed for 15000 total fits on the model to find the optimal hyperparameters that minimized root mean squared error (RMSE). Minimizing RMSE is a standard way to optimize models because it strikes a balance between optimal bias and variance.

The best model was as follows:

```python
GradientBoostingRegressor(learning_rate=0.05, max_depth=2.0, n_estimators=100, min_samples_leaf=0.1, min_samples_split=0.1)
```



## Results
The results of our analysis are quite interesting. After selecting the final model with the optimal hyperparameters, we found that the model predicted adequately, but not incredibly well. Its out-of-sample cross-validated root mean squared error (RMSE) was 58.96s, which is lower than the standard deviation of 64.54s, so we can reliably say that our model predicts better than the average. Additionally, our in-sample RMSE was 55.92s. Because GBM's don't have a true R^2 statistic, we don't report one here. Since the model fits the data somewhat well, as can be seen by the relatively small RMSE, we can use it for limited inference. Based on the variable importance output by the model, the number of previous marathons is the most important variable, followed by the last marathon time run, the age of the athlete, and the year the race was held. In addition, we can use the partial effect plots generated by Python to find additional inference. In essence, these plots show the marginalized increase or decrease that a variable has on the outcome and can be useful for interpreting tree ensembles. The following plot describes the effect of the number of previous marathons run. As could be expected, a higher number of previous top 1000 performances correlates with faster times, since the athlete is likely both fast and consistent. This is shown by the steep downward trend of the plot.

![image](https://user-images.githubusercontent.com/58056607/145109986-16601288-c3dc-4e54-a9fa-bd67db61da2e.png)

Even though GDP wasn't a very important variable, it is interesting to visualize the effect that is often stereotyped in long-distance running of Kenyans and Ethiopians being great runners, as reflected by the steep drop at the beginning of the plot.

![image](https://user-images.githubusercontent.com/58056607/145110090-665878de-36fd-4823-b40d-f109fb4cb8c1.png)

This plot shows the effect of age on the times of runners. As could be expected, the older the athlete, the slower they run. This is also quite interesting considering that many past world record holders transferred to the marathon only upon 'aging out' of traditional track events due to the slowing effect of age.

![image](https://user-images.githubusercontent.com/58056607/145110181-5b4b751e-4b6f-4a1d-97cb-284a56378de9.png)

Finally, as a cautionary tale, we present our prediction results for the 2021 Olympics. It should first be noted that Olympic marathons rarely result in all-time top 1000 race times because the athletes are more focused on placing than time, which generally results in slower times. Therefore, the predictions of our model should not come as a surprise. Using the model trained on the top 1000 marathon performances ever, we had an RMSE of 775s, which is terrible compared to the standard deviation of the data of 384s. The reason for this is clearly explained by the mean bias of the predictions of -685s. Essentially, our model consistently predicts as if the Olympic marathon would result in many top 1000 all-time race times because it was trained on that data. Therefore, our predictions are not comparing 'apples to apples', which explains why they are so bad. In order to combat this, we would need to train our model instead on previous Olympic marathon data, which we don't have access to as of this report.




## Conclusion

In summary, our goal was to model and predict marathon times of the 2021 Olympics based on the data of athletes who ran past marathon races. The data was scraped from various sources online and passed through a gradient boosted regressor algorithm. As seen from our results, the model did not perform as expected due to comparison issues in our predictors and response variables. Fortunately with this project being an exploratory analysis and prediction on the data, we can consider improving our data collection methods in three ways for optimal analysis. First, we can fill in the many NaN values present with their actual data points. Second, we can add more data points to the whole data set as 1000 may be too few. Lastly, we can consider collecting better comparison variables to adequately predict on the y values. Overall, it was an exciting study on what is a historically overlooked sport. 
