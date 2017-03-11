# tweet_prediction
An attempt to analyse tweet data of three connected people, to predict when will a person tweet next. The dataset is taken from an udacity lecture.
This was an interesting analysis. I took the dataset from an Udacity class on machine learning. 

Question to answer - "When will a person tweet next?"

The simplest analysis involves looking at the intertweet times and making a regression model to make the estimate. 
The intertweet time shows a nice exponential distribution. So, the first step was to use a exponential model and calculate an estimated value by curve fitting

The error is still as high as 12 minutes for 50% of the distribution. 

The dataset is very interesting, and clearly the tweeting process is not completely random. It is influenced by many other parameters like your friends. 

The next step I took was to create the following dataframe for input: 

Elapsed time since last tweet| Length of last tweet | Time since last mention | Last mention person || Label (Y) = Time to tweet |

1. Elapsed distance can be found by dividing the intertweet times. 
2. However, finding the closest mention distance was involved as you would run out with computation as the number of steps increase. 

I used a division size of 10 to create a dataset of almost 1.38 million points, and used KNN from sklearn to make the point estimation. 

Results = Accuract of 75% of prediction of tweet time in 30 seconds
