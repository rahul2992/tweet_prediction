import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_train():
	file = "new_gruber_tweets.json"
	tweetdata = pd.read_json(file)
	return tweetdata
	
if __name__ == '__main__':
	tweetdata = load_train()
	tweetdata['created_at'] = pd.to_datetime(tweetdata['created_at'])
	
	tweet_time = []
	for time in tweetdata['created_at']:
		tweet_time.append(time)
	tweet_time.sort()
	
	time_dif = []
	for i in xrange(1, len(tweet_time) - 1):
		temp = (tweet_time[i] - tweet_time[i - 1]).seconds
		time_dif.append(temp)
		
	time_df = pd.Series(time_dif)
	time_df.hist(bins = 30, normed = True)
	plt.show()
	
	
	