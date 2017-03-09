import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import expon

def load_train():
	file = "new_gruber_tweets.json"
	tweetdata = pd.read_json(file)
	return tweetdata
	
def fit_func(x,a,b,c):
	return a*(b)*np.exp(-b*x) + c
	
def intertweet(tweetdata):
	tweet_time = []
	for time in tweetdata['created_at']:
		tweet_time.append(time)
	tweet_time.sort()
	
	time_dif = []
	for i in xrange(1, len(tweet_time) - 1):
		temp = (tweet_time[i] - tweet_time[i - 1]).seconds
		time_dif.append(temp)
		time_df = pd.Series(time_dif)
	return time_df, time_dif
	
def error_func(fitParams, division, time_dif):

	
	beta = (1/fitParams[1])*fitParams[0]+fitParams[1]

	t = division[0:len(division) - 1]
	time_df.hist(bins = 50, normed = True, color = 'blue')
	apx_dist = fit_func(t, fitParams[0], fitParams[1], fitParams[2])	
	plt.plot(t, apx_dist, color = 'yellow')
	plt.show()
	
	exp_diff = []
	for time in time_dif:
		exp_diff.append(time - beta)
	exp_diff_df = pd.Series(exp_diff)
	
	return exp_diff, exp_diff_df
	
	
if __name__ == '__main__':
	tweetdata = load_train()
	tweetdata['created_at'] = pd.to_datetime(tweetdata['created_at'])
	
	time_df, time_dif = intertweet(tweetdata)
	time_df.hist(bins = 30, normed = True)
	plt.show()
	
	count, division = np.histogram(time_dif, bins = 100, normed = True)
	fitParams, fitCov = curve_fit(fit_func, division[0:len(division)-1], count, p0=[0, 3e-4, 0])
	
	exp_diff, exp_diff_df = error_func(fitParams, division, time_dif)
	
	exp_diff_df.hist(bins = 100, normed = True, color = 'blue')
	plt.show()
	
			
	
		