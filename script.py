import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import expon
from operator import itemgetter

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
	#plt.show()
		
	exp_diff = []
	for time in time_dif:
		exp_diff.append(time - beta)
	exp_diff_df = pd.Series(exp_diff)
	return exp_diff, exp_diff_df
	
	
if __name__ == '__main__':
	######Exponential fit exploration
	tweetdata = load_train()
	tweetdata['created_at'] = pd.to_datetime(tweetdata['created_at'])
	
	time_df, time_dif = intertweet(tweetdata)
	time_df.hist(bins = 30, normed = True)
	#plt.show()
	
	count, division = np.histogram(time_dif, bins = 100, normed = True)
	fitParams, fitCov = curve_fit(fit_func, division[0:len(division)-1], count, p0=[0, 3e-4, 0])
	
	exp_diff, exp_diff_df = error_func(fitParams, division, time_dif)
	
	exp_diff_df.hist(bins = 100, normed = True, color = 'blue')
	#plt.show()
	
	###### Checking variance in dataset
	#Divide intertweet time into delta t and time to tweet
	
	timedata = []
	div = 10
	for t in time_dif:
		elapsed = np.arange(0, t, div)
		for t_elaps in elapsed:
			waiting = t - t_elaps
			t_sign = [t_elaps, waiting]
			timedata.append(t_sign)
	timedata = sorted(timedata, key = itemgetter(0)) 
	
	t_0,t_10,t_20,t_30,t_40,t_50,t_60,t_70,t_80,t_90,t_100 = ([] for i in range(11))
	for t in timedata:
		if (t[0] == 0):
			t_0.append(t[1])
		elif (t[0] == 10):
			t_10.append(t[1])
		elif (t[0] == 20):
			t_20.append(t[1])
		elif (t[0] == 30):
			t_30.append(t[1])	
		elif (t[0] == 40):
			t_40.append(t[1])	
		elif (t[0] == 50):
			t_50.append(t[1])	
		elif (t[0] == 60):
			t_60.append(t[1])	
		elif (t[0] == 70):
			t_70.append(t[1])	
		elif (t[0] == 80):
			t_80.append(t[1])	
		elif (t[0] == 90):
			t_90.append(t[1])
		elif (t[0] == 100):
			t_100.append(t[1])		

	t_0_100 = [t_10, t_20, t_30, t_40, t_50, t_60, t_70, t_80, t_90, t_100]
	#print t_0_100
	t_std = []
	t_mean = []
	t_div = np.arange(0, 100, 10)
	for t in t_0_100:
		t_std.append(np.std(t))
		t_mean.append(np.mean(t))

	t_std = pd.Series(t_std)
	t_mean = pd.Series(t_mean)
	t_div = pd.Series(t_div)
	
	t_comb = pd.concat([t_div, t_mean, t_std], axis = 1)
	t_comb.columns = [ 'elapsed', 'mean', 'standard_dev']
	print t_comb
	
	
			
	
			
	
		