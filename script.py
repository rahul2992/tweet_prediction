import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import expon
from operator import itemgetter
import math
import datetime
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

def load_train(file):
	tweetdata = pd.read_json(file)
	print "Loading training data..."
	return tweetdata
	
def fit_func(x,a,b,c):
	return a*(b)*np.exp(-b*x) + c
	
def intertweet(tweetdata):
	tweet_time = []
	for time in tweetdata['created_at']:
		tweet_time.append(time)
	
	time_dif = []
	for i in xrange(1, len(tweet_time)):
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
	
def tweet_length(tweetdata):
	tweet_ln = []
	for tweet in tweetdata['text']:
		length = len(tweet)
		tweet_ln.append(length)
	return tweet_ln
	
def split_intertweet(time_df, div):
	timedata = []
	for t in time_df:
		elapsed = np.arange(0, t, div)
		for t_elaps in elapsed:
			waiting = t - t_elaps
			t_sign = [t_elaps, waiting]
			timedata.append(t_sign)
	return timedata
	
def t_var_analysis(timedata):
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
	
	return t_comb

	
if __name__ == '__main__':
	######Exponential fit exploration
	file = "new_gruber_tweets.json"
	file_2 = "new_arment_tweets.json"
	file_3 = "new_siracusa_tweets.json"
	tweetdata = load_train(file)
	tweetdata_a = load_train(file_2)
	tweetdata_s = load_train(file_3)

	tweetdata['created_at'] = pd.to_datetime(tweetdata['created_at'])
	tweetdata.sort_values(by = 'created_at', ascending = 1, inplace = True)
	tweetdata.reset_index(drop = True, inplace = True)
	tweetdata_a['created_at'] = pd.to_datetime(tweetdata_a['created_at'])
	tweetdata_a.sort_values(by = 'created_at', ascending = 1, inplace = True)
	tweetdata_a.reset_index(drop = True, inplace = True)
	tweetdata_s['created_at'] = pd.to_datetime(tweetdata_s['created_at'])
	tweetdata_s.sort_values(by = 'created_at', ascending = 1, inplace = True)
	tweetdata_s.reset_index(drop = True, inplace = True)

	
	time_df, time_dif = intertweet(tweetdata)
	time_df_a, time_dif_a = intertweet(tweetdata_a)
	time_df_s, time_dif_s = intertweet(tweetdata_s)
	
	#time_df.hist(bins = 30, normed = True)
	#plt.show()
	
	#count, division = np.histogram(time_dif, bins = 100, normed = True)
	#fitParams, fitCov = curve_fit(fit_func, division[0:len(division)-1], count, p0=[0, 3e-4, 0])
	
	#exp_diff, exp_diff_df = error_func(fitParams, division, time_dif)
	
	#exp_diff_df.hist(bins = 100, normed = True, color = 'blue')
	#plt.show()
	
	###### Checking variance in dataset
	#Divide intertweet time into delta t and time to tweet
	#timedata = split_intertweet(time_df, div = 10)
	#timedata = sorted(timedata, key = itemgetter(0))
	#t_variance = t_var_analysis(timedata)
	#print t_variance
	
	armentMentions = []
	siracusaMentions = []
	
	
	#Length of tweets
	tweet_ln = tweet_length(tweetdata)
	tweet_ln = pd.Series(data = tweet_ln)
	
	#print tweetdata_a['user_mentions'].iloc[3]
	#print tweetdata['user'].iloc[0]
	#Mention distance of tweets 
	#Define a time reference  
	time_ref = tweetdata['created_at'].iloc[0]
	
	for i in xrange(0, len(tweetdata_a)):
		tmp = tweetdata_a['user_mentions'].iloc[i]
		tmp_time = tweetdata_a['created_at'].iloc[i]
		if (type(tmp) == list):
			for mention in tmp:
				if (mention['name'] == 'John Gruber'):
					armentMentions.append(tmp_time)
					
	for i in xrange(0, len(tweetdata_s)):
		tmp = tweetdata_s['user_mentions'].iloc[i]
		tmp_time = tweetdata_s['created_at'].iloc[i] 
		if (type(tmp) == list):
			for mention in tmp:
				if (mention['name'] == 'John Gruber'):
					siracusaMentions.append(tmp_time)
	tdist_arment = []
	tdist_siracusa = []
	mention_dist = []
	mention_dist_arment = []
	mention_dist_siracusa = []
	
	test = []					
	mention_name = []
	mention_d = []
	tweet_len = []
	t_elaps = []
	y = []
	div = 1000
	i = 0
	#print time_df.size
	#print tweetdata['created_at'].size
	num_tweets = len(tweetdata)
	#print num_tweets
	#print time_df.size
	#print len(armentMentions)
	#print len(siracusaMentions)
	for i in xrange(0, num_tweets - 1):
		#print i
		#Iterates from 0 - 3232; Drops the last tweet value
		tweet_tm = time_df.iloc[i]
		tweet_ln_temp = tweet_ln.iloc[i]
		#mention_d_temp = mention_dist['distance'].iloc[i]
		#mention_p_temp = mention_dist['person'].iloc[i]
		tweet_time_tmp = tweetdata['created_at'].iloc[i] 
		#print "t1:", tweet_time_tmp
		t_elapsedlist = np.arange(0, tweet_tm, div)
		for t in t_elapsedlist:
			#Add feature 1 - elapsed time
			t_elaps.append(t)
			#print "t:", t
			
			#Add feature 2 - Length of last tweet
			tweet_len.append(tweet_ln_temp)
			
			#Add label
			y_temp = tweet_tm - t
			y.append(y_temp)
			
			#Add feature 3,4 - Last mention person and distance
			t = datetime.timedelta(seconds = t)
			t_abs = t + tweet_time_tmp
			
			for j in xrange(0, len(armentMentions)):
				if (j < len(armentMentions) - 2):
					tmp_0 = abs(t_abs - armentMentions[j])
					tmp_1 = abs(t_abs - armentMentions[j+1])
					tmp_2 = abs(t_abs - armentMentions[j+2])
					if ((tmp_1<tmp_0) & (tmp_1<tmp_2)):
						mention_dist_arment.append(tmp_1)
						break
					elif ((tmp_0<tmp_1) & (tmp_0<tmp_2)):
						mention_dist_arment.append(tmp_0)
						break 
				elif (j == (len(armentMentions) - 2)):
					tmp_0 = abs(t_abs - armentMentions[j])
					tmp_1 = abs(t_abs - armentMentions[j+1])
					if (tmp_1<tmp_0):
						mention_dist_arment.append(tmp_1)
						break
					
			for j in xrange(0, len(siracusaMentions)):
				if (j < len(siracusaMentions) - 2):
					tmp_0 = abs(t_abs - siracusaMentions[j])
					tmp_1 = abs(t_abs - siracusaMentions[j+1])
					tmp_2 = abs(t_abs - siracusaMentions[j+2])
					if ((tmp_1<tmp_0) & (tmp_1<tmp_2)):
						mention_dist_siracusa.append(tmp_1)
						break
					elif ((tmp_0<tmp_1) & (tmp_0<tmp_2)):
						mention_dist_siracusa.append(tmp_0)
						break
				elif (j == (len(siracusaMentions) - 2)):
					tmp_0 = abs(t_abs - siracusaMentions[j])
					tmp_1 = abs(t_abs - siracusaMentions[j+1])
					if (tmp_1<tmp_0):
						mention_dist_siracusa.append(tmp_1)
						break
						
	X = pd.DataFrame({'elapsed_time': t_elaps,
					  'last_tweet_length': tweet_len,
					  'mention_dist_arment' : mention_dist_arment,
					  'mention_dist_siracusa': mention_dist_siracusa
					 })
	
	X['closest_mention'] = np.where(X['mention_dist_arment']<X['mention_dist_siracusa'], X['mention_dist_arment'], X['mention_dist_siracusa'])
	X['mention_person'] = np.where(X['mention_dist_arment']<X['mention_dist_siracusa'], 'arment', 'siracrusa')
	X = X.drop('mention_dist_siracusa', axis = 1)
	X = X.drop('mention_dist_arment', axis = 1)
	X['arment_mentions'] = np.where(X['mention_person'] == 'arment', 1, 0)
	X['siracusa_mentions'] = np.where(X['mention_person'] == 'siracusa', 1, 0)
	X = X.drop('mention_person', axis = 1)

	mention_seconds = []
	for time in X['closest_mention']:
		mention_seconds.append(time.seconds)
	mention_seconds = pd.Series(mention_seconds)
	X = pd.concat([X,mention_seconds], axis = 1)
	X = X.rename(columns = {0: 'mention_dist_sec'})
	X = X.drop('closest_mention', axis = 1)
	
	Y = pd.Series(y)
	#print X.head(5)
	#print Y.head(5)	
	
	#Splitting dataset and training KNN 
	test_size = 0.3
	seed = 7
	X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = test_size, random_state = seed)	
	
	X_train_n = np.asarray(X_train, dtype = 'int')
	Y_train_n = np.asarray(Y_train, dtype = 'int')
	X_val_n = np.asarray(X_val, dtype = 'int')
	Y_val_n = np.asarray(Y_val, dtype = 'int')
	
	
		
						