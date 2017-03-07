import pandas as pd
import matplotlib.pyplot as plt

def load_train():
	file = "new_gruber_tweets.json"
	tweetdata = pd.read_json(file)
	return tweetdata
	
if __name__ == '__main__':
	tweetdata = load_train()
	print tweetdata 