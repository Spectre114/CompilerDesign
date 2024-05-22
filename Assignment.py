





















































import nltk
from os import getcwd
import w1_unittest

nltk.download('twitter_samples')
nltk.download('stopwords')























filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)





import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples 

from utils import process_tweet, build_freqs











all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')









test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg 
test_x = test_pos + test_neg








train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)






print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))























freqs = build_freqs(train_x, train_y)


print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))















print('This is an example of a positive tweet: \n', train_x[0])
print('\nThis is an example of the processed version of the tweet: \n', process_tweet(train_x[0]))















































import math
def sigmoid(z): 
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    
    
    
    z = -z;
    h = 1/(1+ np.exp(z));
    
    
    return h






if (sigmoid(0) == 0.5):
    print('SUCCESS!')
else:
    print('Oops!')

if (sigmoid(4.92) == 0.9927537604041685):
    print('CORRECT!')
else:
    print('Oops again!')






w1_unittest.test_sigmoid(sigmoid)





































-1 * (1 - 0) * np.log(1 - 0.9999) 








-1 * np.log(0.0001) 





























































import numpy as np
def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    
    
    m = x.shape[0]
    
    for i in range(0, num_iters):
        
        
        
        z = np.dot(x,theta)
        
        
        h = sigmoid(z)
        delta = h-y
        
        J = (-1/m)*(np.sum(np.dot(y.T,np.log(h)) + np.dot((1-y).T,np.log(1-h))))

        
        theta = theta - (alpha/m)*(np.dot(x.T,delta))
        
    
    J = float(J)
    return J, theta







np.random.seed(1)

tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)

tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)


tmp_J, tmp_theta = gradientDescent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)
print(f"The cost after training is {tmp_J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}")












w1_unittest.test_gradientDescent(gradientDescent)






































def extract_features(tweet, freqs, process_tweet=process_tweet):
    '''
    Input: 
        tweet: a string containing one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''
    
    word_l = process_tweet(tweet)
    
    
    x = np.zeros(3) 
    
    
    x[0] = 1 
    
    
    
    
    for word in word_l:
        
        
        pair_pos = (word,1.0)
        pair_neg = (word,0.0)
        if(pair_pos in freqs):
            x[1] += freqs[pair_pos]
        
        if(pair_neg in freqs):
            x[2] += freqs[pair_neg]
        
    
    
    x = x[None, :]  
    assert(x.shape == (1, 3))
    return x








tmp1 = extract_features(train_x[0], freqs)
print(tmp1)












tmp2 = extract_features('blorb bleeeeb bloooob', freqs)
print(tmp2)











w1_unittest.test_extract_features(extract_features, freqs)















X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)


Y = train_y


J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")




























def predict_tweet(tweet, freqs, theta):
    '''
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output: 
        y_pred: the probability of a tweet being positive or negative
    '''
    
    
    x = extract_features(tweet, freqs)
    
    
    y_pred = sigmoid(np.dot(x,theta))
    
    
    
    return y_pred






for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    print( '%s -> %f' % (tweet, predict_tweet(tweet, freqs, theta)))    
    

















my_tweet = 'I am learning :)'
predict_tweet(my_tweet, freqs, theta)






w1_unittest.test_predict_tweet(predict_tweet, freqs, theta)






























def test_logistic_regression(test_x, test_y, freqs, theta, predict_tweet=predict_tweet):
    """
    Input: 
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (
    """
    
    
    
    
    y_hat = []
    
    for tweet in test_x:
        
        y_pred = predict_tweet(tweet, freqs, theta)
        
        if y_pred > 0.5:
            
            y_hat.append(1.0)
        else:
            
            y_hat.append(0.0)

    
    
    equal = 0
    m = 0
    for predicted, actual in zip(y_hat, test_y):
        if predicted == actual:
            equal += 1
        m += 1

    accuracy = np.float64(equal / m)

    
    
    return accuracy





tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")










w1_unittest.unittest_test_logistic_regression(test_logistic_regression, freqs, theta)











print('Label Predicted Tweet')
for x,y in zip(test_x,test_y):
    y_hat = predict_tweet(x, freqs, theta)

    if np.abs(y - (y_hat > 0.5)) > 0:
        print('THE TWEET IS:', x)
        print('THE PROCESSED TWEET IS:', process_tweet(x))
        print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(process_tweet(x)).encode('ascii', 'ignore')))











my_tweet = 'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!'
print(process_tweet(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
else: 
    print('Negative sentiment')

