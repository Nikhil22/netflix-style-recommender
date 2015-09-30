from numpy import *
import scipy.io, scipy.optimize

def normalize_ratings(ratings, did_rate):
	num_movies = ratings.shape[0]
	ratings_mean = zeros(shape = (num_movies, 1))
	ratings_norm = zeros(shape = ratings.shape)
	
	for i in range(num_movies):
		# Get all the indexes where there is a 1
		idx = where(did_rate[i] ==1)[0]
		
		# Calculate mean rating of ith movie only from user's that gave a rating
		ratings_mean[i] = mean(ratings[i, idx])
		ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]
		
	return (ratings_norm, ratings_mean)

def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	# Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
	# --------------------------------------------------------------------------------------------------------------
	# Get the first 30 (10 * 3) rows in the 48 X 1 column vector
	first_30 = X_and_theta[:num_movies * num_features]
	# Reshape this column vector into a 10 X 3 matrix
	X = first_30.reshape((num_features, num_movies)).transpose()
	# Get the rest of the 18 the numbers, after the first 30
	last_18 = X_and_theta[num_movies * num_features:]
	# Reshape this column vector into a 6 X 3 matrix
	theta = last_18.reshape(num_features, num_users ).transpose()
	
	# we multiply by did_rate because we only want to consider observations for which a rating was given
	difference = X.dot( theta.T ) * did_rate - ratings
	
	# we calculate the gradients (derivatives) of the cost with respect to X and theta
	X_grad = difference.dot( theta ) + reg_param * X
	theta_grad = difference.T.dot( X ) + reg_param * theta
	
	# wrap the gradients back into a column vector 
	return r_[X_grad.T.flatten(), theta_grad.T.flatten()]

			
def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	# Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
	# --------------------------------------------------------------------------------------------------------------
	# Get the first 30 (10 * 3) rows in the 48 X 1 column vector
	first_30 = X_and_theta[:num_movies * num_features]
	# Reshape this column vector into a 10 X 3 matrix
	X = first_30.reshape((num_features, num_movies)).transpose()
	# Get the rest of the 18 the numbers, after the first 30
	last_18 = X_and_theta[num_movies * num_features:]
	# Reshape this column vector into a 6 X 3 matrix
	theta = last_18.reshape(num_features, num_users ).transpose()
	
	# we multiply by did_rate because we only want to consider observations for which a rating was given
	# we calculate the sum of squared errors here.  
	# in other words, we calculate the squared difference between our hypothesis (predictions) and ratings
	cost = sum( (X.dot( theta.T ) * did_rate - ratings) ** 2 ) / 2
	
	# we get the sum of the square of every element of X and theta
	regularization = (reg_param / 2) * (sum( theta**2 ) + sum(X**2))
	return cost + regularization
	
def loadMovies():
	movie_dict = {}
	movie_index = 0
	with open('/Users/nikhilbhaskar/Desktop/SmoothOperator/movies.txt', 'rb') as yo:
		file_contents = yo.readlines()
		for content in file_contents:
			movie_dict[movie_index] = content.strip().split(' ', 1)[1]
			movie_index += 1

	return movie_dict

# initialize some random movie ratings
ratings = array([[8, 4, 0, 0, 4], [0, 0, 8, 10, 4], [8, 10, 0, 0, 6], [10, 10, 8, 10, 10], [0, 0, 0, 0, 0], [2, 0, 4, 0, 6], [8, 6, 4, 0, 0], [0, 0, 6, 4, 0], [0, 6, 0, 4, 10], [0, 4, 6, 8, 8]]);
# Here's what the ratings matrix looks like
'''
[[ 8  4  0  0  4]
 [ 0  0  8 10  4]
 [ 8 10  0  0  6]
 [10 10  8 10 10]
 [ 0  0  0  0  0]
 [ 2  0  4  0  6]
 [ 8  6  4  0  0]
 [ 0  0  6  4  0]
 [ 0  6  0  4 10]
 [ 0  4  6  8  8]]
'''

# logical not
did_rate = (ratings != 0) * 1;
# Here's what the did_rate matrix looks like
'''
[[1 1 0 0 1]
 [0 0 1 1 1]
 [1 1 0 0 1]
 [1 1 1 1 1]
 [0 0 0 0 0]
 [1 0 1 0 1]
 [1 1 1 0 0]
 [0 0 1 1 0]
 [0 1 0 1 1]
 [0 1 1 1 1]]
'''

# Let's rate some movies (my name is Nikhil)
nikhil_ratings = zeros((10, 1))
nikhil_ratings[0] = 7;
nikhil_ratings[4] = 8;
nikhil_ratings[7] = 3;

# Here's what nikhil_ratings column vector looks like
'''
[[ 7.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 8.]
 [ 0.]
 [ 0.]
 [ 3.]
 [ 0.]
 [ 0.]]
'''

# add nikhil_ratings to ratings
ratings = append(nikhil_ratings, ratings, axis=1);
# Here's what the updated ratings matrix looks like now
'''
[[  7.   8.   4.   0.   0.   4.]
 [  0.   0.   0.   8.  10.   4.]
 [  0.   8.  10.   0.   0.   6.]
 [  0.  10.  10.   8.  10.  10.]
 [  8.   0.   0.   0.   0.   0.]
 [  0.   2.   0.   4.   0.   6.]
 [  0.   8.   6.   4.   0.   0.]
 [  3.   0.   0.   6.   4.   0.]
 [  0.   0.   6.   0.   4.  10.]
 [  0.   0.   4.   6.   8.   8.]]
'''

did_rate = append(((nikhil_ratings != 0) * 1), did_rate, axis = 1) 
# Here's what the updated did_rate matrix looks like now
'''
[[1 1 1 0 0 1]
 [0 0 0 1 1 1]
 [0 1 1 0 0 1]
 [0 1 1 1 1 1]
 [1 0 0 0 0 0]
 [0 1 0 1 0 1]
 [0 1 1 1 0 0]
 [1 0 0 1 1 0]
 [0 0 1 0 1 1]
 [0 0 1 1 1 1]]
'''

# Normalize ratings
ratings_norm, ratings_mean = normalize_ratings(ratings, did_rate)

# Here's what the normalized ratings matrix looks like
'''
 [[ 1.25        2.25       -1.75        0.          0.         -1.75      ]
 [ 0.          0.          0.          0.66666667  2.66666667 -3.33333333]
 [ 0.          0.          2.          0.          0.         -2.        ]
 [ 0.          0.4         0.4        -1.6         0.4         0.4       ]
 [ 0.          0.          0.          0.          0.          0.        ]
 [ 0.         -2.          0.          0.          0.          2.        ]
 [ 0.          2.          0.         -2.          0.          0.        ]
 [-1.33333333  0.          0.          1.66666667 -0.33333333  0.        ]
 [ 0.          0.         -0.66666667  0.         -2.66666667  3.33333333]
 [ 0.          0.         -2.5        -0.5         1.5         1.5       ]]
 '''
 # Here's what the ratings_mean matrix looks like
'''
[[ 5.75      ]
 [ 7.33333333]
 [ 8.        ]
 [ 9.6       ]
 [ 8.        ]
 [ 4.        ]
 [ 6.        ]
 [ 4.33333333]
 [ 6.66666667]
 [ 6.5       ]]
'''

num_movies, num_users = shape(ratings)
num_features = 3

# Initialize Parameters theta (user_prefs), X (movie_features)

movie_features = random.randn( num_movies, num_features )
user_prefs = random.randn( num_users, num_features )
initial_X_and_theta = r_[movie_features.T.flatten(), user_prefs.T.flatten()] 

# Regularization paramater
reg_param = 30.0

# fprime simply refers to the derivative (gradient) of the calculate_cost function
# We iterate 100 times
minimized_cost_and_optimal_params = scipy.optimize.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta, \
								args=(ratings, did_rate, num_users, num_movies, num_features, reg_param), \
								maxiter=100, disp=True, full_output=True )
								
# Retrieve the minimized cost and the optimal values of the movie_features (X) and user_prefs (theta) matrices
cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]


# Extract movie_features and user_prefs from optimal_movie_features_and_user_prefs
first_30 = optimal_movie_features_and_user_prefs[:num_movies * num_features]
movie_features = first_30.reshape((num_features, num_movies)).transpose()
last_18 = optimal_movie_features_and_user_prefs[num_movies * num_features:]
user_prefs = last_18.reshape(num_features, num_users ).transpose()

# Make predictions by calculating the dot product of the movie_features and user_prefs matrices
all_predictions = movie_features.dot( user_prefs.T )

# get my predictions by extracting the first column vector from all_predictions
# add back the mean 
predictions_for_nikhil = all_predictions[:, 0:1] + ratings_mean

# we use argsort . we cannot simply use sort(predictions_for_nikhil)
sorted_indexes = predictions_for_nikhil.argsort(axis=0)[::-1]
predictions_for_nikhil = predictions_for_nikhil[sorted_indexes]

# load our movies
all_movies = loadMovies()

# since we only have 10 movies, let's display all ratings
for i in range(num_movies):
		# grab index (integer), which remember, are all sorted based on the prediction values 
		index = sorted_indexes[i, 0]
		print "Predicting rating %.1f for movie %s" % (predictions_for_nikhil[index], all_movies[index])
		
# Here's the result
'''
Predicting rating 7.4 for movie A Very Harold and Kumar Christmas (2011)
Predicting rating 8.0 for movie Straight Outta Compton (2011)
Predicting rating 6.7 for movie Notorious (2009)
Predicting rating 8.1 for movie Ted (2012)
Predicting rating 4.4 for movie Cinderella (2015)
Predicting rating 4.0 for movie Toy Story 3 (2010)
Predicting rating 6.1 for movie Frozen (2013)
Predicting rating 9.8 for movie Harold and Kumar Escape From Guantanamo Bay (2008)
Predicting rating 5.8 for movie Tangled (2010)
Predicting rating 6.6 for movie Get Rich Or Die Tryin' (2005)
'''
