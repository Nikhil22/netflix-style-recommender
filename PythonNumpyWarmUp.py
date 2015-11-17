
# coding: utf-8

# In[25]:

# Handy for matrix manipulations, likE dot product and tranpose

from numpy import *


# In[26]:

# Declare and initialize a 2d numpy array (just call it a matrix, for simplicity)
# This how we will be organizing our data. very simple, and easy to manipulate.

data = array([[1, 2, 3], [1, 2, 3]])
print data


# In[27]:

# Get dimensions of matrix

data.shape


# In[28]:

# Declare and initialize a matrix of zeros

zeros_matrix = zeros((1,2))
print zeros_matrix


# In[29]:

# Declare and initialize a matrix of ones

ones_matrix = ones((1,2))
print ones_matrix


# In[30]:

# Declare and initialize a matrix of random integers from 0-10

rand_matrix = random.randint(10, size = (10, 5))
print rand_matrix


# In[31]:

# Declare and initialize a column vector

col_vector = random.randint(10, size = (10, 1))
print col_vector


# In[32]:

# Access and print the first element of the column vector

print col_vector[0]


# In[33]:

# Change the first element of the column vector

col_vector[0] = 100
print col_vector


# In[34]:

# Access and print the first element of rand_matrix
print rand_matrix[0, 0]


# In[35]:

# Access and print the all rows of first column of rand_matrix
print rand_matrix[:, 0:1]


# In[36]:

# Access and print the all columns of first row of rand_matrix
print rand_matrix[0:1, :]


# In[37]:

# Access the 2nd, 3rd and 5th columns fo the first row rand_matrix
# Get the result in a 2d numpy array
cols = array([[1,2,3]])
print rand_matrix[0, cols]


# In[38]:

# Flatten a matrix
flattened = rand_matrix.T.flatten()
print flattened


# In[39]:

# Dot product
rand_matrix_2 = random.randint(10, size = (5,2))
dot_product = rand_matrix.dot(rand_matrix_2)
print dot_product


# In[ ]:
