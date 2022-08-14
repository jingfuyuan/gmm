# Gaussian mixture model for collaborative filtering  

## Introduction  
This is a project of my **Machine learning with Python** course. The task is to build a mixture model for collaborative filtering.

In this project, a small portion of the Netflix database was used. The dataset is a partialy observed rating matrix, of which the rows represent the users and columns represent movies. One user only rates some movies and many others have not been rated by this user. The problem to be solved here is to predict the rating of those unwatched movies. Gaussian mixture model is used to solve this problem.

## What I did in this project  
- Implemented naive EM algorithm for data classification. 
- Compared K-means clustering and soft clustering with EM
- Implemented the fuction to calculate Bayesian information criterion (BIC).  ${BIC}(M) = l - \frac{1}{2}p \log n$ 
- Implemented EM algorithm for maxtrix completion
- Used EM algorithm to complete the partial Netflix matrix

