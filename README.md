# NBARookie19
Predicting the points per game of rookies of draft class 2019 using machine learning.

First of all I collected collected college stats of past first round picks and then their NBA stats. Only players drafted after 2013 were used.
Then I collected college stats of prominent scorers of 2019 draft class.

Using scikit-learn, I created three models: a linear regression, a ridge regression, and a support vector regression.
Each model uses scikit-learn's train test split function with a test size of 20%.
The players' data was then put in to each of the three models to predict their points per game.
