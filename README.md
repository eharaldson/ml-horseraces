# ml-horseraces
This was a project I've worked on to predict winners of horse races using different ml models. 

I got the data from https://www.racing-bet-data.com/. I used some of the features, they had available: Pace, Weight, Age, etc. But most of the 140+ features I ended up with were historical based features: like the number of wins in the last 5 races for that horse or the average placing of that horse in the last 3 races. I also repeated these kinds of features for the specific horse and jockey pairing or looking at the past performance for that horse in that distance or similar distances. This is all done in the data_prep.ipynb

After all the feature engineering I decided to first look at a simple model that takes into account the features for that horse at that race time without comparing to other horses in the race. There are varying number of horses in the races so this will be an easy way to start and look at some simple models, like logistic regression and random forest for binary classification. I also trained a deep classification model using pytorch. 

To compare the models I would check a validation set which is the final 10% of races in the data ordered by time. The score would be the log loss of the winning horse for that race

-ln(p_w)

Where p_w is the model's probability prediction of the horse that actually won. This would penalise lower probabilities for the winning horse and so a lower average log loss would mean a better model. For logistic regression and random forest it had a log loss of 2.01 and 1.92 respectively. For the neural network it achieved a score of 1.85. Looking at the industry implied probability (using the odds for the horses) there was a log loss of 1.65 - so these models (even though they were using the betting odds in the model) did not achieve performance near that of the industry odds.

To get some interaction between horses I thought it could be done in 2 ways: 1 way was to use a model that could look at interactions between horse feature vectors and the second was to do some pairwise modelling (and then look at all the pairwise results for a race to get the horses likelihood of winning. I started with the first method and found the following papers on graph neural networks (GNNs) https://arxiv.org/abs/1812.08434. I thought the underlying methods of these types of networks and there inherent learning of interactions between nodes (independant of the size of the graph) could be applied to this problem. I therefore create the mix_model which uses a graph convolution layer to average the effect of learnable weights on each horse feature vector. Then pass this "race" embedding along with a certain horse feature vector into a linear layer to get a final prediction. 

I could only achieve 1.79 for the log loss on the validation using this so the next step would be to look at the pairwise modelling...
