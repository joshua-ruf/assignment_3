## Assignment 3 - CS 7641 - Joshua Ruf

##### implement 2 clustering algorithms:
1. k means
2. EM

##### implement 4 dimensionality reduction algorithms:
1. PCA
2. ICA
3. randomized PCA
4. another (variance threshold)

#### Steps:
- with two datasets (from assignment 1)
- run clustering algorithms and describe results
- run dimensionality reduction and describe results
- redo clustering with the dimensionality reduction (many results but describe most interesting)
- run the neural network on the reduced dimension version of one data set
- run the neural network on the reduced dimension version of one data set, using the clusters as labels

##### Turn in:
- briefly describe the datasets again
- explain methods, i.e. how to choose k
- description of clusters obtained
- rerun RP many times
- describe if the clusters were the same before and after dimensionality reduction
- describe the difference in the neural network this time around with reduced dimension, and with the clusters as labels

#### Datasets:

For both datasets I normalize the features to be within the range -1 and 1.

##### Dataset 1

This dataset consists of a number of employees, their pre hire survey scores (questions about their strengths and weaknesses) as well as some basic demographic features. In the first assignment I predicted whether the employee would be employed after 6 months. The dataset is quite wide with 48 features and 2739 observations. The features in this dataset are somewhat clustered in that there are a bunch of boolean values that correspond to strengths, and another group that corresponds to weaknesses. As well demographic features are another set. All in all I think there is a fair bit of room to go with dimensionality reduction and I expect the clustering algorithms to perform poorly until such measures are taken.

##### Dataset 2

This dataset consists of transformed movie reviews from the website rogerebert.com for all 3.5 and 4 star reviews. The idea was to predict what movies since his death in 2013 should be given the classification "great movie" as the critics that took over his site have refrained from using that distinction. The text reviews were passed to NLTK's simple sentiment analysis which for each review simply creates an index for the number of positive, negative, neutral, and compound score which is some transformation of the first three. I think the fact that this feature contains information about the first three features will result in some interesting results. I also included the review length in characters. In total there are 2415 observations and 5 features.

#### Clustering

##### Dataset 1

We see from a silhouette analysis that 5 clusters is the chosen optimum. The silhouette metric compares the intra-cluster distance to the mean nearest inter-cluster distance. Scores range between -1 and 1, with positive being more desirable clusters. Scores below zero are thought to mean that the sample is assigned to the wrong cluster.

![](plots/kmeans_dataset_1.png)

Here we visualize the first two principle components such that we can see how varied these clusters are (at least amongst the dimensions with the highest variance). It seems like clusters 1, 2, and 3 are more or less distinct in these two dimensions, however clusters 4 and 5 are overlaid with no clear decision boundary. This makes sense as the data is quite wide and the first two principle components do not capture a high amount of variance. As such in a higher dimensional space we can expect the clusters 4 and 5 to vary on other dimensions.

![](plots/kmeans_dataset_1_clusters_pca_1_and_2.png)

Interestingly, with the same silhouette approach EM chose 2 clusters. This is the number of classes in the original prediction problem so in some ways this is encouraging. That said, the silhouette scores are a fair bit lower than KMeans, indicating that the EM algorithm might not be as well suited to the problem. I would guess that EM's assigning probabilities instead of hard cluster thresholds means that the decision boundary is less defined.

![](plots/em_dataset_1.png)

Indeed that appears to be the case, with the exception of a tight cluster of points on the far right side of the 0th PC the 0th cluster basically surrounds the 1st cluster.

![](plots/em_dataset_1_clusters_pca_1_and_2.png)

#### Dimensionality Reduction

