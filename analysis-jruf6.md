## Assignment 3 - CS 7641 - Joshua Ruf

##### Turn in:
- briefly describe the datasets again
- explain methods, i.e. how to choose k
- description of clusters obtained
- rerun RP many times
- describe if the clusters were the same before and after dimensionality reduction
- describe the difference in the neural network this time around with reduced dimension, and with the clusters as labels

### Datasets:

For both datasets I normalize the features to be within the range -1 and 1.

##### Dataset 1

This dataset consists of a number of employees, their pre hire survey scores (questions about their strengths and weaknesses) as well as some basic demographic features. In the first assignment I predicted whether the employee would be employed after 6 months. The dataset is quite wide with 48 features and 2739 observations. The features in this dataset are somewhat clustered in that there are a bunch of boolean values that correspond to strengths, and another group that corresponds to weaknesses. As well demographic features are another set. All in all I think there is a fair bit of room to go with dimensionality reduction and I expect the clustering algorithms to perform poorly until such measures are taken.

##### Dataset 2

This dataset consists of transformed movie reviews from the website rogerebert.com for all 3.5 and 4 star reviews. The idea was to predict what movies since his death in 2013 should be given the classification "great movie" as the critics that took over his site have refrained from using that distinction. The text reviews were passed to NLTK's simple sentiment analysis which for each review simply creates an index for the number of positive, negative, neutral, and compound score which is some transformation of the first three. I think the fact that this feature contains information about the first three features will result in some interesting results. I also included the review length in characters. In total there are 2415 observations and 5 features.

### Clustering

##### Dataset 1

We see from a silhouette analysis that 5 clusters is the chosen optimum. The silhouette metric compares the intra-cluster distance to the mean nearest inter-cluster distance. Scores range between -1 and 1, with positive being more desirable clusters. Scores below zero are thought to mean that the sample is assigned to the wrong cluster.

![](plots/kmeans_dataset_1.png)

Here we visualize the first two principle components such that we can see how varied these clusters are (at least amongst the dimensions with the highest variance). It seems like clusters 1, 2, and 3 are more or less distinct in these two dimensions, however clusters 4 and 5 are overlaid with no clear decision boundary. This makes sense as the data is quite wide and the first two principle components do not capture a high amount of variance. As such in a higher dimensional space we can expect the clusters 4 and 5 to vary on other dimensions.

![](plots/kmeans_dataset_1_clusters_pca_1_and_2.png)

Interestingly, with the same silhouette approach EM chose 2 clusters. This is the number of classes in the original prediction problem so in some ways this is encouraging. That said, the silhouette scores are a fair bit lower than KMeans, indicating that the EM algorithm might not be as well suited to the problem. I would guess that EM's assigning probabilities instead of hard cluster thresholds means that the decision boundary is less defined.

![](plots/em_dataset_1.png)

Indeed that appears to be the case, with the exception of a tight cluster of points on the far right side of the 0th PC the 0th cluster basically surrounds the 1st cluster.

![](plots/em_dataset_1_clusters_pca_1_and_2.png)

### Dimensionality Reduction

On the x axis I've plotted number of features and on the y axis I've plotted the root mean squared difference between the original data and the data transformed into the smaller space and then projected back into the original space. This can be thought of as a measure of information loss since rmse of zero means that no information is stripped away while higher values correspond to more information loss. Looking at the figure there are a few things to note:

1. the curvature of the relationship between number of features and information loss is different for each algorithm: PCA and ICA have a nice kink around feature 34 indicating that there are much more diminishing returns of adding features beyond that point. Variance Threshold is more linear with a smaller kink around the same place. Random Projections however have the opposite curve, indicating that the information gain (at least according to the crude RMSE metric being used) is actually greatest at about feature 35. This is totally possible since by choosing randomly the likelihood of choosing the most useful features increases as more features are chosen overall.
2. the PCA and FastICA algorithms return almost the same results (blue plus orange line combines in seaborn to a nice brown). I'm a bit surprised by this, maybe for this particular dataset the orthogonality and statistical dependence achieve similar results.
3. None of the algorithms point to there being just a handful of features that explain a large amount of the information contained in the original data.
4. It makes sense to me that PCA and ICA would retain the most information, the randomized projections is a faster version of PCA (sacrificing choosing the optimal direction of variance) and variance threshold is just a much simpler approach that does not consider features together at all.

Overall I'm not surprised, going into this (as well as the first) assignment, I knew this data contained a lot of noise and perhaps even measurement error such that dimensionality reduction is necessary but also bounded in its utility.

![](plots/dimensionality_reduction_dataset_1.png)

These results are confirmed more or less by the variance explained, obtained from the eigenvalues of the PCA algorithm. Again, around feature 34 the additional variance explained by the last features is quite low. There appears to be a slight kind around feature 10, but it's very minor.

![](plots/dimensionality_reduction_dataset_1_pca_variance.png)

### Clustering After Dimensionality Reduction

While the visualizations above show that 32-34 is likely the optimal number of features to reduce to, I chose to reduce it more to get a better sense of how the clustering algorithms perform with a much smaller feature space. I chose 17 features, ran all four dimension reduction techniques and passed them to the two clustering algorithms. As before, EM chooses k=2 across all dimension reduction techniques. Interestingly, KMeans now does the same for most dimension reduction techniques, PCA being the only exception. Overall, we do see that the silhouette metrics are generally higher than with the full feature space. I think the combination of all silhouette scores being higher but the clustering algorithms preferring to use only one cluster is interesting. To me, it implies that the data does not have clear clusters.

![](plots/Clustering_with_17_features_of_dataset_1.png)

Showing the same first and second principle components, we see an almost identical story, indicating that the clusters have not moved much as a result of removing the least useful features.

![](plots/kmeans_dataset_1_with_17_features_clusters_pca_1_and_2.png)

### Neural Network

#### Full feature vs reduced feature dataset, predicting labels

I ran the neural network from the first assignment on dataset 1 twice using the same hyperparameters in the grid search cross validation. The first time I used the full featured dataset with 48 features, while the second time I used only the 17 features selected by PCA. In both scenarios I upsampled the minority class to deal with class imbalance, and used f1-score as the selection metric in cross validation. As such, the training accuracy score is not that helpful as it runs on the upsampled data.

![](plots/neural_net_confusion_matrix_full_dataset_1.png)

Unsurprisingly, the reduced feature dataset trained in less than half the time of the full feature dataset, 0.088s to 0.22s -- the numbers are not that different but remember there are just under 3000 training samples. The two models performed very similarly in terms of f1 score. In training, the best CV f1 score was 0.6658 for the smaller dataset compared to 0.6688 for the larger dataset. Testing on unseen data, the smaller model achieved an f1 score of 0.2261, and the larger model achieved an f1 score of 0.2269. That said, with the reduced training time, the smaller model is likely the better bet. Surprisingly the larger model opted to use a hidden layer with 2 neurons while the smaller model used 8 neurons. One would expect the opposite so I am of the opinion that this relates back to the original problems working with this data in assignment 1: it's really noisy. Since these data describe employment and pre hire surveys the data can be highly subjective and employees leave roles for any number of reasons that might not be predictable in data.

![](plots/neural_net_confusion_matrix_reduced_dataset_1.png)

#### Full feature vs reduced feature dataset, predicting clusters

Using the reduced feature data, I predicted the clusters obtained using KMeans with k=2 since the original labels have two classes. This means compare the neural network's ability to predict the labels in the same way as the clusters. As well, we can begin by seeing how well the clusters approximate the labels. In our data the labels match the clusters with an accuracy of 72% and this corresponds to an f1-score of about 0.32 for comparison. This should not be thought of as a measure of how well KMeans performs because our original labels could have been very poorly designed and contain no information at all - were that the case it does not reflect badly on the clustering algorithm but instead the practitioner (or the data).

![](plots/neural_net_confusion_matrix_reduced_dataset_1_predicting_clusters.png)

Overall, the neural network had a comparatively easy time predicting the clusters as compared to the actual labels. For both f1 score and accuracy it achieved train and test scores of over 99%. This speaks to the way in which the means are constructed, and ease with which a neural network can learn that kind of linear function. Since the clusters are born out of the data itself, there is effectively no redundant information, every feature is used in the construction of a cluster. It seems likely then, that by the universal approximation theorem, a neural network with 3 sufficiently large hidden layers can predict the clusters coming from a KMeans algorithm with arbitrarily high precision. Furthermore, a number of models fit in this CV process were stopped early by reaching the max number of iterations, further indication that improving upon the accuracy is possible.

