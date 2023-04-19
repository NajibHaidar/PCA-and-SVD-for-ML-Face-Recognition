# PCA-and-SVD-for-ML-Face-Recognition

### Table of Contents
[Abstract](#Abstract)
<a name="Abstract"/>

[Sec. I. Introduction and Overview](#sec-i-introduction-and-overview)     
<a name="sec-i-introduction-and-overview"/>

[Sec. II. Theoretical Background](#sec-ii-theoretical-background)     
<a name="sec-ii-theoretical-background"/>

[Sec. III. Algorithm Implementation and Development](#sec-iii-algorithm-implementation-and-development)
<a name="sec-iii-algorithm-implementation-and-development"/>

[Sec. IV. Computational Results](#sec-iv-computational-results)
<a name="sec-iv-computational-results"/>

[Sec. V. Summary and Conclusions](#sec-v-summary-and-conclusions)
<a name="sec-v-summary-and-conclusions"/>


### Abstract
This project involves working with a dataset of 2414 downsampled grayscale images of faces, with 39 different faces and about 65 lighting scenes for each face. The goal is to compute and analyze the correlation between images using dot products, and to find the most highly correlated and uncorrelated images. Additionally, the project involves finding the first six eigenvectors with the largest magnitude eigenvalues and the first six principal component directions using SVD, comparing the first eigenvector and the first SVD mode, and computing the percentage of variance captured by each of the first 6 SVD modes. The implementation involves plotting the correlation matrix, the most highly correlated and uncorrelated images, and the first 6 SVD modes.

### Sec. I. Introduction and Overview
#### Introduction:

The analysis of high-dimensional data is a common challenge in modern data science. In this project, we will explore various techniques for analyzing high-dimensional data using a dataset of facial images. The dataset contains 39 different faces, with about 65 lighting scenes for each face, resulting in a total of 2414 images. Each image has been downsampled to 32x32 pixels and converted to grayscale, resulting in a matrix of size 1024x2414. This matrix will be stored in the variable X and referenced throughout this project description.

#### Overview:

We will begin by computing a correlation matrix between the first 100 images in the dataset and plotting it using pcolor. From the correlation matrix, we will identify the most highly correlated and uncorrelated images and plot them. Next, we will repeat the correlation matrix computation, but this time using a 10x10 matrix.

We will then explore two techniques for analyzing the dataset: eigenvector decomposition and singular value decomposition (SVD). In the first approach, we will compute the matrix $Y = XX^T$ (where $X^T$ is the transpose of X) and find the first six eigenvectors with the largest magnitude eigenvalue. In the second approach, we will use SVD to find the first six principal component directions.

Finally, we will compare the first eigenvector obtained from eigenvector decomposition with the first SVD mode obtained from SVD and compute the norm of the difference in their absolute values. Additionally, we will compute the percentage of variance captured by each of the first six SVD modes and plot the first six SVD modes.

Overall, this project will provide a practical example of how to analyze high-dimensional data using various techniques and demonstrate the importance of dimensionality reduction for visualizing and understanding complex datasets.

###  Sec. II. Theoretical Background

In this project, we will use some fundamental concepts and techniques from linear algebra and multivariate statistics. In particular, we will use matrix algebra, correlation analysis, eigendecomposition, singular value decomposition (SVD), and principal component analysis (PCA).

**Matrix algebra** is an essential tool in many areas of science and engineering. A matrix is a rectangular array of numbers or symbols, and matrix algebra involves operations such as addition, subtraction, multiplication, and inversion of matrices. In this project, we will work with a matrix of facial images, where each column of the matrix represents a downsampled grayscale image of size 32x32 pixels.

**Correlation analysis** is a statistical technique used to measure the relationship between two or more variables. In statistics, correlation refers to the measure of association or the strength of the relationship between two variables. Correlation can take on a value between -1 and 1, where -1 represents a perfect negative correlation, 0 represents no correlation, and 1 represents a perfect positive correlation.

A positive correlation means that as the value of one variable increases, the value of the other variable also tends to increase. For example, the height and weight of individuals have a positive correlation, which means that taller people tend to weigh more than shorter people.

A negative correlation means that as the value of one variable increases, the value of the other variable tends to decrease. For example, the number of hours spent studying and the number of errors made on a test may have a negative correlation, which means that as the number of hours spent studying increases, the number of errors made on the test decreases.

A zero correlation means that there is no association between the two variables. For example, the shoe size and the favorite color of individuals are not correlated, which means that there is no relationship between them.

In this project, we will compute a correlation matrix between the first 100 images in the matrix X, where each element of the matrix represents the dot product (correlation) between two images.

**Eigendecomposition** is a technique used to decompose a matrix into its eigenvectors and eigenvalues. In this project, we will use eigendecomposition to compute the first six eigenvectors of the matrix Y, where Y is the product of the matrix X and its transpose.

**PCA (Principal Component Analysis)** is a statistical technique used to identify patterns in multivariate data. In this project, we will use PCA to analyze the structure of the matrix X and to identify the most significant features that capture the variability in the data.

PCA is a commonly used statistical technique for dimensionality reduction. It aims to find the directions of maximal variance in a dataset and projects the data onto these directions to obtain a lower-dimensional representation of the data. The directions found by PCA are called principal components, and they are obtained by eigendecomposing the covariance matrix of the data.

**SVD (Singular Value Decomposition)** is a more general decomposition method that can be applied to any matrix, unlike eigendecomposition, which is only applicable to square matrices. In this project, we will use SVD to compute the first six principal component directions of the matrix X.

SVD is a matrix factorization technique that decomposes a matrix M (m*n) into three matrices: U, Σ, and V* (_in the code, these matricies have respectively been called U, S, and Vt_). The matrix Σ is diagonal and contains the singular values (in decreasing order) of the original matrix, which correspond to the square roots of the eigenvalues of the covariance matrix of the data. The matrix U contains the left singular vectors, and the matrix V* contains the right singular vectors.

It is important to note that any time a matrix A is multiplied by another matrix B, only two primary things can occur: B will _stretch_ and _rotate_ A. Hence, the three matricies SVD splits a matrix M into simply rotate M (V*), stretch M (Σ), and rotate M (U).

This concept can be visualized in figure 1 below. If we have a 2D disk (of a real sqaure matrix M), rotation first occurs, then stretching M will create an elipse with major and minor axes σ1 and σ2, and then we rotate again and find ourselves in a new coordiante system. We can say that u1 and u2 are unit orthonormal vectors known as principal axes along σ1 and σ2 and σ1 and σ2 are singular values. Thus, u1 and u2 determine the direction in which the stretching occurs while the singular valeus determine the magnitude of this stretch (eg. σ1u1).

![image](https://user-images.githubusercontent.com/116219100/232977674-cfd2a2f3-18ae-41e9-93cf-9585034cc857.png)
*Figure 1: Visualization of the 3 SVD matrices U, Σ, and V*

In the context of PCA, the principal components can be obtained from the right singular vectors of the data matrix. This is because the right singular vectors are the eigenvectors of the covariance matrix of the data. Therefore, the right singular vectors are often called the PCA modes.

However, the left singular vectors of the data matrix can also be useful in some applications, such as in image compression. In this case, the left singular vectors are called the SVD modes. The SVD modes and the PCA modes are related, but they are not exactly the same thing, as the SVD modes are not guaranteed to be orthogonal, while the PCA modes are always orthogonal.

### Sec. III. Algorithm Implementation and Development

Initally, the image data had to be collected and stored into a matrix called X so that the data could be manipulated. This was done using the following lines of code:

```
results = loadmat('yalefaces.mat')

X=results['X']
```

**results** is a Python dictionary object that contains keys and values. One of the keys in results is 'X', which corresponds to the variable containing the data matrix we are interested in. Therefore, we use results['X'] to extract the data matrix and assign it to the variable X.

Then, a 100x100 correlation matrix C was created between the dataset's first 100 images using Numpy's corrcoef method:

```
m = 100
Xm = X[:, 0:m]
C = np.corrcoef(Xm.T)
```

**np.corrcoef** uses dot product to calculate the correlation between the columns of the input matrix. Specifically, np.corrcoef first centers the data (subtracts the mean from each column), then calculates the covariance matrix by taking the _dot product_ of the centered matrix with its transpose, and finally normalizes the covariance matrix by dividing each element by the product of the standard deviations of the two corresponding variables. This normalized covariance matrix is equivalent to the correlation matrix.

The reason Xm.T (transpose of Xm) was passed in as a parameter rather than Xm itself is because, this way, it calculates the correlation coefficients between the columns of Xm (i.e., between the individual images) rather than between the rows. Moreover, the np.corrcoef() function expects the input data to have rows as variables and columns as observations. By transposing Xm, we get the desired format for inputting the data to np.corrcoef().

After plotting the correlation matrix using **pcolor**, we sought to find out which pair of images from our minimized dataset were most highly correlated and which pair were most uncorrelated:

```
max_corr = np.max((C - np.eye(m)))
print('Max Correlation:', max_corr)
max_indices = np.where((C - np.eye(m)) == max_corr)
print("Most highly correlated pair:", max_indices[0][0] + 1, "and", max_indices[1][0] + 1)
print()

min_corr = np.min((C - np.eye(m)))
print('Min Correlation:', min_corr)
min_indices = np.where((C - np.eye(m)) == min_corr)
print("Most highly uncorrelated pair:", min_indices[0][0] + 1, "and", min_indices[1][0] + 1)
```

The maxmimum correlation was calculated by simply using np.max() (which returns the maximum value in an n-dimensional Numpy array) after subtracting np.eye(m) from our 100x100 matrix C. **np.eye(m)** creates a 2D identity matrix of size m x m, where m is the number of columns (100) in the matrix Xm. By subtracting np.eye(m) from C, the diagonal elements of the correlation matrix C are set to zero. This is because the diagonal elements of C correspond to the correlation between the same image with itself, _which is always 1_. By setting these elements to zero, we ignore the correlation between an image and itself and only consider the correlation between different images. Then, we find the indicies of the max value(s) using **np.where()** which returns the indicies at which the condition _C - np.eye(m)) == max_corr_ is met. We can expect to get at least two pairs with the same max correlation since correlation matricies are symmetric and thus we may get the pair (4,5) and (5,4). When printing, the +1 ensures that we are naming the image correctly since Python arrays are 0-indexed while our data is not.

The minimum correlation and its indicies are then discovered in the exact same way except using np.min rather than np.max.

Next, the same process was executed once more except with a 10x10 correlation matrix of a randomly selected list of images: [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005] (Just for clarification, the first image is labeled as one, not zero like Python might do).

```
image_set = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]
m2 = len(image_set)
Xc = np.zeros((X.shape[0], m2))
for i, image in enumerate(image_set):
  Xc[:, i] = X[:, image - 1]

C2 = np.corrcoef(Xc.T)
```

First a list of the intended images was created and stored in image_set. Next, the new subset of X, now called Xc (previously called Xm) was created with the same number of rows as X and with m2 (10) columns creating a (1024, 10) matrix of zeros. Then, Xc was populated by column with each column corresponding to the image number - 1 (due to indexing). A new correlation matrix C2 was created using **np.corrcoef(Xc.T)**. The most and least correlated images were then found using the same technqiue that was previously descibed.

The next task in the project was to create the matrix $Y = XX^T$ (where $X^T$ is the transpose of X) and find the first six eigenvectors with the largest magnitude eigenvalue. This was done using the aforementioned **eigendecomposition**:

```
# Calculate Y
Y = np.matmul(X,X.T)

# Find the eigenvalues and eigenvectors of Y
eigvals, eigvecs = np.linalg.eigh(Y)

# Sort in descending order by largest magnitude eigenvalue
idx = eigvals.argsort()[::-1]   
eigVals_sorted = eigvals[idx]
eigvecs_sorted = eigvecs[:,idx]

# Grab the first six eigenvectors with largest magnitude eigenvalue
top_six_eigvecs = eigvecs[:, :6]
```

**np.linalg.eigh(Y)** performs the eigendecomposition and returns the eigenvalues and eigenvectors of the matrix Y. Then, argsort() was used to sort the eigenvalues indicies in ascending order and the slicing technique of [::-1] reverses the vector to return one in descending order. The eigenvector columns were then sorted based on this order of having the largest magnitude eigenvalue. The first six were then sliced out and stored in _top_six_eigvecs_.

Then, SVD was applied on the matrix X and the first six principal component directions were extracted:

```
# SVD decomposition of matrix X
U, S, Vt = np.linalg.svd(X)

# First six principal component directions
PC_directions = U[:, :6]
```

**np.linalg.svd(X)** performs the SVD on matrix X and returns the previously described matrices U, Σ, and V* as U, S, and Vt_ respectively). The first six principal component directions can then be sliced out of the U matrix.

Next, we compared the first eigenvector v1 with the first SVD mode u1 and compute the norm of difference of their absolute values:

```
v1 = top_six_eigvecs[0]
u1 = PC_directions[0, :]
print('First eigenvector v1 from Y:', v1)
print('First SVD mode u1 from X:', u1)

# Compute norm of difference of absolute values
diff_norm = np.linalg.norm(np.abs(v1) - np.abs(u1))

print()
print("Norm of difference:", diff_norm)
```

The norm of difference was found using **np.linalg.norm(np.abs(v1) - np.abs(u1))** where np.abs() returns the absolute value of its paramter.

Finally, the percentage of variance captured by each of the first 6 SVD modes was computed and plotted:

```
# Compute the total variance of the data
total_var = np.sum(S ** 2)

# Compute the percentage of variance captured by each of the first 6 SVD modes
variance_percentage = [(S[i] ** 2 / total_var) * 100 for i in range(6)]
```

The matrix S contains the singular values of X, which we use to compute the variance captured by each of the first 6 SVD modes.

The total variance of the data can be computed by summing the squares of all singular values (i.e., total_var = np.sum(S ** 2)). The percentage of variance captured by each of the first 6 SVD modes can then be computed as follows:

- Square each singular value up to the 5th index (i.e., S[i] ** 2 for i in the range 0 to 5).
- Divide each squared singular value by the total variance (total_var).
- Multiply the result by 100 to get the percentage of variance captured by each mode (i.e., (S[i] ** 2 / total_var) * 100 for i in the range 0 to 5).

### Sec. IV. Computational Results

In the first part, the 100x100 correlation matrix of the first 100 images produced expected results. We can see in figure 2 that the normalized color map ranges from -1 to 1 where -1 means highly uncorrelated (negative correlation) and 1 means perfectly correlated. The map is symmetic as expected since the 100 images were comapred with each other and so the diagonal shows perfectly correlated images sicne they are being compared to themselves at this axis.  


![image](https://user-images.githubusercontent.com/116219100/233007554-7e4d067c-0d35-43b7-8102-754d21b90913.png)
*Figure 2: 100x100 correlation matrix of the first 100 images*

We can see that some image pairs are highly positivly correlated (dark red) whilst others are highly negatively correlated (dark blue). We also have a great number of images that are not correlated at all (white).

![image](https://user-images.githubusercontent.com/116219100/233008197-921d118d-17eb-489a-9a01-8a85adb56170.png)
*Figure 3: Highly correlated and highly uncorrelated image pairs in first 100 images of X*

Figure 3 demonstrates the results obtained for the correlation pairing. Image 6 and image 63 had the highest positive correlation at a near perfect value of 97%. This makes sense as first of all the two images seem to be of the same person and the lighting conditions seem very similar in both images. I believe these were the two biggest driving factors in the algorithm concluding this correlation. We can see that the bright (yellow) spots are almost perfectly common for both images in the same spots. This definitely aids the algorithm in registering that the faces have the same structure and hence, are highly positvely correlated. 

Image 16 and image 82 had the highest negative correlation value at nearly -78%. The features that made the images 6 and 63 similar simply rarely exist here. We can see that where image 16 is lit up, image 82 dimmed down and the opposite is true as well. Therefore, the algorithm recognizes this opposing realation between the images and thus classifies them as negatively correlated images.

![image](https://user-images.githubusercontent.com/116219100/233012509-75725516-0f8a-4300-8133-223a2ffb6caf.png)
*Figure 4: 10x10 correlation matrix of the 10 images in random image set*

The analysis of figure 4 is simialr to that of figure 2. We can see the expected symmetry of the correlation matrix as well as the variation in correlation between different pairs of images.

![image](https://user-images.githubusercontent.com/116219100/233013092-842f776c-97da-4887-ac43-313e37dbd69d.png)
*Figure 5: Highly correlated and highly uncorrelated image pairs in random set of 10*

Figure 5 shows that from the random set, image 2400 and image 113 were the most highly positively correlated images with a correlation of 97%. However, after seeing the images, one could argue that these images may not be of the same person and so why have they been assigned near perfect correlation? It comes back to the shadows formed and what the algorithm can recognize as correlated. It sees that the entire left hemisphere of both images is almost completely dark and thus must 'look' the same. Additionally, both images have bright spots on their noses and a similar shape is formed there too. Note that in figure 3 the correlation was very high due to light being present and the face structure distributing the light similarly in both images, thus the algorithm recognized that these 2 images were also the same. Images 2400 and 113 have been paired due to lack of light not allowing structural face differences to appear: the complete opposite reasoning. Images 5 and 1024 are seen as highly uncorrelated with a correlation coefficient of -70%. Firstly, where image 5 is bright, image 1024 is dark thus indicating a negative correlation.

### Sec. V. Summary and Conclusions

In this project, we explored the use of least-squares error and 2D loss landscapes for fitting a mathematical model to a dataset. We found that the Nelder-Mead method was effective in finding the optimal parameters for the model, and that the 2D loss landscape provided useful insights into the behavior of the model function and the sensitivity of the model to changes in the parameter values. We also found that the 19th degree polynomial was the best model for fitting this particular dataset, but that was only due to overfitting and caution should be exercised when extrapolating beyond the range of the data. Aditionally, we concluded how training data has a large effect on the flexibility and accuracy of a model. Overall, this project demonstrates the usefulness of data modeling and the power of mathematical models for analyzing and understanding complex datasets in machine learning.
