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

The maxmimum correlation was calculated by simply using np.max() (which returns the maximum value in an n-dimensional Numpy array) after subtracting np.eye(m) from our 100x100 matrix C. **np.eye(m)** creates a 2D identity matrix of size m x m, where m is the number of columns (100) in the matrix Xm. By subtracting np.eye(m) from C, the diagonal elements of the correlation matrix C are set to zero. This is because the diagonal elements of C correspond to the correlation between the same image with itself, _which is always 1_. By setting these elements to zero, we ignore the correlation between an image and itself and only consider the correlation between different images. Then, we find the indicies of the max value(s) using **np.where()** which returns the indicies at which the condition _C - np.eye(m)) == max_corr_ is met.

### Sec. IV. Computational Results

In the first part, it can be seen from the plot below that the resulting fit was not very accurate but still pretty close to emulating the flucuations of the given data points. The **minimum error** was determined to be around **1.593** while the 4 optimal parameters found were **A=2.17**, **B=0.91**, **C=0.73**, **D=31.45**.

![image](https://user-images.githubusercontent.com/116219100/231102068-249d81e0-9f32-4f40-bfec-1a8ee8a93291.png)
*Figure 2: Fitting f(x) on X and Y using least-squares fit*


In the next part, the 2D loss (error) landscape's color maps yieled interesting results and provided insight to the affect certain parameters have on the function's number of minima.

![image](https://user-images.githubusercontent.com/116219100/231120449-630141c5-62f8-4375-89e4-be8d496d7aa1.png)
*Figure 3: Error Landscape, fix AB and sweep CD*

In figure 3 we can see that the effect of C when A and B are fixed is very minimal. For the entire range of C, the color is almost the same whereas as D ranges from 15 to 45 the error increases dramatically from well below 100 to over 500. This shows that the value of D has a very big impact when A and B are fixed.

![image](https://user-images.githubusercontent.com/116219100/231120490-b3197e4d-729a-43c4-a936-de2912acbc60.png)
*Figure 4: Error Landscape, fix AC and sweep BD*

Figure 4 is very interesting because we can see that when A and C are fixed, the error tends to minimize the closer B gets to around 17. Note that this is true for D being between 15 and 45. At first glance it looks like D does not have any influence, however, some ripples can be noticed at values of around 22, 27, 34, and 41 for D. We can see that at these values of D, the error is very minimal when B is between 10 and 15. 

![image](https://user-images.githubusercontent.com/116219100/231120534-47e8030b-c240-4e9d-92d9-63001816dc9a.png)
*Figure 5: Error Landscape, fix AD and sweep BC*

In figure 5 we can see that the effect of C when A and D are fixed is very minimal. For the entire range of C, the color is almost the same whereas as B ranges from 0 to 30 the error increases dramatically from well below 100 to over 500. This shows that the value of B has a very big impact when A and D are fixed.

![image](https://user-images.githubusercontent.com/116219100/231120588-bb073468-cb03-4b7b-badb-3af858a114d4.png)
*Figure 6: Error Landscape, fix BC and sweep AD*

Figure 6 is a good example of what a convex error surface would look like. Notice how no matter what point we start with here, if we follow the color descent gradient we will always end up in the minimum error region that is well below 5. We can also see that this lowest error occurs at almost A=17 and D=18 when B and C are fixed to their optimal values.

![image](https://user-images.githubusercontent.com/116219100/231120638-f7cfaf66-13c8-465e-8b23-35193e4023a6.png)
*Figure 7: Error Landscape, fix BD and sweep AC*

In figure 7 we can see that the effect of C when B and D are fixed is very minimal. For the entire range of C, the color is almost the same whereas as A ranges from 0 to 30 the error increases dramatically from well below 100 to over 500. This shows that the value of A has a very big impact when B and D are fixed.

![image](https://user-images.githubusercontent.com/116219100/231120679-7027a978-9c71-4eb3-8563-553a2a0a95e0.png)
*Figure 8: Error Landscape, fix BD and sweep AC*

Figure 8 is very interesting because we can see that when C and D are fixed, the error tends to minimize the closer B gets to around 0. Note that this is true for A being between 0 and 30. At first glance it looks like A does not have any influence, however, some ripples can be noticed at values of A. We can see that at these values of A, the error is very minimal, especially the closer B is to 0. It should also be noted that there are about 3 yellow lines showing that if A is at these values and B is closer to 30, the error increases by a good amount.


Moving on to testing different polynomial models against different sections of the data, we can see how training on different portions of the data leads to varying errors:

```
LINE MODEL, -Test End-:
Training Error: 2.242749386808539
Test Error: 3.4392356574390317

LINE MODEL, -Test Middle-:
Training Error: 1.8516699043293752
Test Error: 2.943490105614687


PARABOLA MODEL, -Test End-:
Training Error: 2.125539348277377
Test Error: 9.035130793088825

PARABOLA MODEL, -Test Middle-:
Training Error: 1.85083641159579
Test Error: 2.910426615782527


19th DEGREE POLYNOMIAL MODEL, -Test End-:
Training Error: 0.02835144302630829
Test Error: 30023572038.458946

19th DEGREE POLYNOMIAL MODEL, -Test Middle-:
Training Error: 0.16381508563760222
Test Error: 507.53804019224077
```

Recall that the given data points oscillate but still steadily increase as X increases.

For the line model, both types of tests yielded very low error in training and testing. However it seems that this model adapted better (although slim) to the version where we removed the middle values during training. This could be interpreted that since we are drawing a line here, the incline is what is important. The total incline can be better depicted by taking points in the beginning and then end of the entire data set and thus this would reduce error.

For the parabola model, the test error after training on the last 10 points was noticeably higher than that on the middle points. This could be interpreted that since the degree of curvature of a parabola will depend on future points, this was better captured when taking points in the beginning and then end of the entire data set and thus this would reduce error.

For the 19th degree polynomial model, the training error in both cases was almost nonexistent. This makes sense since a 19th degree polynomial can pass through all 10 data points with ease due to its degree. However, the test error was astronomical when testing the last 10 points and very large when testing the middle 10 points. This greatly descibes a phenomenon called overfitting. The model has been trained so strongly on the training data (by passing through each point) that when it is given data outside this data set its behaviour is very offset. The test on the middle points was definitely much better (although still not good) than the test on the end points. The interpretation for this is that since the problem here is overfitting, the gap presented by skipping the middle 10 points in training allows this model to be less overfit than its counterpart. The one tested on the first 20 points catches onto the given data points much more strongly and thus results in a much stronger effect of overfitting thus greatly increasing the error when the data changes from what the model was trained on.

### Sec. V. Summary and Conclusions

In this project, we explored the use of least-squares error and 2D loss landscapes for fitting a mathematical model to a dataset. We found that the Nelder-Mead method was effective in finding the optimal parameters for the model, and that the 2D loss landscape provided useful insights into the behavior of the model function and the sensitivity of the model to changes in the parameter values. We also found that the 19th degree polynomial was the best model for fitting this particular dataset, but that was only due to overfitting and caution should be exercised when extrapolating beyond the range of the data. Aditionally, we concluded how training data has a large effect on the flexibility and accuracy of a model. Overall, this project demonstrates the usefulness of data modeling and the power of mathematical models for analyzing and understanding complex datasets in machine learning.
