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

**Correlation analysis** is a statistical technique used to measure the relationship between two or more variables. In this project, we will compute a correlation matrix between the first 100 images in the matrix X, where each element of the matrix represents the dot product (correlation) between two images.

**Eigendecomposition** is a technique used to decompose a matrix into its eigenvectors and eigenvalues. In this project, we will use eigendecomposition to compute the first six eigenvectors of the matrix Y, where Y is the product of the matrix X and its transpose.

**SVD** is a more general decomposition method that can be applied to any matrix, unlike eigendecomposition, which is only applicable to square matrices. In this project, we will use SVD to compute the first six principal component directions of the matrix X.

**PCA** is a statistical technique used to identify patterns in multivariate data. In this project, we will use PCA to analyze the structure of the matrix X and to identify the most significant features that capture the variability in the data.

### Sec. III. Algorithm Implementation and Development

The implemenation began by studying the data given and it quickly became clear that a model function including cosine would make sense:

![image](https://user-images.githubusercontent.com/116219100/231102445-480b4510-7659-4146-a764-f623350de300.png)
*Figure 1: Plot of data points X and Y*

Once the model function was down and the number of parameters was determined, a conventional approach was taken to find the 'optimal' parameters that would yield minimum error through least-sqaures error. First, a helper function, **LSE** (least-squares error), which would calculate the least-squares error given 4 parameters as an array_like object, **c**, the given input data set, **x**, and the given output data set, **y**.

```
def LSEfit(c, x, y):
    E = np.sqrt(np.sum((c[0]*np.cos(c[1]*x)+c[2]*x+c[3]-y)**2)/n)
    return E
```

Then, optimization was applied using the SciPy library's optimize module which was imported as opt:

```
# set the initial guess for the parameters
c0 = np.array([3, 1*np.pi/4, 2/3, 32])

# perform optimization
res = opt.minimize(LSEfit, c0, args=(X, Y), method='Nelder-Mead')

# get the optimized parameters
c = res.x
```
As previously mentioned, the initial guess will vary the results since this is a nonlinear model with an unknown number of solutions. The initial guess used was the best I could find after trial and error. Optimization was done using the **LSEFit** function mentioned above and the Nelder-Mead method. I used this method because it is useful for optimizing functions that are not differentiable or whose derivatives are difficult to compute, however, do note that this method is a popular choice for optimization problems with a small number of variables, but can become inefficient in high-dimensional spaces or if the function being optimized is highly nonlinear or has multiple local minima.

After the optimized parameters were found, the minimum error (the furthest down the optimization algorithm could go) was found by passing these optimized parameters through the **LSEFit** function. The results will be discussed in **section IV.**.


In the next part of the project, the previously found parameters were studied more deeeply. Two of the four parameters were fixed at their optimal value while the other two were swept from 0 to 30 and a 2D loss (error) landscape was generated. Consider the case where A and B were the fixed parameters and C and D were swept across:

```
# FIX A B
# Initialize error grid
error_gridAB = np.zeros((len(C_range), len(D_range)))

# Loop through C and D ranges and compute error for each combination
for i, C in enumerate(C_range):
    for j, D in enumerate(D_range):
        # Compute error for fixed A and B and swept C and D
        error = compute_error(A_fixed, B_fixed, C, D, X, Y)
        # Store error in error grid
        error_gridAB[i, j] = error

# Generate x and y meshes from the ranges of C and D
C_mesh, D_mesh = np.meshgrid(C_range, D_range)

# Create a new figure and axis
fig, ax = plt.subplots()

# Plot the error grid as a pcolor map
pcm = ax.pcolormesh(C_mesh, D_mesh, error_gridAB, cmap='viridis', shading='auto')

# Add a colorbar to the plot
plt.colorbar(pcm).set_label('Error', fontweight='bold')
```

Note that color meshes are particularly useful in visualizing 2D arrays or grids (as can be seen in **section IV.**, hence why they were used. The error was stored in a error grid to be plotted against the meshes generated from the swept ranges. 

The above process was repeated for all six combinations of two fixed and two swept parameters.


Next, the first 20 data points were used as training data to determine the coeficients to fit a line, parabola and 19th degree polynomial to the data. Then, using these coeficients, the prediction accuracy of the model was tested against the last 10 data points using the least-squares error equation:

```
# Fit a parabola to the data
parabola_coeffs = np.polyfit(X[:20], Y[:20], deg=2)
parabola_predictions_train = np.polyval(parabola_coeffs, X[:20])
parabola_error_train = np.sqrt(np.sum((Y[:20] - parabola_predictions_train) ** 2)/20)

# Compute errors on test data
parabola_predictions_test = np.polyval(parabola_coeffs, X[-10:])
parabola_error_test = np.sqrt(np.sum((Y[-10:] - parabola_predictions_test) ** 2)/10)
```

Since all three of these fits are polynomials, the optimal coefficients were found using Numpy library's **polyfit** method. An initial guess was not required for this method since polynomials have a known number of solutions and therefore it iteratively minimizes the sum of squares of the residuals between the data and the polynomial fit until it determines the best-fit coefficients.

Finally, the same procedure was done except the training data became the first 10 and last 10 data points and then this model was fit to the 10 held out middle points (test data). 

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
