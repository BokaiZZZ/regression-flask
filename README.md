# Regression Microservice
This is the second individual project of IDS721 in Duke University.

## Project introduction 
This project is to create a kubernetes based microservice. The microservice is built by Flask and deployed using Google Kubernetes Engine. The container is pushed to the container registry and deployed in kubernetes. The continuous delivery is achieved using Cloud Build in GCP. 

## Function demonstration
### Home page
Go to the [deployed website](http://34.122.55.167:8080/) and you can see the title "Welcome to Linear Regression Model Home". 

### Plot demo 
Add /show after the home url, the page of regression plot of randomly generated 2-dimensional data will be displayed as figure below. Users can choose the regression type (Linear Regression, Lasso, Ridge) and CV folds. After pressing plot, the user can see a plot of random generated data using the chosed fitting parameters.

![image](https://user-images.githubusercontent.com/97444802/154597994-e6fb32e3-f1e4-4c2b-afa6-2d80c51c8cfb.png)

### Model build interface
Apart from the two pages above, adding /build_model after home url is an interface to obtain model coefficeints of the post data X, y and fitting model parameters. The detailed request format can be found in app.py. All input parameters can be found in model.py, which will also be expanded in the following section.    

## Function introduction 
The microservice is mainly to automatically tune blackbox regression models. The inputs are listed below. 

- A learning algorithm, i.e., a function that takes as input a matrix X ∈ R<sup>n×p</sup> and a vector of responses Y ∈ R<sup>n</sup> and returns a function that maps inputs to outputs, i.e, maps R<sup>p</sup> into R. The algorithm is one of the three object function: ordinary least squares Linear Regression, Lasso and Ridge. 
- Training data X ∈ R<sup>n×p</sup> and Y ∈ R<sup>n</sup>
- A regularization method that belongs to the set {Dropout, NoiseAddition, Robust}
- An positive integer M indicating the number of Monte Carlo replicates to be used if the method specified is Dropout or NoiseAddition
- A vector c of column bounds to be used if the method specified is Robust
- A vector alpha that multiplies the L1 or L2 term.
- A positive integer K indicating the number of CV-folds to be used to tune the amount of regularization, e.g., K = 5 indicates five-fold CV
- A criterion to be used to evaluate the method that belongs to the set {MSE, MAD} where MSE encodes mean square error and MAD encodes mean absolute deviation.
