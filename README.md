# Regression Microservice
This is the second individual project of IDS721 in Duke University.

## Project introduction 
This project is to create a kubernetes based microservice. The microservice is built by Flask and deployed using Google Kubernetes Engine. The container is pushed to the container registry and deployed in kubernetes. 

## Function demonstration
### Home page
Go to the [deployed website](http://34.122.55.167:8080/) and you can see the title "Linear Regression Model Home". 

### Plot demo 
Add /show after the home url, the page of regression plot of randomly generated 2-dimensional data will be displayed. Users can choose the regression type (Linear Regression, Lasso, Ridge) and CV folds. After pressing plot, the user can see a plot of random generated data using the chosed fitting parameters.
