% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/thors.R
\name{thors}
\alias{thors}
\title{Construct a cost-sensitive classifier from training samples and validation samples.}
\usage{
thors(xtrain, ytrain, xvalid, yvalid, beta = 0.5, cv = FALSE,
  mrg.est = c("train", "valid"), cores = "auto", base = c("logistic",
  "svm", "knn", "randomforest", "lda", "nb", "tree"), ...)
}
\arguments{
\item{xtrain}{the n * p observation matrix for training the base classifier. n observations, p features.}

\item{ytrain}{n 2-class labels for training the base classifier.}

\item{xvalid}{the n * p observation matrix for finding the optimal threshold. IF \code{NULL}, cv must be assign an positive integer.}

\item{yvalid}{n 2-class labels for finding the optimal threshold. IF \code{NULL}, cv must be assign an positive integer.}

\item{beta}{the cost ratio of FP and FN cases. Default = 0.5.}

\item{cv}{the indicator for choosing the threshold based on the cross-validation. If \code{FALSE}, \code{xvalid} and \code{yvalid} can not be NUll. If cv is assigned a specific integer \code{a}, then THORS will be applied based on a-fold cross validation and the threshold is the average value of thresholds chosen in each fold. Default = \code{FALSE}.}

\item{mrg.est}{the estimation method for estimating the marginal probabilities of two classes. IF \code{train}, the marginal probabilities are estimated based on the training set. If \code{valid}, the marginal probabilities are estimated based on the validation set. Default = \code{"train"}.}

\item{cores}{how many cores to be used. If \code{"auto"}, all the cores detected will be used. Default = \code{"auto"}.}

\item{base}{the base classifier chosen. Default = \code{"logistic"}.
\itemize{
\item logistic: Multi-logistic regression. \link{multinom} in \code{nnet} package.
\item svm: Support Vector Machines. \code{\link[e1071]{svm}} in \code{e1071} package.
\item knn: k-Nearest Neighbor classifier. \code{\link[class]{knn}} in \code{knn} package.
\item randomforest: Random Forest. \code{\link[randomForest]{randomForest}} in \code{randomForest} package.
\item lda: Linear Discriminant Analysis. \code{\link[MASS]{lda}} in \code{MASS} package.
\item nb: Naive Bayes. \code{\link[e1071]{naiveBayes}} in \code{e1071} package.
\item tree: Decision Tree. \code{\link[tree]{tree}} in \code{tree} package.
}}

\item{...}{additional arguments.}
}
\value{
An object with S3 class thors.
 \item{fit}{a list including the fitting paramters, the threshold, the base classifier type, the method for estimating mariginal probabilities, and the indicator of cross-validation.}
 \item{threshold}{the threshold chosen for the binary classifcation}
 \item{base}{the base classifier type chosen.}
 \item{mrg.est}{the method for estimating the marginal probabilities.}
 \item{cv}{the indicator of cross-validation.}
}
\description{
Given the training data, the validation data, the cost matrix and the form of the base classifier,
\code{thors} can build a cost-sensitive classifierbased on the thresholding strategy.
}
\examples{
## calculate the threshold based on the validation data directly
library(datasets)
data(iris)
set.seed(1)
beta <- 0.5
D <- iris[1:100,][sample(100),]
D$Species <- as.numeric(D$Species)-1 # assign class "setosa" as class 0 and class "versicolor" as class 1.
xtrain <- D[1:40,]
ytrain <- D$Species[1:40]
xvalid <- D[40+1:40,]
yvalid <- D$Species[40+1:40]
xtest <- D[80+1:20,]
ytest <- D$Species[80+1:20]
fit <- thors(xtrain, ytrain, xvalid, yvalid,  beta = beta, cv = FALSE, base = "logistic")
ypred <- predict(fit, xtest)
cost(ytest, ypred, beta = beta, A=10, form = "average")

\dontrun{
## use the 5-fold cross-validation to calculate the threshold
library(datasets)
data(iris)
set.seed(1)
beta <- 0.5
D <- iris[1:100,][sample(100),]
D$Species <- as.numeric(D$Species)-1 # assign class "setosa" as class 0 and class "versicolor" as class 1.
xtrain <- D[1:80,]
ytrain <- D$Species[1:80]
xtest <- D[80+1:20,]
ytest <- D$Species[80+1:20]
fit <- thors(xtrain, ytrain, xvalid = NULL, yvalid = NULL,  beta = beta, cv = 5, base = "logistic")
ypred <- predict(fit, xtest)
cost(ytest, ypred, beta = beta, A=10, form = "average")
}
}
\references{
Tian, Y., & Zhang, W. (2018). THORS: An Efficient Approach for Making Classifiers Cost-sensitive. arXiv preprint arXiv:1811.028
}
\seealso{
\code{\link{predict.thors}}, \code{\link{cost}}.
}
