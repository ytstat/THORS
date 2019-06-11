#' Construct a cost-sensitive classifier from training samples and validation samples.
#'
#' Given the training data, the validation data, the cost matrix and the form of the base classifier,
#' \code{thors} can build a cost-sensitive classifierbased on the thresholding strategy.
#' @export
#' @importFrom e1071 svm
#' @importFrom e1071 naiveBayes
#' @importFrom e1071 permutations
#' @importFrom class knn
#' @importFrom naivebayes naive_bayes
#' @importFrom glmnet cv.glmnet
#' @importFrom MASS lda
#' @importFrom randomForest randomForest
#' @importFrom ada ada
#' @importFrom parallel detectCores
#' @importFrom parallel makeCluster
#' @importFrom parallel stopCluster
#' @importFrom doParallel registerDoParallel
#' @import foreach
#' @import doParallel
#' @importFrom graphics plot
#' @importFrom stats glm
#' @importFrom stats predict
#' @importFrom stats pbinom
#' @importFrom stats rnorm
#' @importFrom stats sd
#' @importFrom stats rbinom
#' @importFrom tree tree
#' @importFrom graphics polygon
#' @importFrom nnet multinom
#' @importFrom caret createFolds
#' @param xtrain the n * p observation matrix for training the base classifier. n observations, p features.
#' @param ytrain n 2-class labels for training the base classifier.
#' @param xvalid the n * p observation matrix for finding the optimal threshold. IF \code{NULL}, cv must be assign an positive integer.
#' @param yvalid n 2-class labels for finding the optimal threshold. IF \code{NULL}, cv must be assign an positive integer.
#' @param beta the cost ratio of FP and FN cases. Default = 0.5.
#' @param cv the indicator for choosing the threshold based on the cross-validation. If \code{FALSE}, \code{xvalid} and \code{yvalid} can not be NUll. If cv is assigned a specific integer \code{a}, then THORS will be applied based on a-fold cross validation and the threshold is the average value of thresholds chosen in each fold. Default = \code{FALSE}.
#' @param mrg.est the estimation method for estimating the marginal probabilities of two classes. IF \code{train}, the marginal probabilities are estimated based on the training set. If \code{valid}, the marginal probabilities are estimated based on the validation set. Default = \code{"train"}.
#' @param cores how many cores to be used. If \code{"auto"}, all the cores detected will be used. Default = \code{"auto"}.
#' @param base the base classifier chosen. Default = \code{"logistic"}.
#' \itemize{
#' \item logistic: Multi-logistic regression. \link{multinom} in \code{nnet} package.
#' \item svm: Support Vector Machines. \code{\link[e1071]{svm}} in \code{e1071} package.
#' \item knn: k-Nearest Neighbor classifier. \code{\link[class]{knn}} in \code{knn} package.
#' \item randomforest: Random Forest. \code{\link[randomForest]{randomForest}} in \code{randomForest} package.
#' \item lda: Linear Discriminant Analysis. \code{\link[MASS]{lda}} in \code{MASS} package.
#' \item nb: Naive Bayes. \code{\link[e1071]{naiveBayes}} in \code{e1071} package.
#' \item tree: Decision Tree. \code{\link[tree]{tree}} in \code{tree} package.
#' }
#' @param ... additional arguments.
#' @return An object with S3 class thors.
#'  \item{fit}{a list including the fitting paramters, the threshold, the base classifier type, the method for estimating mariginal probabilities, and the indicator of cross-validation.}
#'  \item{threshold}{the threshold chosen for the binary classifcation}
#'  \item{base}{the base classifier type chosen.}
#'  \item{mrg.est}{the method for estimating the marginal probabilities.}
#'  \item{cv}{the indicator of cross-validation.}
#' @references Tian, Y., & Zhang, W. (2018). THORS: An Efficient Approach for Making Classifiers Cost-sensitive. arXiv preprint arXiv:1811.028
#' @seealso \code{\link{predict.thors}}, \code{\link{cost}}.
#' @examples
#' ## calculate the threshold based on the validation data directly
#' library(datasets)
#' data(iris)
#' set.seed(1)
#' beta <- 0.5
#' D <- iris[1:100,][sample(100),]
#' D$Species <- as.numeric(D$Species)-1 # assign class "setosa" as class 0 and class "versicolor" as class 1.
#' xtrain <- D[1:40,]
#' ytrain <- D$Species[1:40]
#' xvalid <- D[40+1:40,]
#' yvalid <- D$Species[40+1:40]
#' xtest <- D[80+1:20,]
#' ytest <- D$Species[80+1:20]
#' fit <- thors(xtrain, ytrain, xvalid, yvalid,  beta = beta, cv = FALSE, base = "logistic")
#' ypred <- predict(fit, xtest)
#' cost(ytest, ypred, beta = beta, A=10, form = "average")
#'
#' \dontrun{
#' ## use the 5-fold cross-validation to calculate the threshold
#' library(datasets)
#' data(iris)
#' set.seed(0)
#' beta <- 0.5
#' D <- iris[1:100,][sample(100),]
#' D$Species <- as.numeric(D$Species)-1 # assign class "setosa" as class 0 and class "versicolor" as class 1.
#' xtrain <- D[1:80,]
#' ytrain <- D$Species[1:80]
#' xtest <- D[80+1:20,]
#' ytest <- D$Species[80+1:20]
#' fit <- thors(xtrain, ytrain, xvalid = NULL, yvalid = NULL,  beta = beta, cv = 5, base = "logistic")
#' ypred <- predict(fit, xtest)
#' cost(ytest, ypred, beta = beta, A=10, form = "average")
#' }

thors <- function(xtrain, ytrain, xvalid, yvalid,  beta = 0.5,
                  cv = FALSE, mrg.est = c("train", "valid"), cores = "auto",
                  base = c("logistic", "svm", "knn", "randomforest", "lda", "nb", "tree"), ...){
  base <- match.arg(base)
  mrg.est <- match.arg(mrg.est)
  if(cores == "auto"){
    num.cores <- detectCores()
  }else{
    num.cores <- cores
  }
  cl <- makeCluster(num.cores)
  registerDoParallel(cl)
  if(cv){
    folds <- createFolds(1:length(ytrain), k = cv)
    Dtrain <- data.frame(xtrain, class=as.factor(ytrain))
    t.list <- foreach(k = 1:cv, .combine = "c",
                      .packages = c("MASS", "tree", "naivebayes", "caret", "e1071",
                                    "randomForest", "nnet")) %dopar% {
      if(base=="logistic"){
        fit <- suppressWarnings(glm(class~., data=Dtrain[-folds[[k]],], family = "binomial"))
        xvalid <- xtrain[folds[[k]],]
        yvalid <- ytrain[folds[[k]]]
        score.valid <- predict(fit,xvalid,type="response")
      }else if(base=="lda"){
        fit <- lda(class~., data=Dtrain[-folds[[k]],])
        xvalid <- xtrain[folds[[k]],]
        yvalid <- ytrain[folds[[k]]]
        score.valid <- predict(fit, xvalid)$posterior[,2]
      }else if(base=="svm"){
        fit <- svm(class~., data=Dtrain[-folds[[k]],], probability=TRUE, kernel = "linear")
        xvalid <- xtrain[folds[[k]],]
        yvalid <- ytrain[folds[[k]]]
        score.valid <- attr(predict(fit, xvalid, probability = TRUE), "probabilities")[,2]
      }else if(base == "randomforest"){
        fit <- randomForest(class~., data=Dtrain[-folds[[k]],], ...)
        xvalid <- xtrain[folds[[k]],]
        yvalid <- ytrain[folds[[k]]]
        score.valid <- predict(fit, xvalid, type="prob")[,2]
      }else if(base == "nb"){
        fit <- naive_bayes(class~., data=Dtrain[-folds[[k]],], usekernel = FALSE)
        xvalid <- xtrain[folds[[k]],]
        yvalid <- ytrain[folds[[k]]]
        score.valid <- predict(fit, xvalid, type="prob")[,2]
      }else if(base == "knn"){
        fit <- knn3(class~., data=Dtrain[-folds[[k]],], k = 10)
        xvalid <- xtrain[folds[[k]],]
        yvalid <- ytrain[folds[[k]]]
        score.valid <- predict(fit, xvalid, type="prob")[,2]
      }else if(base == "tree"){
        fit <- tree(class~., data=Dtrain[-folds[[k]],])
        xvalid <- xtrain[folds[[k]],]
        yvalid <- ytrain[folds[[k]]]
        score.valid <- predict(fit, xvalid, type="vector")[,2]
      }

      if(mrg.est=="train"){
        xtrain.cv <- xtrain[-folds[[k]],]
        ytrain.cv <- ytrain[-folds[[k]]]
        p0 <- sum(ytrain.cv == 0)/length(ytrain.cv)
        p1 <- 1-p0
      }else{
        p0 <- sum(yvalid == 0)/length(yvalid)
        p1 <- 1-p0
      }
      n0 <- sum(yvalid == 0)
      n1 <- sum(yvalid == 1)

      s <- score.valid[order(score.valid)]
      s0 <- score.valid[yvalid==0]
      s1 <- score.valid[yvalid==1]
      s0 <- s0[order(s0)]
      s1 <- s1[order(s1)]


      C <- sapply(1:length(s), function(i){
        p1*length(s1[s1<=s[i]])/n1 - beta*p0*length(s0[s0<=s[i]])/n0
      })
      s[which.min(C)]
    }

    if(base=="logistic"){
      fit <- suppressWarnings(glm(class~., data=Dtrain, family = "binomial"))
    }else if(base=="lda"){
      fit <- lda(class~., data=Dtrain)
    }else if(base=="svm"){
      fit <- svm(class~.,data=Dtrain,probability=TRUE)
    }else if(base == "randomforest"){
      fit <- randomForest(class~., data=Dtrain, ...)
    }else if(base == "nb"){
      fit <- naive_bayes(class~., data=Dtrain, usekernel = FALSE)
    }else if(base == "knn"){
      fit <- knn3(class~., data=Dtrain, k = 10)
    }else if(base == "tree"){
      fit <- tree(class~.,data=Dtrain)
    }
    t <- mean(t.list)
  }else{
    if(mrg.est=="train"){
      p0 <- sum(ytrain == 0)/length(ytrain)
      p1 <- 1-p0
    }else{
      p0 <- sum(yvalid == 0)/length(yvalid)
      p1 <- 1-p0
    }
    n0 <- sum(yvalid == 0)
    n1 <- sum(yvalid == 1)
    Dtrain <- data.frame(xtrain,class=factor(ytrain))
    if(base == "logistic") {
      fit <- suppressWarnings(glm(class~., data=Dtrain, family = "binomial"))
      score.valid <- predict(fit,xvalid,type="response")
    }else if(base == "lda"){
      fit <- lda(class~., data=Dtrain)
      score.valid <- predict(fit, xvalid)$posterior[,2]
    }else if(base=="svm"){
      fit <- svm(class~.,data=Dtrain, probability=TRUE, kernel = "linear")
      score.valid <- attr(predict(fit, xvalid, probability = TRUE), "probabilities")[,2]
    }else if(base == "randomforest"){
      fit <- randomForest(class~., data=Dtrain)
      score.valid <- predict(fit, xvalid, type="prob")[,2]
    }else if(base == "nb"){
      fit <- naive_bayes(class~., data=Dtrain, usekernel = FALSE)
      score.valid <- predict(fit, xvalid, type="prob")[,2]
    }else if(base == "knn"){
      fit <- knn3(class~., data=Dtrain, k = 10)
      score.valid <- predict(fit, xvalid, type="prob")[,2]
    }else if(base == "tree"){
      fit <- tree(class~.,data=Dtrain)
      score.valid <- predict(fit, xvalid, type="vector")[,2]
    }


    s <- score.valid[order(score.valid)]
    s0 <- score.valid[yvalid==0]
    s1 <- score.valid[yvalid==1]
    s0 <- s0[order(s0)]
    s1 <- s1[order(s1)]

    C <- foreach(i = 1:length(s), .combine = "c") %dopar% {
      p1*length(s1[s1<=s[i]])/n1 - beta*p0*length(s0[s0<=s[i]])/n0
    }

    t <- s[which.min(C)]
  }

  object <- list(fits=fit, threshold=t, base=base, mrg.est=mrg.est, cv=cv)
  class(object) <- "thors"

  stopCluster(cl)

  object
}
