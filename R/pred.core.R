pred.core <- function(object, newx=NULL, ...){
  if(object$base=="logistic"){
    score.test <- predict(object$fits, newx, type="response")
  }else if(object$base == "lda"){
    score.test <- predict(object$fits, newx)$posterior[,2]
  }else if(object$base == "svm"){
    score.test <- attr(predict(object$fits, newx, probability = TRUE), "probabilities")[,2]
  }else if(object$base == "randomforest"){
    score.test <- predict(object$fits, newx, type = "prob")[,2]
  }else if(object$base == "nb"){
    score.test <- predict(object$fits, newx, type = "prob")[,2]
  }else if(object$base == "knn"){
    score.test <- predict(object$fits, newx, type = "prob")[,2]
  }else if(object$base == "tree"){
    score.test <- predict(object$fits, newx, type = "vector")[,2]
  }
  ypred <- rep(0, length(score.test))
  ypred[score.test>object$threshold] <- 1
  ypred
}

