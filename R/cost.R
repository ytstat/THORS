#' Calculate the test cost given the true test labels and the predicted ones.
#'
#' @export
#' @param y the true labels of the test data
#' @param ypred the predicted labels of the test data
#' @param beta the cost ratio of FP and FN cases. Default = 0.5.
#' @param A the cost of FN cases.
#' @param form the form of the test cost. If \code{"average"}, the average cost will be calculated. If \code{total}, the total cost on the test set will be calculated.
#' @return the predicted labels.
#' @seealso \code{\link{thors}}, \code{\link{cost}}.
#' @references Tian, Y., & Zhang, W. (2018). THORS: An Efficient Approach for Making Classifiers Cost-sensitive. arXiv preprint arXiv:1811.028

cost <- function(y, ypred, beta, A, form=c("average", "total")){
  form <- match.arg(form)
  A*(length(y[y==0 & ypred==1])*beta + length(y[y==1 & ypred==0]))/length(y)
}

