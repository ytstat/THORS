#' Predict the outcome of a set of new observations using the fitted thors object.
#'
#' @export
#' @param object the fitted npmc object using \code{thors}.
#' @param newx a set of new observations.
#' @param ... additional arguments.
#' @return the test cost.
#' @seealso \code{\link{thors}}, \code{\link{cost}}.

predict.thors <- function(object, newx=NULL, ...){
  pred.core(object, newx)
}

