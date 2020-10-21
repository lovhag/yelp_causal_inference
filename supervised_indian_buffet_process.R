setwd("/Users/antmats/Desktop/causality-and-causal-inference/project")

# Load libraries
library(texteffect)
library(cdfquantreg)

# Read data
X <- read.csv('X_lemmatized.csv', header=TRUE)
Y <- read.csv('Y_lemmatized.csv', header=TRUE)
print(names(X[, colSums(X != 0) == 0]))
# X <- X[, colSums(X != 0) > 0]
# X <- subset(X, select=-(buffett))

# Extract categorized outcomes
useful_discrete <- rep(0, nrow(Y))
for (i in seq(1, nrow(Y))) {
  if (Y$useful[i] == 0) {
    useful_discrete[i] <- 0
  } else if (Y$useful[i] <= 5) {
    useful_discrete[i] <- 1
  } else if (Y$useful[i] <= 10) {
    useful_discrete[i] <- 2
  } else {
    useful_discrete[i] <- 3
  }
}
Y <- cbind(Y, useful_discrete)

# Sample training indexes
i_train <- sample(1:nrow(X), size=0.3*nrow(X), replace=FALSE)

# Fit a sIBP using a single set of parameters
K <- 3
s_fit <- sibp(X, Y$useful_discrete, K=K, alpha=6, sigmasq.n=1.0, a=0.1, b=0.1, sigmasq.A=5, train.ind=i_train, silent=TRUE)
sibp_exclusivity(sibp.fit=s_fit, X=X, num.words=10)
sibp_top_words(s_fit, colnames(X), num.words=5, verbose=FALSE)

# Infer treatments in the test set
nu_test <- infer_Z(s_fit, X, newX=FALSE)

# Save probabilities for each row having the column treatment to file
treatments <- as.data.frame(matrix(nrow=nrow(X), ncol=K+1))
treatments[,1] = rep(0, nrow(X))
treatments[s_fit$test.ind, 1] = 1
for (i in seq(1, K)) {
  treatments[s_fit$test.ind, i+1] = nu_test[,i]
  treatments[s_fit$train.ind, i+1] = s_fit$nu[,i]
}
colnames(treatments) <- c('is_test_data', paste('Z', seq(1, K), sep=''))
write.csv(treatments, 'sibp_treatments.csv', row.names=FALSE)


### Parameter sweep ###

# Fit multiple sIBPs using different sets of parameters
sibp_search <- sibp_param_search(X, Y$useful_discrete, K=2, alphas=c(6, 8, 10), sigmasq.ns=c(1.0), iters=3, train.ind=i_train)

# Get metric for evaluating most promising parameter configurations
sibp_rank_runs(sibp_search, X, 10)

# Qualitatively look at the top candidate
sibp_top_words(sibp_search[['6']][['1']][[1]], colnames(X), 10, verbose=TRUE)

# Select the most interesting treatments to investigate
sibp_fit <- sibp_search[['4']][['1']][[1]]

# Estimate the AMCE using the test set
amce <- sibp_amce(sibp_fit, X, Y$useful_discrete)

# Plot 95 % confidence intervals for the AMCE of each treatment
sibp_amce_plot(amce)


### Example ###

# Load the Wikipedia biography data
data(BioSample)

# Divide into training and test sets
x <-  BioSample[,-1]
y <-  BioSample[,1]
set.seed(1)
train.ind <- sample(1:nrow(x), size=0.5*nrow(x), replace=FALSE)

# Fit a sIBP using different parameters
sibp.search <- sibp_param_search(x, y, K=2, alphas=c(2, 4, 6, 8), sigmasq.ns=c(0.8, 1), iters=5, train.ind=train.ind)

# Get metric for evaluating most promising parameter configurations
s <- sibp_rank_runs(sibp.search, x, 10)

# Qualitatively look at the top candidates
sibp_top_words(sibp.search[['4']][['0.8']][[1]], colnames(x), 10, verbose=TRUE)
sibp_top_words(sibp.search[['4']][['1']][[1]], colnames(x), 10, verbose=TRUE)

# Select the most interesting treatments to investigate
sibp.fit <- sibp.search[['4']][['0.8']][[1]]

# Estimate the AMCE using the test set
amce <- sibp_amce(sibp.fit, x, y)

# Plot 95 % confidence intervals for the AMCE of each treatment
sibp_amce_plot(amce)
