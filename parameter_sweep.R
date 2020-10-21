library(texteffect)

parameter_sweep <- function(X, Y, i_train, k_values, alphas, sigmas, n_iterations, save_to) {
  for (k in k_values) {
    writeLines(paste('Fitting models for k =', k))
    
    sibp_search <- sibp_param_search(X, Y, K=k, alphas=alphas, 
                                     sigmasq.ns=sigmas, 
                                     iters=n_iterations, train.ind=i_train)
    
    runs_ranked <- sibp_rank_runs(sibp_search, X, num.words=10)
    
    runs_ranked_max <- aggregate(runs_ranked, list(runs_ranked$alpha, runs_ranked$sigmasq.n), max)
    runs_ranked_max <- subset(runs_ranked_max, select=-c(Group.1, Group.2, iter))
    runs_ranked_max <- runs_ranked_max[order(runs_ranked_max$exclu, decreasing=TRUE),]
    
    alpha_1 <- runs_ranked_max[1, 'alpha']
    sigma_1 <- runs_ranked_max[1, 'sigmasq.n']
    temp <- runs_ranked[which(runs_ranked$alpha==alpha_1 & runs_ranked$sigmasq.n==sigma_1),]
    iteration_1 <- temp[which.max(temp$exclu),]$iter
    top_words_1 <- sibp_top_words(sibp_search[[toString(alpha_1)]][[toString(sigma_1)]][[iteration_1]], 
                                  colnames(X), num.words=5, verbose=FALSE)
    
    alpha_2 <- runs_ranked_max[2, 'alpha']
    sigma_2 <- runs_ranked_max[2, 'sigmasq.n']
    temp <- runs_ranked[which(runs_ranked$alpha==alpha_2 & runs_ranked$sigmasq.n==sigma_2),]
    iteration_2 <- temp[which.max(temp$exclu),]$iter
    top_words_2 <- sibp_top_words(sibp_search[[toString(alpha_2)]][[toString(sigma_2)]][[iteration_2]], 
                                  colnames(X), num.words=5, verbose=FALSE)
    
    write.table(runs_ranked, paste(save_to, '/k-', k, '-runs.txt', sep=''), sep='\t', row.names=FALSE)
    write.table(runs_ranked_max, paste(save_to, '/k-', k, '-runs-max.txt', sep=''), sep='\t', row.names=FALSE)
    write.table(top_words_1, paste(save_to, '/k-', k, '-words-1.txt', sep=''), sep='\t', row.names=FALSE)
    write.table(top_words_2, paste(save_to, '/k-', k, '-words-2.txt', sep=''), sep='\t', row.names=FALSE)
  }
}

# Read data
X <- read.csv('X_lemmatized.csv', header=TRUE)
Y <- read.csv('Y_lemmatized.csv', header=TRUE)

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

# Run parameter sweep
k_values <- c(1, 2, 3, 4, 5)
alphas <- c(4, 6, 8)
sigmas <- c(0.7, 1.0, 1.3)
n_iterations <- 5
parameter_sweep(X, Y$useful_discrete, i_train, k_values, alphas, sigmas, n_iterations, save_to='parameter-sweep-02')
