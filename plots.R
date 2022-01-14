library(tidyverse)
library(ggplot2)
source('plots-helpers.R')

n1_seq <- c(15, 25, 35, 50)
mu0 <- '1.0'
N <- '500'
n0_seq <- c(25, 37, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000)

results <- list()
results_local <- list()
for (jj in seq_along(n1_seq)) {
  n1 = as.character(n1_seq[jj])
  results[[n1]] <- list()
  results_local[[n1]] <- list()
  
  for (ii in seq_along(n0_seq)) {
    print(paste0("n1: ", as.character(n1_seq[jj]), ", n0: ", as.character(n0_seq[ii])))
    n0 = as.character(n0_seq[ii])
    upload = read_sims(N = N, n0 = n0, n1 = n1, mu0 = mu0)
    
    results[[n1]][[n0]] = data.frame(
      pval = sort(upload$global$V1),
      n0 = n0_seq[ii],
      n1 = n1_seq[jj],
      i = 1:nrow(upload$global)
    )
    
    temp = upload$local
    temp$n0 = n0_seq[ii]
    temp$n1 = n1_seq[jj]
    results_local[[n1]][[n0]] = temp
  }
  
  results[[n1]] = reduce(results[[n1]], bind_rows)
  results_local[[n1]] = reduce(results_local[[n1]], bind_rows)
}

globs <- reduce(results, bind_rows)
locs <- reduce(results_local, bind_rows)

globs %>% 
  group_by(n0, n1) %>%
  summarize(fp=sum(pval<0.05), n=n(), rate=mean(pval<0.05)) %>%
  ggplot(aes(x = n0, y = rate, col = as.factor(n1))) +
  geom_point() + 
  geom_line() +
  labs(y = 'power', col = 'n1') +
  scale_y_continuous(breaks = seq(0.5, 1.0, by = 0.1)) +
  theme_bw()


### Adjusted Cross Entropy Plots
n1 <- '50'
mu0 <- '1.0'
N <- '500'
n0_seq <- c(25, 37, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000)

results <- list()
results_local <- list()

for (ii in seq_along(n0_seq)) {
  print(paste0("n1: ", n1, ", n0: ", as.character(n0_seq[ii])))
  n0 = as.character(n0_seq[ii])
  upload = read_sims(N = N, n0 = n0, n1 = n1, mu0 = mu0,
                     root = 'data/normal-experiments-with-cross-entropy/',
                     ce = TRUE)
  
  temp = upload$global
  temp$n0 = n0_seq[ii]
  temp$n1 = as.numeric(n1)
  temp$i = 1:nrow(upload$global)
  results[[n0]] = temp
  
  temp = upload$local
  temp$n0 = n0_seq[ii]
  temp$n1 = as.numeric(n1)
  results_local[[n0]] = temp
}

globs <- reduce(results, bind_rows)
locs <- reduce(results_local, bind_rows)



globs %>% 
  group_by(n0) %>%
  summarize(fp=sum(pval<0.05), n=n(), rate=mean(pval<0.05), mean_ce = mean(ce), mean_adj_ce = mean(adj_ce)) %>%
  pivot_longer(cols = c('rate', 'mean_ce', 'mean_adj_ce'), names_to = 'metric') %>%
  ggplot(aes(x = n0, y = value, col = metric)) +
  geom_point() + 
  geom_line() +
  labs(y = '') +
  theme_bw() +
  scale_color_discrete(name = '', labels = c('Adj. Cross Entropy', 'Cross Entropy', 'Power')) 

meds <- globs %>% 
  filter(n0 %in% c(200, 500, 700, 1000)) %>%
  group_by(n0) %>%
  summarize(med_ce = median(ce),
            med_adj_ce = median(adj_ce))
  
globs %>% 
  filter(n0 %in% c(200, 500, 700, 1000)) %>%
  ggplot(aes(x = adj_ce)) +
  geom_histogram(bins = 20) +
  geom_vline(data = meds, aes(xintercept = med_adj_ce), col = 'red') +
  facet_grid(. ~ n0)

ce_loss <- function(y_real, y_prob, eps = 1e-7) {
  clip_it <- function(y){
    pmax(eps, pmin(1-eps, y))
  }
  y_prob <- clip_it(y_prob)
  ce_vec <- -1*(y_real*log(y_prob) + (1 - y_real)*log(1 - y_prob))
  return(mean(ce_vec))
}

brier_loss <- function(y_real, y_prob) {
  mean((y_real - y_prob)^2)
}

locs %>%
  mutate(prob_real = dnorm(x)/(dnorm(x) + dnorm(x, mean = 1)),
         pi_hat_eval = 0.5,
         pi_hat_test_1 = n1/(n0 + n1),
         pi_hat_test_0 = n0/(n0 + n1),
         prob_est_adj = ((pi_hat_eval/pi_hat_test_1)*prob_est)/((pi_hat_eval/pi_hat_test_1)*prob_est + (pi_hat_eval/pi_hat_test_0)*(1 - prob_est))) %>%
  #filter(prob_est > 0.05, prob_est < 0.95) %>%
  group_by(n0) %>%
  summarize(ce = ce_loss(Y, prob_est, eps = 1e-15),
            ce_adj = ce_loss(Y, prob_est_adj, eps = 1e-15),
            ce_best = ce_loss(Y, prob_real, eps = 1e-15),
            brier = brier_loss(Y, prob_est),
            brier_adj = brier_loss(Y, prob_est_adj),
            brier_best = brier_loss(Y, prob_real)) %>%
  pivot_longer(cols = c('ce', 'ce_adj')) %>%
  ggplot(aes(x = n0, y = value, col = name)) + geom_point() + geom_line() + geom_hline(yintercept = 0.581726, col = 'black', linetype = 2)

### PART OF THE PROBLEM WAS JUST PROB ESTIMATES CLOSE TO ZERO AND ONE!! Brier
# score kind of fixes this.. but there now isn't a huge asymptote in the loss at all!

   
  

