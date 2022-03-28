library(tidyverse)
library(ggplot2)
source('plots-helpers.R')

#
#### OLD STUFF ####

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
                     root = 'data/012022_normal/',
                     type = 'new')
  
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
  summarize(fp=sum(pval<0.05), n=n(), rate=mean(pval<0.05), 
            mean_ce = mean(ce), sd_ce = sd(ce),
            mean_adj_ce = mean(adj_ce), sd_adj_ce = sd(adj_ce),
            mean_bs = mean(bs), sd_bs = sd(bs),
            mean_adj_bs = mean(adj_bs), sd_adj_bs = sd(adj_bs),
            mean_mse = mean(mse), sd_mse = sd(mse),
            mean_mae = mean(mae), sd_mae = sd(mae)) %>%
  pivot_longer(cols = c('rate', 'mean_ce', 'mean_adj_ce', 'mean_bs', 'mean_adj_bs', 'mean_mse', 'mean_mae'), 
               names_to = 'metric') %>%
  ggplot(aes(x = n0, y = value, col = metric)) +
  geom_point() + 
  geom_line() +
  labs(y = '') +
  theme_bw() 
  #scale_color_discrete(name = '', labels = c('Adj. Cross Entropy', 'Cross Entropy', 'Power')) 

globs %>% 
  group_by(n0) %>%
  summarize(fp=sum(pval<0.05), n=n(), rate=mean(pval<0.05), 
            mean_ce = mean(ce), sd_ce = sd(ce),
            mean_adj_ce = mean(adj_ce), sd_adj_ce = sd(adj_ce),
            mean_bs = mean(bs), sd_bs = sd(bs),
            mean_adj_bs = mean(adj_bs), sd_adj_bs = sd(adj_bs),
            mean_mse = mean(mse), sd_mse = sd(mse),
            mean_mae = mean(mae), sd_mae = sd(mae)) %>%
  pivot_longer(cols = c('mean_adj_bs', 'mean_mse', 'mean_mae'), 
               names_to = 'metric') %>%
  mutate(std = case_when(
    metric == 'mean_adj_bs' ~ sd_adj_bs,
    metric == 'mean_mse' ~ sd_mse,
    metric == 'mean_mae' ~ sd_mae
  )) %>% 
  ggplot(aes(x = n0, y = value, col = metric)) +
  geom_point() + 
  geom_line() +
  geom_ribbon(aes(ymin=value - std, ymax=value + std), linetype=2, alpha=0.1) +
  labs(y = '') +
  theme_bw() 

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

median_ae <- function(y_prob_real, y_prob) {
  median(abs(y_prob_real - y_prob))
}

locs %>%
  #filter(prob_est > 0.05, prob_est < 0.95) %>%
  group_by(n0) %>%
  summarize(med_ae = median_ae(adjusted_prob_est, true_prob),
            med_ae_sd = sd(abs(adjusted_prob_est - true_prob))) %>%
  ggplot(aes(x = n0, y = med_ae)) + geom_point() + geom_line() +
  geom_ribbon(aes(ymin=med_ae - med_ae_sd, ymax=med_ae + med_ae_sd), linetype=2, alpha=0.1)



### PART OF THE PROBLEM WAS JUST PROB ESTIMATES CLOSE TO ZERO AND ONE!! Brier
# score kind of fixes this.. but there now isn't a huge asymptote in the loss at all!

   

# LPDs
locs %>%
  filter(n0 == 1000) %>%
  ggplot(aes(x = x, y = LPD)) +
  geom_point() + geom_smooth(method = 'gam')
  

#### MC Example
n1_seq <- c(35, 75)
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
    upload = read_sims_mc(N = N, n0 = n0, n1 = n1, 
                          root = 'data/012022_mc/')
    
    temp = upload$global
    temp$n0 = n0_seq[ii]
    temp$n1 = as.numeric(n1)
    temp$i = 1:nrow(upload$global)
    results[[n1]][[n0]] = temp
    
    temp = upload$local
    temp$n0 = n0_seq[ii]
    temp$n1 = as.numeric(n1)
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

#### NEW STUFF 03/21/21 ####

# Validity study
mtrain_seq <- c(1, 2, 3, 5, 10)
alpha_seq <- c('0.0')
L <- '16'
N <- '500'
ntrain <- '300'
neval <- '300'
meval <- '1'

results <- list()
#results_local <- list()
for (jj in seq_along(alpha_seq)) {
  alpha = alpha_seq[jj]
  results[[alpha]] <- list()
  #results_local[[alpha]] <- list()
  
  for (ii in seq_along(mtrain_seq)) {
    print(paste0("alpha: ",alpha_seq[jj], ", mtrain: ", as.character(mtrain_seq[ii])))
    mtrain = as.character(mtrain_seq[ii])
    upload = read_sims(N = N, L = L, ntrain = ntrain, mtrain = mtrain, 
                       neval = neval, meval = meval, alpha = alpha, delta = alpha,
                       root = 'data/032322_validity/')
    
    temp = upload$global
    temp$alpha = as.numeric(alpha)
    temp$mtrain = mtrain_seq[ii]
    temp$i = 1:nrow(upload$global)
    results[[alpha]][[mtrain]] = temp
    
    #temp = upload$local
    #temp$alpha = as.numeric(alpha)
    #temp$mtrain = mtrain_seq[ii]
    #results_local[[alpha]][[mtrain]] = temp
  }
  
  results[[alpha]] = reduce(results[[alpha]], bind_rows)
  #results_local[[alpha]] = reduce(results_local[[alpha]], bind_rows)
}

globs <- reduce(results, bind_rows)
#locs <- reduce(results_local, bind_rows)

# pval histograms (should be uniform)
globs %>% 
  ggplot(aes(x = pval)) + geom_histogram(bins = 20) +
  facet_wrap(alpha ~ mtrain, nrow = 3)

# One example to show in meeting
globs %>%
  filter(alpha == 0.7, mtrain == 3) %>%
  ggplot(aes(x = pval)) + geom_histogram(bins = 20)


# TODO: Look into this... why are so many of the p-values small? Get out the 
# algorithm and think about where in the process stuff could be going wrong.

# Only thing I can think of: there's something different about your 
# generate_sample_custom function than the generate_sample function from 
# the original AR package.
#
# To address above: Nope... not that. I changed so that they are exactly the same and still
# get these funky results. Try what they suggested before and run with random model (alpha = delta = 0)?