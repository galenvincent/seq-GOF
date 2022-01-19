library(tidyverse)
library(ggplot2)
library(viridis)

# Read files output from experiments
read_sims <- function (N = '500', B = '200', L = '1', n1 = '15', n0 = '15', 
                       m1 = '25', m0 = '25', mu1 = '0.0', mu0 = '2.0', 
                       sigma1 = '1.0', sigma0 = '1.0',
                       root = 'data/normal-experiments/',
                       ce = FALSE) {
  
  base = paste0('reps_',N, '-B_',B, '-L_',L, '-n1_',n1, '-n0_',n0, '-m1_',m1, '-m0_',m0, 
                '-mu1_',mu1, '-mu0_',mu0, '-sigma1_',sigma1, '-sigma0_',sigma0)
  
  if (ce == FALSE) {
    local = paste0(root, base, '-local.csv')
    global = paste0(root, base, '-global.csv')
    
    local.data <- read.csv(local)
    global.data <- read.csv(global, header=FALSE)
    
  } else {
    local = paste0(root, base, '-local', '-adj_ce.csv')
    global = paste0(root, base, '-global', '-adj_ce.csv')
    
    local.data <- read.csv(local)
    global.data <- read.csv(global)
  }
  
  return(list(local = local.data, global = global.data))
}

# Summarize rejection fraction (global) by n0 group
rejection_summary <- function(global_df, n0tf = TRUE){
  if (n0tf == TRUE) {
    global_df %>% 
      group_by(n0) %>%
      summarize(fp=sum(pval<0.05), n=n(), rate=mean(pval<0.05))
  } else {
    global_df %>%
      summarize(fp=sum(pval<0.05), n=n(), rate=mean(pval<0.05))
  }
}