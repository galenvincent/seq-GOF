library(tidyverse)
library(ggplot2)
library(viridis)
library(data.table)

# Read files output from experiments
read_sims <- function (N = '500', Q = '200', L = '16', ntrain = '300', mtrain = '1', 
                       neval = '300', meval = '1', alpha = '0.6', delta = '0.6',
                       root = 'data/') {
  
  base = paste0('reps_',N, '-Q_',Q, '-L_',L, '-alpha_',alpha, '-delta_',delta, 
                '-ntrain_',ntrain, '-mtrain',mtrain, 
                '-neval_',neval, '-meval_',meval)
  
  local = paste0(root, base, '-local.csv')
  global = paste0(root, base, '-global.csv')
  
  local.data <- fread(local)
  global.data <- fread(global)

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