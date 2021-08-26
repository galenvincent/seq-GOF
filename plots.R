library(tidyverse)
library(ggplot2)
source('plots-helpers.R')

n0_seq <- c(15, 20, 25, 30, 40, 50, 75, 100)
N <- 300
results <- list()
results_local <- list()
for (ii in seq_along(n0_seq)) {
  print(paste0("loading ", toString(ii), "/", toString(length(n0_seq)), " ..."))
  n0 = as.character(n0_seq[ii])
  upload = read_sims(n0 = n0)
  
  results[[n0]] = data.frame(
    pval = sort(upload$global$V1),
    n0 = n0_seq[ii],
    i = 1:nrow(upload$global)
  )
  
  temp = upload$local
  temp$n0 = n0_seq[ii]
  results_local[[n0]] = temp
}