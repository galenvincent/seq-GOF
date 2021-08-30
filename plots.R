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

   
  

