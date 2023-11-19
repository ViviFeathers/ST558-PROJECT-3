library(tidyverse)
library(rmarkdown)

edu <- c("1", "2", "3", "4", "5")
out <- c("Education_Level_1", "Education_Level_2", "Education_Level_3", "Education_Level_4", "Education_Level_5")
output_md <- paste0(out, ".md")
params = lapply(edu, FUN = function(x){list(Education = x)})
reports <- tibble(output_md, params)

getwd()
apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "Project_3_final.Rmd", output_file = x[[1]], params = x[[2]])
      })

