# ST558-PROJECT-3

## Introduction

Diabetes are one of the most prevalent chronic disease that are facing people in the United States as well as in the world. 

Although there are multiple treatment regime for diabetic individual to effectively control their blood sugar level, there is no cure for the disease. To make things worse, about 25% of diabetic individuals were not diagnosed. So there are urgent unmet medical needs to predict potential pre-diabetic/diabetic individuals from their life-style related information before they get diagnosed at a hospital. Making meaningful insight by exploring data analysis and making prediction using large cohort comprehensive health data set could help provide solution to the need mentioned above.

The data set we'll be exploring is from The Behavioral Risk Factor Surveillance System (BRFSS), a health-related telephone survey that is collected annually by the CDC. The response is a binary variable that has 2 classes. 0 is for no diabetes, and 1 is for pre-diabetes or
diabetes. There are also 21 variables collected in the data set that might have some association with the response.

In order to learn the distribution of those variables and their relationship with each other, we will conduct a series of data exploratory analysis and generate contingency tables, summary tables, chi-square tests, bar plots, density plots and correlation chart.

After getting some general understanding about the data, we are planning on spitting the data set into training and test set, fitting some prediction models on the training set with different model types and turning parameters, also utilizing the cross-validation method for model selection, and log-loss for model performance evaluation. Our final goal is to find the model that returns the lowest log-loss value when applied on the test set for prediction.

## Packages
The packages we used in our program are listed as below:  
`tidyverse`  
`ggplot2`  
`caret`  
`GGally`  
`devtools`  
`leaps`  
`glmnet`  
`RWeka`  
`gridExtra`  
`readr`  
`scales`

## Automation code

library(tidyverse)  
library(rmarkdown)  

edu <- c("1", "2", "3", "4", "5")  
out <- c("Education_Level_1", "Education_Level_2", "Education_Level_3", "Education_Level_4", "Education_Level_5")  
output_md <- paste0(out, ".md")  
params = lapply(edu, FUN = function(x){list(Education = x)})  
reports <- tibble(output_md, params)  
apply(reports, MARGIN = 1,  
      FUN = function(x){  
        render(input = "Project_3_final.Rmd", output_file = x[[1]], params = x[[2]])  
      })  

## Links to the pages

1.  Analysis for [Never attended school, kindergarten or elementary](https://vivifeathers.github.io/ST558-PROJECT-3/Education_Level_1)
2.  Analysis for [Some high school](https://vivifeathers.github.io/ST558-PROJECT-3/Education_Level_2)
3.  Analysis for [High school graduate](https://vivifeathers.github.io/ST558-PROJECT-3/Education_Level_3)
4.  Analysis for [Some college or technical school](https://vivifeathers.github.io/ST558-PROJECT-3/Education_Level_4)
5.  Analysis for [College graduate](https://vivifeathers.github.io/ST558-PROJECT-3/Education_Level_5)
 


