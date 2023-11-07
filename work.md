Project 3
================
Vivi
2023-11-02

- [Introduction](#introduction)
- [Data](#data)
- [Summarizations](#summarizations)
  - [The response: “Diabetes_binary”](#the-response-diabetes_binary)
  - [Contigency table and Chi-square](#contigency-table-and-chi-square)

# Introduction

# Data

First of all, We will call the required packages and read in the
“diabetes_binary_health_indicators_BRFSS2015.csv” file. According to the
[data
dictionary](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/?select=diabetes_binary_health_indicators_BRFSS2015.csv),
there are 22 columns and most of them are categorical variables, we will
convert them to factors and replace the original variables in the code
below using a `as.factor` function. Lastly, we will use a `filter`
function and subset the “diabetes” data set corresponding to the
`params$Education` value in YAML header for R markdown automation
purpose.

``` r
library(tidyverse)
library(ggplot2)
library(caret)
library(GGally)

#use `as.factor` function to replace the original variables.
diabetes <- read_csv(file = "diabetes_binary_health_indicators_BRFSS2015.csv")
diabetes$Diabetes_binary <- as.factor(diabetes$Diabetes_binary)
diabetes$HighBP <- as.factor(diabetes$HighBP)
diabetes$HighChol <- as.factor(diabetes$HighChol)
diabetes$CholCheck <- as.factor(diabetes$CholCheck)
diabetes$Smoker <- as.factor(diabetes$Smoker)
diabetes$Stroke <- as.factor(diabetes$Stroke)
diabetes$HeartDiseaseorAttack <- as.factor(diabetes$HeartDiseaseorAttack)
diabetes$PhysActivity <- as.factor(diabetes$PhysActivity)
diabetes$Fruits <- as.factor(diabetes$Fruits)
diabetes$Veggies <- as.factor(diabetes$Veggies)
diabetes$HvyAlcoholConsump <- as.factor(diabetes$HvyAlcoholConsump)
diabetes$AnyHealthcare <- as.factor(diabetes$AnyHealthcare)
diabetes$NoDocbcCost <- as.factor(diabetes$NoDocbcCost)
diabetes$GenHlth <- as.factor(diabetes$GenHlth)
diabetes$DiffWalk <- as.factor(diabetes$DiffWalk)
diabetes$Sex <- as.factor(diabetes$Sex)
diabetes$Age <- as.factor(diabetes$Age)
diabetes$Education  <- as.factor(diabetes$Education)
diabetes$Income <- as.factor(diabetes$Income)
diabetes              
```

    ## # A tibble: 253,680 × 22
    ##    Diabetes_binary HighBP HighChol CholCheck   BMI Smoker Stroke
    ##    <fct>           <fct>  <fct>    <fct>     <dbl> <fct>  <fct> 
    ##  1 0               1      1        1            40 1      0     
    ##  2 0               0      0        0            25 1      0     
    ##  3 0               1      1        1            28 0      0     
    ##  4 0               1      0        1            27 0      0     
    ##  5 0               1      1        1            24 0      0     
    ##  6 0               1      1        1            25 1      0     
    ##  7 0               1      0        1            30 1      0     
    ##  8 0               1      1        1            25 1      0     
    ##  9 1               1      1        1            30 1      0     
    ## 10 0               0      0        1            24 0      0     
    ## # ℹ 253,670 more rows
    ## # ℹ 15 more variables: HeartDiseaseorAttack <fct>, PhysActivity <fct>,
    ## #   Fruits <fct>, Veggies <fct>, HvyAlcoholConsump <fct>, AnyHealthcare <fct>,
    ## #   NoDocbcCost <fct>, GenHlth <fct>, MentHlth <dbl>, PhysHlth <dbl>,
    ## #   DiffWalk <fct>, Sex <fct>, Age <fct>, Education <fct>, Income <fct>

``` r
table(diabetes$Education)
```

    ## 
    ##      1      2      3      4      5      6 
    ##    174   4043   9478  62750  69910 107325

``` r
# subset data set based on parameter in YAML header

#diabetes_sub <- diabetes %>% 

#                   filter(Education == params$Education)

diabetes_sub <- diabetes %>% 

                   filter(Education == "6")

diabetes_sub
```

    ## # A tibble: 107,325 × 22
    ##    Diabetes_binary HighBP HighChol CholCheck   BMI Smoker Stroke
    ##    <fct>           <fct>  <fct>    <fct>     <dbl> <fct>  <fct> 
    ##  1 0               0      0        0            25 1      0     
    ##  2 0               1      1        1            25 1      0     
    ##  3 0               1      0        1            30 1      0     
    ##  4 1               0      0        1            25 1      0     
    ##  5 0               0      1        1            33 1      1     
    ##  6 0               1      0        1            33 0      0     
    ##  7 0               0      0        0            23 0      0     
    ##  8 0               0      1        1            28 0      0     
    ##  9 0               0      0        1            32 0      0     
    ## 10 1               1      1        1            37 1      1     
    ## # ℹ 107,315 more rows
    ## # ℹ 15 more variables: HeartDiseaseorAttack <fct>, PhysActivity <fct>,
    ## #   Fruits <fct>, Veggies <fct>, HvyAlcoholConsump <fct>, AnyHealthcare <fct>,
    ## #   NoDocbcCost <fct>, GenHlth <fct>, MentHlth <dbl>, PhysHlth <dbl>,
    ## #   DiffWalk <fct>, Sex <fct>, Age <fct>, Education <fct>, Income <fct>

# Summarizations

Now we are ready to perform an exploratory data analysis, and give some
Summarizations about the center, spread and distribution of numeric
variables in the form of tables and plots. Also provide contingency
tables and bar plots for categorical variables.

## The response: “Diabetes_binary”

Since “Diabetes_binary” is a binary variable, we will create a one way
contingency table and see the count of subjects with and without
diabetes in this education group, also visualize the result in a bar
chart.

``` r
table(diabetes_sub$Diabetes_binary)
```

    ## 
    ##     0     1 
    ## 96925 10400

``` r
g <- ggplot(data = diabetes_sub, aes(x = Diabetes_binary, fill = Diabetes_binary))
g + geom_bar(alpha = 0.6) +
  scale_x_discrete(breaks=c("0","1"),
        labels=c("No", "Yes")) +
  scale_fill_discrete(name = "Diabetes Diagnosis", labels = c("No", "Yes")) +
  labs(x = "Diabetes Diagnosis", y = "Subject count", title = "Bar plot of subject with diabetes and subject withou diabetes count") 
```

![](work_files/figure-gfm/one-1.png)<!-- -->

## Contigency table and Chi-square

Now we want to investigate the relationship between having diabetes vs
all the categorical variables. we will create a function which generates
a contingency table, calculates the row percentage for each level of the
corresponding categorical variable in both diagnosis groups, and gives
the chi-square result based on the contingency table. We need to pay
attention to the categorical variable with a significant chi-square
result (p value smaller than 0.05), that means this categorical variable
may have certain relationship with the diabetes diagnosis.

``` r
chisq <- function (x) {
  a <- table(diabetes_sub$Diabetes_binary, x)
  c <- cbind(a, a/rowSums(a))
  b <- chisq.test(a, correct=FALSE)
  return(list(a, c, b))
}

chisq(diabetes_sub$HighBP)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 66157 30768
    ##   1  2930  7470
    ## 
    ## [[2]]
    ##       0     1         0         1
    ## 0 66157 30768 0.6825587 0.3174413
    ## 1  2930  7470 0.2817308 0.7182692
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 6579.5, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$HighChol)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 61767 35158
    ##   1  3604  6796
    ## 
    ## [[2]]
    ##       0     1         0         1
    ## 0 61767 35158 0.6372659 0.3627341
    ## 1  3604  6796 0.3465385 0.6534615
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 3334.1, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$CholCheck)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0  3761 93164
    ##   1    49 10351
    ## 
    ## [[2]]
    ##      0     1           0         1
    ## 0 3761 93164 0.038803198 0.9611968
    ## 1   49 10351 0.004711538 0.9952885
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 318.81, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$Smoker)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 64668 32257
    ##   1  5847  4553
    ## 
    ## [[2]]
    ##       0     1         0         1
    ## 0 64668 32257 0.6671963 0.3328037
    ## 1  5847  4553 0.5622115 0.4377885
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 459.38, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$Stroke)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 94869  2056
    ##   1  9635   765
    ## 
    ## [[2]]
    ##       0    1         0          1
    ## 0 94869 2056 0.9787877 0.02121228
    ## 1  9635  765 0.9264423 0.07355769
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 1005.5, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$HeartDiseaseorAttack)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 91849  5076
    ##   1  8393  2007
    ## 
    ## [[2]]
    ##       0    1         0          1
    ## 0 91849 5076 0.9476296 0.05237039
    ## 1  8393 2007 0.8070192 0.19298077
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 3012.6, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$PhysActivity)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 13781 83144
    ##   1  2849  7551
    ## 
    ## [[2]]
    ##       0     1         0         1
    ## 0 13781 83144 0.1421821 0.8578179
    ## 1  2849  7551 0.2739423 0.7260577
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 1245.3, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$Fruits)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 29108 67817
    ##   1  3760  6640
    ## 
    ## [[2]]
    ##       0     1         0         1
    ## 0 29108 67817 0.3003147 0.6996853
    ## 1  3760  6640 0.3615385 0.6384615
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 165.7, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$Veggies)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 12068 84857
    ##   1  1794  8606
    ## 
    ## [[2]]
    ##       0     1         0         1
    ## 0 12068 84857 0.1245086 0.8754914
    ## 1  1794  8606 0.1725000 0.8275000
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 192.32, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$HvyAlcoholConsump)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 90705  6220
    ##   1 10125   275
    ## 
    ## [[2]]
    ##       0    1         0          1
    ## 0 90705 6220 0.9358267 0.06417333
    ## 1 10125  275 0.9735577 0.02644231
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 235.18, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$AnyHealthcare)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0  2376 94549
    ##   1   218 10182
    ## 
    ## [[2]]
    ##      0     1          0         1
    ## 0 2376 94549 0.02451380 0.9754862
    ## 1  218 10182 0.02096154 0.9790385
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 5.025, df = 1, p-value = 0.02498

``` r
chisq(diabetes_sub$NoDocbcCost)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 91874  5051
    ##   1  9659   741
    ## 
    ## [[2]]
    ##       0    1         0          1
    ## 0 91874 5051 0.9478875 0.05211246
    ## 1  9659  741 0.9287500 0.07125000
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 67.376, df = 1, p-value = 2.243e-16

``` r
chisq(diabetes_sub$GenHlth)
```

    ## [[1]]
    ##    x
    ##         1     2     3     4     5
    ##   0 26023 41480 22751  5274  1397
    ##   1   463  2626  4343  2146   822
    ## 
    ## [[2]]
    ##       1     2     3    4    5          1         2         3          4
    ## 0 26023 41480 22751 5274 1397 0.26848594 0.4279598 0.2347279 0.05441321
    ## 1   463  2626  4343 2146  822 0.04451923 0.2525000 0.4175962 0.20634615
    ##            5
    ## 0 0.01441321
    ## 1 0.07903846
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 8890, df = 4, p-value < 2.2e-16

``` r
chisq(diabetes_sub$DiffWalk)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 89388  7537
    ##   1  7647  2753
    ## 
    ## [[2]]
    ##       0    1         0          1
    ## 0 89388 7537 0.9222388 0.07776116
    ## 1  7647 2753 0.7352885 0.26471154
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 3786.9, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$Sex)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 53299 43626
    ##   1  4538  5862
    ## 
    ## [[2]]
    ##       0     1         0         1
    ## 0 53299 43626 0.5498994 0.4501006
    ## 1  4538  5862 0.4363462 0.5636538
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 487.38, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$Income)
```

    ## [[1]]
    ##    x
    ##         1     2     3     4     5     6     7     8
    ##   0  1087  1230  1908  3136  5507 10773 17638 55646
    ##   1   220   315   434   584   983  1597  2094  4173
    ## 
    ## [[2]]
    ##      1    2    3    4    5     6     7     8          1          2          3
    ## 0 1087 1230 1908 3136 5507 10773 17638 55646 0.01121486 0.01269022 0.01968532
    ## 1  220  315  434  584  983  1597  2094  4173 0.02115385 0.03028846 0.04173077
    ##            4          5         6         7        8
    ## 0 0.03235491 0.05681713 0.1111478 0.1819758 0.574114
    ## 1 0.05615385 0.09451923 0.1535577 0.2013462 0.401250
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 1531, df = 7, p-value < 2.2e-16
