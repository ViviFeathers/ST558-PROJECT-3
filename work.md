Project 3
================
Vivi
2023-11-02

- [Introduction](#introduction)
- [Data](#data)
- [Summarizations](#summarizations)
  - [The response: “Diabetes_binary”](#the-response-diabetes_binary)
  - [Contigency table and Chi-square](#contigency-table-and-chi-square)
  - [“HighBP”](#highbp)
  - [“HighChol”](#highchol)
  - [“BMI”](#bmi)
  - [Correlation among numeric
    variables](#correlation-among-numeric-variables)
- [Modeling](#modeling)
  - [Training and test set](#training-and-test-set)
  - [Log-Loss](#log-loss)
  - [Logistic regression](#logistic-regression)
  - [Model selection for logistic
    regression](#model-selection-for-logistic-regression)
  - [Random forest](#random-forest)
  - [Logistic model tree](#logistic-model-tree)
- [Final model selection](#final-model-selection)

# Introduction

# Data

First of all, We will call the required packages and read in the
“diabetes_binary_health_indicators_BRFSS2015.csv” file. According to the
[data
dictionary](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/?select=diabetes_binary_health_indicators_BRFSS2015.csv),
there are 22 columns and most of them are categorical variables, we will
convert them to factors and replace the original variables in the code
below using the `as.factor` function. Additionally, since Education
level one has very a few subjects, we will combine Education level 1 and
2 then replace the original Education variable using a series of
`if_else` functions. Lastly, we will utilize a `filter` function and
subset the “diabetes” data set corresponding to the `params$Education`
value in YAML header for R markdown automation purpose.

``` r
library(tidyverse)
library(ggplot2)
library(caret)
library(GGally)
library(devtools)
library(leaps)
library(glmnet)
library(RWeka)


diabetes <- read_csv(file = "diabetes_binary_health_indicators_BRFSS2015.csv")

# for log-loss purpose, create a new variable and assign value as "YES" for the records have Diabetes_binary = 1 and 'NO' otherwise.
diabetes <- diabetes %>%
            mutate(diabetes_dx = as.factor(if_else(Diabetes_binary == 1, 'Yes', 'No')))
#use `as.factor` function to replace the original variables.
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
diabetes$Income <- as.factor(diabetes$Income)

# combine level 1 and 2 of education
diabetes$Education  <- as.factor(if_else(diabetes$Education == 1, 1, 
                                    if_else(diabetes$Education == 2, 1,
                                       if_else(diabetes$Education == 3, 2, 
                                           if_else(diabetes$Education == 4, 3,
                                                    if_else(diabetes$Education == 5, 4,
                                                           if_else(diabetes$Education == 6, 5, NA)))))))

# subset data set based on parameter in YAML header

#diabetes_sub <- diabetes %>% 

#                   filter(Education == params$Education)

diabetes_sub <- diabetes %>% 

                   filter(Education == "3")

diabetes_sub
```

    ## # A tibble: 62,750 × 23
    ##    Diabetes_binary HighBP HighChol CholCheck   BMI Smoker Stroke
    ##    <fct>           <fct>  <fct>    <fct>     <dbl> <fct>  <fct> 
    ##  1 0               1      1        1            40 1      0     
    ##  2 0               1      1        1            28 0      0     
    ##  3 0               1      1        1            25 1      0     
    ##  4 0               0      0        1            24 0      0     
    ##  5 1               1      1        1            28 0      0     
    ##  6 0               1      1        1            21 0      0     
    ##  7 0               1      1        1            22 0      1     
    ##  8 0               1      1        1            28 1      0     
    ##  9 1               1      1        1            27 1      0     
    ## 10 0               0      0        1            23 0      0     
    ## # ℹ 62,740 more rows
    ## # ℹ 16 more variables: HeartDiseaseorAttack <fct>, PhysActivity <fct>,
    ## #   Fruits <fct>, Veggies <fct>, HvyAlcoholConsump <fct>, AnyHealthcare <fct>,
    ## #   NoDocbcCost <fct>, GenHlth <fct>, MentHlth <dbl>, PhysHlth <dbl>,
    ## #   DiffWalk <fct>, Sex <fct>, Age <fct>, Education <fct>, Income <fct>,
    ## #   diabetes_dx <fct>

# Summarizations

Now we are ready to perform an exploratory data analysis, and give some
Summarizations about the center, spread and distribution of numeric
variables in the form of tables and plots. Also provide contingency
tables and bar plots for categorical variables.

## The response: “Diabetes_binary”

Since “Diabetes_binary” is a binary variable, we will create a one way
contingency table with `table` function to see the count of subjects
with and without diabetes in this education group. Also visualize the
result in a bar chart using `ggplot` + `geom_bar` function. For label
specification on x axis, x ticks, y axis and legend, we will use the
`scale_x_discrete`, the `scale_fill_discrete` and the `labs` functions.

``` r
table(diabetes_sub$Diabetes_binary)
```

    ## 
    ##     0     1 
    ## 51684 11066

``` r
g <- ggplot(data = diabetes_sub, aes(x = Diabetes_binary, fill = Diabetes_binary))
g + geom_bar(alpha = 0.6) +
  scale_x_discrete(breaks=c("0","1"),
        labels=c("No", "Yes")) +
  scale_fill_discrete(name = "Diabetes Diagnosis", labels = c("No", "Yes")) +
  labs(x = "Diabetes Diagnosis", y = "Subject count", title = "Bar plot of subject with diabetes and subject withou diabetes count") 
```

![](work_files/figure-gfm/one-1.png)<!-- -->

From this bar chart, we can see which diagnosis group has more subjects
in this education level.

## Contigency table and Chi-square

Now we want to investigate the relationship between having diabetes vs
all the categorical variables. we will create a function which generates
a contingency table, calculates the row percentage for each level of the
corresponding categorical variable in both diagnosis groups, and gives
the chi-square result based on the contingency table.

``` r
chisq <- function (x) {
  #generate contingency table
  a <- table(diabetes_sub$Diabetes_binary, x)
  #calculate the row percentage and combine with original data set
  c <- cbind(a, a/rowSums(a))
  #run chi-square test
  b <- chisq.test(a, correct=FALSE)
  return(list(a, c, b))
}

chisq(diabetes_sub$HighBP)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 28453 23231
    ##   1  2577  8489
    ## 
    ## [[2]]
    ##       0     1         0         1
    ## 0 28453 23231 0.5505185 0.4494815
    ## 1  2577  8489 0.2328755 0.7671245
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 3679, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$HighChol)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 30169 21515
    ##   1  3620  7446
    ## 
    ## [[2]]
    ##       0     1         0         1
    ## 0 30169 21515 0.5837203 0.4162797
    ## 1  3620  7446 0.3271281 0.6728719
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 2414.7, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$CholCheck)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0  2264 49420
    ##   1    88 10978
    ## 
    ## [[2]]
    ##      0     1           0         1
    ## 0 2264 49420 0.043804659 0.9561953
    ## 1   88 10978 0.007952286 0.9920477
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 324.74, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$Smoker)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 24435 27249
    ##   1  4992  6074
    ## 
    ## [[2]]
    ##       0     1         0         1
    ## 0 24435 27249 0.4727769 0.5272231
    ## 1  4992  6074 0.4511115 0.5488885
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 17.179, df = 1, p-value = 3.401e-05

``` r
chisq(diabetes_sub$Stroke)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 49449  2235
    ##   1 10003  1063
    ## 
    ## [[2]]
    ##       0    1         0          1
    ## 0 49449 2235 0.9567564 0.04324356
    ## 1 10003 1063 0.9039400 0.09606000
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 510.6, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$HeartDiseaseorAttack)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 46753  4931
    ##   1  8530  2536
    ## 
    ## [[2]]
    ##       0    1         0         1
    ## 0 46753 4931 0.9045933 0.0954067
    ## 1  8530 2536 0.7708296 0.2291704
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 1555.6, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$PhysActivity)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 16448 35236
    ##   1  4715  6351
    ## 
    ## [[2]]
    ##       0     1         0         1
    ## 0 16448 35236 0.3182416 0.6817584
    ## 1  4715  6351 0.4260799 0.5739201
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 474.21, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$Fruits)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 22019 29665
    ##   1  4992  6074
    ## 
    ## [[2]]
    ##       0     1         0         1
    ## 0 22019 29665 0.4260313 0.5739687
    ## 1  4992  6074 0.4511115 0.5488885
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 23.385, df = 1, p-value = 1.326e-06

``` r
chisq(diabetes_sub$Veggies)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 13180 38504
    ##   1  3298  7768
    ## 
    ## [[2]]
    ##       0     1         0         1
    ## 0 13180 38504 0.2550112 0.7449888
    ## 1  3298  7768 0.2980300 0.7019700
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 87.107, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$HvyAlcoholConsump)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 48619  3065
    ##   1 10820   246
    ## 
    ## [[2]]
    ##       0    1         0          1
    ## 0 48619 3065 0.9406973 0.05930269
    ## 1 10820  246 0.9777697 0.02223025
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 250.63, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$AnyHealthcare)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0  3711 47973
    ##   1   546 10520
    ## 
    ## [[2]]
    ##      0     1          0         1
    ## 0 3711 47973 0.07180172 0.9281983
    ## 1  546 10520 0.04934032 0.9506597
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 72.715, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$NoDocbcCost)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 46595  5089
    ##   1  9837  1229
    ## 
    ## [[2]]
    ##       0    1         0          1
    ## 0 46595 5089 0.9015363 0.09846374
    ## 1  9837 1229 0.8889391 0.11106091
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 15.974, df = 1, p-value = 6.423e-05

``` r
chisq(diabetes_sub$GenHlth)
```

    ## [[1]]
    ##    x
    ##         1     2     3     4     5
    ##   0  6959 16840 17715  7514  2656
    ##   1   265  1630  4133  3477  1561
    ## 
    ## [[2]]
    ##      1     2     3    4    5          1         2         3         4
    ## 0 6959 16840 17715 7514 2656 0.13464515 0.3258262 0.3427560 0.1453835
    ## 1  265  1630  4133 3477 1561 0.02394723 0.1472980 0.3734864 0.3142057
    ##            5
    ## 0 0.05138921
    ## 1 0.14106271
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 4555.5, df = 4, p-value < 2.2e-16

``` r
chisq(diabetes_sub$DiffWalk)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 41928  9756
    ##   1  6551  4515
    ## 
    ## [[2]]
    ##       0    1         0         1
    ## 0 41928 9756 0.8112375 0.1887625
    ## 1  6551 4515 0.5919935 0.4080065
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 2493.5, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$Sex)
```

    ## [[1]]
    ##    x
    ##         0     1
    ##   0 29014 22670
    ##   1  6106  4960
    ## 
    ## [[2]]
    ##       0     1         0         1
    ## 0 29014 22670 0.5613730 0.4386270
    ## 1  6106  4960 0.5517802 0.4482198
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 3.4034, df = 1, p-value = 0.06506

``` r
chisq(diabetes_sub$Income)
```

    ## [[1]]
    ##    x
    ##        1    2    3    4    5    6    7    8
    ##   0 2711 3436 4992 6312 7403 9159 8186 9485
    ##   1  883 1256 1519 1717 1643 1713 1306 1029
    ## 
    ## [[2]]
    ##      1    2    3    4    5    6    7    8          1          2          3
    ## 0 2711 3436 4992 6312 7403 9159 8186 9485 0.05245337 0.06648092 0.09658695
    ## 1  883 1256 1519 1717 1643 1713 1306 1029 0.07979396 0.11350081 0.13726731
    ##           4         5         6         7          8
    ## 0 0.1221268 0.1432358 0.1772115 0.1583856 0.18351908
    ## 1 0.1551599 0.1484728 0.1547985 0.1180192 0.09298753
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 1183.7, df = 7, p-value < 2.2e-16

We need to pay attention to the categorical variable with a significant
chi-square result (p value smaller than 0.05), that means this
categorical variable may have certain relationship with the diabetes
diagnosis.

## “HighBP”

As we all know, high blood pressure and diabetes are related, we want to
create a bar chart and visualize the high blood pressure subjects’ count
and ratio in each diagnosis group. Here we will again, use the `gglot`
function and the `geom_bar` function to create the plot. Like last bar
chart, we also will set labels for x axis, x ticks, y axis and legend
with the `scale_x_discrete`, the `scale_fill_discrete` and the`labs`
functions.

``` r
# create a bar plot using the gglot and the geom_bar function.
h <- ggplot(data = diabetes_sub, aes(x = Diabetes_binary, fill = HighBP))
h + geom_bar(position = "dodge", alpha = 0.6) +
  scale_x_discrete(breaks=c("0","1"),
       labels=c("No", "Yes")) +
  scale_fill_discrete(name = "High Blood Pressure", labels = c("No", "Yes")) +
  labs(x = "Diabetes Diagnosis", y = "Subject count", title = "Bar plot of high blood pressure subject count in diabetes and non-diabetes group")
```

![](work_files/figure-gfm/bar1-1.png)<!-- -->

In the bar chart above, we need to focus on the count and ratio of
subjects with high blood pressure vs subjects without in both diabetes
and non-diabetes group, and verify if the ratio of high blood pressure
subjects in the diabetes group is higher than the non-diabetes group’s
as we assumed.

## “HighChol”

Another health condition that associates with diabetes is high
cholesterol, we also want to create a bar chart and compare the high
cholesterol subjects’ count and ratio in each diagnosis group, using all
the functions we used for previous bar plots.

``` r
# create a bar plot using the gglot and the geom_bar function.
h <- ggplot(data = diabetes_sub, aes(x = Diabetes_binary, fill = HighChol))
h + geom_bar(position = "dodge", alpha = 0.6) +
  scale_x_discrete(breaks=c("0","1"),
       labels=c("No", "Yes")) +
  scale_fill_discrete(name = "High Cholesterol", labels = c("No", "Yes")) +
  labs(x = "Diabetes Diagnosis", y = "Subject count", title = "Bar plot of high cholesterol subject count in diabetes and non-diabetes group")
```

![](work_files/figure-gfm/bar2-1.png)<!-- -->

In the bar chart above, we also need to look at the count and ratio of
subjects with high cholesterol vs subjects without in both diabetes and
non-diabetes group, and verify if the ratio of high cholesterol subjects
in the diabetes group is higher than the non-diabetes group’s as we
expected.

## “BMI”

The association of high BMI and diabetes are proved in many scientific
studies, in order to verify if this association also exists in our data,
we are looking into the distribution of BMI in both diagnosis group by
creating a summary table using `group_by` and `summarise` function, as
well as generating a kernel density plot with `ggplot` and
`geom_density` function. Additionally, we want to conduct a two sample
t-test with `t.test` function and investigate if the means in each
diagnosis group are different from each other.

``` r
#create a summary table for BMI to display the center and spread
diabetes_sub %>%
  group_by(Diabetes_binary) %>%
  summarise(Mean = mean(BMI),  Standard_Deviation = sd(BMI), 
            Variance = var(BMI), Median = median(BMI), 
            q1 = quantile(BMI, probs = 0.25),
            q3 = quantile(BMI, probs = 0.75))
```

    ## # A tibble: 2 × 7
    ##   Diabetes_binary  Mean Standard_Deviation Variance Median    q1    q3
    ##   <fct>           <dbl>              <dbl>    <dbl>  <dbl> <dbl> <dbl>
    ## 1 0                28.4               6.43     41.3     27    24    31
    ## 2 1                32.1               7.42     55.0     31    27    36

``` r
#generate a kernal density plot to show the distribution of BMI
i <- ggplot(data = diabetes_sub, aes(x = BMI, fill = Diabetes_binary)) 
i + geom_density(adjust = 1, color="#e9ecef", alpha=0.5, position = 'dodge') +
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  labs(x = "BMI", title = "Kernal density plot of BMI distribution across diagnosis group") 
```

![](work_files/figure-gfm/bmi-1.png)<!-- -->

``` r
#conduct a two sample t test for BMI in both diagnosis groups
t.test(BMI ~ Diabetes_binary, data = diabetes_sub, alternative = "two.sided", var.equal = FALSE)
```

    ## 
    ##  Welch Two Sample t-test
    ## 
    ## data:  BMI by Diabetes_binary
    ## t = -48.424, df = 14826, p-value < 2.2e-16
    ## alternative hypothesis: true difference in means between group 0 and group 1 is not equal to 0
    ## 95 percent confidence interval:
    ##  -3.827193 -3.529413
    ## sample estimates:
    ## mean in group 0 mean in group 1 
    ##        28.39372        32.07202

The summary table gives us the center (mean and median) and spread
(standard_deviation, variance, q1 and q3) of BMI in each diagnosis
group. The density plot shows the distribution of BMI in each diagnosis
group, we can estimate the means and standard deviations, also visualize
their differences. Lastly, the two sample t-test returns the 95%
confidence interval of mean difference, and the t statistic, degree of
freedom and p-value as t-test result, we need to see if the p-value is
smaller than 0.05 to decide whether we will reject the null hypothesis
(true difference in means between two diagnosis is equal to 0).

## Correlation among numeric variables

Before fitting any model, we also want to test the correlation among
numeric variables, in case there is collinearity that diminishes our
ability to determine which variables are responsible for change in the
response variable. For this reason, we should determine correlations
between all variables and consider removing ones that are problematic.
We can do this by creating a correlation plot with `ggpairs()` from the
`GGally` package.

``` r
x <- diabetes_sub[, c(5, 16, 17)] 
GGally::ggpairs(x)
```

![](work_files/figure-gfm/graphics-1.png)<!-- -->

Let’s look for the pairs that have correlation coefficient higher than
0.4. If we do find such pairs, we need to refer to the relationship
between each variable and the response, as well as some background
knowledge (may do a literature review), in order to decide which
variable will be include into the models.

# Modeling

Since we gained some basic understanding of the data, we can now fit
different models and select the best one for prediction.

## Training and test set

The first step is to split our data to a training and test set. We will
use the `createDataPartition` function and split the data by 70% as the
training set and 30% as the test set.

``` r
set.seed(20)
index <- createDataPartition(diabetes_sub$Diabetes_binary, p = 0.70, list = FALSE)
train <- diabetes_sub[index, ]
test <- diabetes_sub[-index, ]
```

## Log-Loss

All of our models’ performance will be evaluated by Log-loss. Log loss,
also known as logarithmic loss, indicates how close a prediction
probability comes to the corresponding true binary value. Thus, it is a
common evaluation metric for binary classification models. There are
three steps to calculate Log Loss:

1.  Finding the corrected probabilities.  
2.  Taking a log of corrected probabilities.  
3.  Taking the negative average of the values from step 2.

And the formula is as below:

![](log_loss_formula.PNG)

From the formula we can tell, the lower log-loss is, the better the
model performs.

The reason that log-loss is preferred to accuracy is: For accuracy,
model returns 1 if predicted probability is \> .5 otherwise 0. We could
get some prediction probabilities very close to 0.5, but are converted
to 1 and 0 which actually match with the true results. Such model could
have 100% accuracy but their prediction probabilities are at board line
of being wrong. If we use accuracy for model selection, we will keep a
bad model. On the other hand, log-loss calculate how far away the
prediction probabilities are from the true values, and the log-loss of
such model will be high, which reveal the model’s true performance.
Thus, log-loss is more reliable and accurate for model selection.

## Logistic regression

The first model we are fitting is logistic regression, which is a
generalized linear model that models the probability of an event by
calculating the log-odds for the event based on the linear combination
of one or more independent variables. The most common logistic
regression has a single binary variable as the response, usually the two
values are coded as “0” and “1”.

In logistic regression, the dependent variable is a logit, which is the
natural log of the odds and assumed to be linearly related to X as the
formula below:

![](logistic.PNG)

In our data set, logistic regression models are more suitable for our
binary response than linear regression models because:

1.  If we use linear regression, the predicted values will become
    greater than one and less than zero if we move far enough on the
    X-axis. Such values are not realistic for binary variable.  
2.  One of the assumptions of regression is that the variance of Y is
    constant across values of X (homoscedasticity). This can not be the
    case with a binary variable.  
3.  The significance testing of the b weights rest upon the assumption
    that errors of prediction (Y-Y’) are normally distributed. Because Y
    only takes the values 0 and 1, this assumption is pretty hard to
    justify.

Now, let’s fit 3 logistic regression models with the `train` function on
the training data set, and use cross-validation with 5 folds for model
selection. After the best model is picked, use the `predict` function to
apply it on the test data set and merge the predicted results with the
test set’s true response. We will use the `mnLogLoss` function to
evaluate their performance with a log-loss method.

After log-loss values get calculated for all three models, we will
create a data frame with all the log-loss values and their corresponding
model names, then return the row with the lowest log-loss value by using
the `which.min` function. Lastly, we will insert the model name as an
inline R code into the conclusion.

### Model one

We will include all the low level variables in our first model:

``` r
#fit the model on the training data set
full_log <- train(diabetes_dx ~ HighBP + HighChol + CholCheck + BMI + Smoker + Stroke + HeartDiseaseorAttack 
                  + PhysActivity + Fruits + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth + MentHlth 
                  + PhysHlth + DiffWalk + Sex + Age + Income,
              data=train, 
              method = "glm", 
              family = "binomial",
              metric="logLoss",
              preProcess = c("center", "scale"),
              trControl = trainControl(method = "cv", number = 5, classProbs=TRUE, summaryFunction=mnLogLoss))
full_log
```

    ## Generalized Linear Model 
    ## 
    ## 43926 samples
    ##    20 predictor
    ##     2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (40), scaled (40) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 35141, 35140, 35141, 35141, 35141 
    ## Resampling results:
    ## 
    ##   logLoss  
    ##   0.3858393

``` r
summary(full_log)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Coefficients:
    ##                        Estimate Std. Error  z value Pr(>|z|)    
    ## (Intercept)           -2.012481   0.019011 -105.861  < 2e-16 ***
    ## HighBP1                0.325233   0.016157   20.129  < 2e-16 ***
    ## HighChol1              0.255956   0.014651   17.471  < 2e-16 ***
    ## CholCheck1             0.199733   0.025832    7.732 1.06e-14 ***
    ## BMI                    0.379720   0.013567   27.989  < 2e-16 ***
    ## Smoker1               -0.047999   0.014295   -3.358 0.000786 ***
    ## Stroke1                0.033833   0.011491    2.944 0.003236 ** 
    ## HeartDiseaseorAttack1  0.080988   0.012119    6.683 2.35e-11 ***
    ## PhysActivity1         -0.027622   0.013841   -1.996 0.045962 *  
    ## Fruits1                0.003012   0.014349    0.210 0.833739    
    ## Veggies1              -0.007467   0.013847   -0.539 0.589690    
    ## HvyAlcoholConsump1    -0.184320   0.019447   -9.478  < 2e-16 ***
    ## AnyHealthcare1         0.027842   0.016414    1.696 0.089839 .  
    ## NoDocbcCost1          -0.012307   0.014377   -0.856 0.391965    
    ## GenHlth2               0.281994   0.037707    7.479 7.51e-14 ***
    ## GenHlth3               0.594335   0.038060   15.616  < 2e-16 ***
    ## GenHlth4               0.618684   0.031881   19.406  < 2e-16 ***
    ## GenHlth5               0.424598   0.023964   17.718  < 2e-16 ***
    ## MentHlth              -0.029407   0.014308   -2.055 0.039845 *  
    ## PhysHlth              -0.011458   0.015973   -0.717 0.473177    
    ## DiffWalk1              0.072819   0.014565    5.000 5.74e-07 ***
    ## Sex1                   0.118430   0.014645    8.087 6.14e-16 ***
    ## Age2                  -0.027461   0.046403   -0.592 0.553987    
    ## Age3                   0.088897   0.044758    1.986 0.047016 *  
    ## Age4                   0.177934   0.045768    3.888 0.000101 ***
    ## Age5                   0.212338   0.048390    4.388 1.14e-05 ***
    ## Age6                   0.339143   0.056628    5.989 2.11e-09 ***
    ## Age7                   0.437057   0.067666    6.459 1.05e-10 ***
    ## Age8                   0.494629   0.074211    6.665 2.64e-11 ***
    ## Age9                   0.559486   0.073763    7.585 3.33e-14 ***
    ## Age10                  0.611992   0.071818    8.521  < 2e-16 ***
    ## Age11                  0.568401   0.067426    8.430  < 2e-16 ***
    ## Age12                  0.473539   0.060659    7.807 5.87e-15 ***
    ## Age13                  0.445385   0.064427    6.913 4.74e-12 ***
    ## Income2                0.003501   0.017821    0.196 0.844253    
    ## Income3               -0.018202   0.019728   -0.923 0.356186    
    ## Income4               -0.030760   0.021198   -1.451 0.146762    
    ## Income5               -0.073551   0.022412   -3.282 0.001032 ** 
    ## Income6               -0.091181   0.023886   -3.817 0.000135 ***
    ## Income7               -0.121376   0.023761   -5.108 3.25e-07 ***
    ## Income8               -0.157405   0.025721   -6.120 9.38e-10 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 40925  on 43925  degrees of freedom
    ## Residual deviance: 33806  on 43885  degrees of freedom
    ## AIC: 33888
    ## 
    ## Number of Fisher Scoring iterations: 6

``` r
#apply the best model on the test set and merge the predicted results with the true response into one data frame
predicted <- data.frame(obs=test$diabetes_dx,
             pred=predict(full_log, test),
             predict(full_log, test, type="prob"))

#calculate the log-loss
a <- mnLogLoss(predicted, lev = levels(predicted$obs))
```

### Model two

We will include all the low level variables and interactions between
HighBP & HighChol, HighBP & BMI, HighChol & BMI, PhysActivity & BMI, and
BMI & GenHlth in our second model:

``` r
#fit the model on the training data set
inter_log <- train(diabetes_dx ~ HighBP + HighChol + CholCheck + BMI + Smoker + Stroke + HeartDiseaseorAttack 
                  + PhysActivity + Fruits + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth + MentHlth 
                  + PhysHlth + DiffWalk + Sex + Age + Income + HighBP:HighChol + HighBP:BMI + HighChol:BMI + PhysActivity:BMI + BMI:GenHlth,
              data=train, 
              method = "glm", 
              family = "binomial",
              metric="logLoss",
              preProcess = c("center", "scale"),
              trControl = trainControl(method = "cv", number = 5, classProbs=TRUE, summaryFunction=mnLogLoss))
inter_log
```

    ## Generalized Linear Model 
    ## 
    ## 43926 samples
    ##    20 predictor
    ##     2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (48), scaled (48) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 35141, 35141, 35140, 35141, 35141 
    ## Resampling results:
    ## 
    ##   logLoss  
    ##   0.3857017

``` r
summary(inter_log)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Coefficients:
    ##                         Estimate Std. Error  z value Pr(>|z|)    
    ## (Intercept)           -2.0044630  0.0192845 -103.941  < 2e-16 ***
    ## HighBP1                0.2724196  0.0684602    3.979 6.91e-05 ***
    ## HighChol1              0.0787615  0.0635967    1.238 0.215548    
    ## CholCheck1             0.2000197  0.0258185    7.747 9.40e-15 ***
    ## BMI                    0.2957494  0.0694205    4.260 2.04e-05 ***
    ## Smoker1               -0.0485935  0.0143177   -3.394 0.000689 ***
    ## Stroke1                0.0345280  0.0115143    2.999 0.002711 ** 
    ## HeartDiseaseorAttack1  0.0815299  0.0121487    6.711 1.93e-11 ***
    ## PhysActivity1         -0.1735190  0.0590196   -2.940 0.003282 ** 
    ## Fruits1                0.0029280  0.0143669    0.204 0.838506    
    ## Veggies1              -0.0075369  0.0138634   -0.544 0.586680    
    ## HvyAlcoholConsump1    -0.1844169  0.0194648   -9.474  < 2e-16 ***
    ## AnyHealthcare1         0.0274419  0.0164358    1.670 0.094989 .  
    ## NoDocbcCost1          -0.0124291  0.0144014   -0.863 0.388114    
    ## GenHlth2               0.4166740  0.1480826    2.814 0.004896 ** 
    ## GenHlth3               0.5940521  0.1478190    4.019 5.85e-05 ***
    ## GenHlth4               0.6032224  0.1217321    4.955 7.22e-07 ***
    ## GenHlth5               0.3966224  0.0859990    4.612 3.99e-06 ***
    ## MentHlth              -0.0292884  0.0143413   -2.042 0.041128 *  
    ## PhysHlth              -0.0117623  0.0160080   -0.735 0.462476    
    ## DiffWalk1              0.0716506  0.0146019    4.907 9.25e-07 ***
    ## Sex1                   0.1183596  0.0146692    8.069 7.11e-16 ***
    ## Age2                  -0.0236513  0.0464234   -0.509 0.610423    
    ## Age3                   0.0924722  0.0448754    2.061 0.039337 *  
    ## Age4                   0.1830009  0.0458868    3.988 6.66e-05 ***
    ## Age5                   0.2171467  0.0485398    4.474 7.69e-06 ***
    ## Age6                   0.3451620  0.0568163    6.075 1.24e-09 ***
    ## Age7                   0.4442473  0.0679076    6.542 6.07e-11 ***
    ## Age8                   0.5024945  0.0744742    6.747 1.51e-11 ***
    ## Age9                   0.5680437  0.0740321    7.673 1.68e-14 ***
    ## Age10                  0.6209774  0.0720788    8.615  < 2e-16 ***
    ## Age11                  0.5769475  0.0676726    8.526  < 2e-16 ***
    ## Age12                  0.4817350  0.0608679    7.914 2.48e-15 ***
    ## Age13                  0.4539989  0.0646467    7.023 2.18e-12 ***
    ## Income2                0.0034896  0.0178601    0.195 0.845093    
    ## Income3               -0.0186880  0.0197638   -0.946 0.344369    
    ## Income4               -0.0312279  0.0212367   -1.470 0.141434    
    ## Income5               -0.0736972  0.0224532   -3.282 0.001030 ** 
    ## Income6               -0.0918691  0.0239376   -3.838 0.000124 ***
    ## Income7               -0.1223112  0.0238140   -5.136 2.81e-07 ***
    ## Income8               -0.1575266  0.0257656   -6.114 9.73e-10 ***
    ## `HighBP1:HighChol1`    0.0219126  0.0291150    0.753 0.451677    
    ## `HighBP1:BMI`          0.0425357  0.0675876    0.629 0.529125    
    ## `HighChol1:BMI`        0.1632143  0.0607437    2.687 0.007211 ** 
    ## `BMI:PhysActivity1`    0.1441648  0.0567542    2.540 0.011080 *  
    ## `BMI:GenHlth2`        -0.1311778  0.1411235   -0.930 0.352617    
    ## `BMI:GenHlth3`        -0.0008124  0.1492988   -0.005 0.995658    
    ## `BMI:GenHlth4`         0.0147412  0.1260295    0.117 0.906887    
    ## `BMI:GenHlth5`         0.0280676  0.0886627    0.317 0.751572    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 40925  on 43925  degrees of freedom
    ## Residual deviance: 33787  on 43877  degrees of freedom
    ## AIC: 33885
    ## 
    ## Number of Fisher Scoring iterations: 6

``` r
#apply the best model on the test set and merge the predicted results with the true response into one data frame
predicted2 <- data.frame(obs=test$diabetes_dx,
             pred=predict(inter_log, test),
             predict(inter_log, test, type="prob"))

#calculate the log-loss
b <- mnLogLoss(predicted2, lev = levels(predicted2$obs))
```

### Model three

We will include all the low level variables and polynomial term for
numeric variables in our third model:

``` r
#fit the model on the training data set
poly_log <- train(diabetes_dx ~  HighBP + HighChol + CholCheck + I(BMI^2) + BMI + Smoker 
                  + Stroke + HeartDiseaseorAttack + PhysActivity + Fruits + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth 
                  + I(MentHlth^2) + MentHlth + I(PhysHlth^2) + PhysHlth + DiffWalk + Sex + Age + Income,
              data=train, 
              method = "glm", 
              family = "binomial",
              metric="logLoss",
              preProcess = c("center", "scale"),
              trControl = trainControl(method = "cv", number = 5, classProbs=TRUE, summaryFunction=mnLogLoss))
poly_log
```

    ## Generalized Linear Model 
    ## 
    ## 43926 samples
    ##    20 predictor
    ##     2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (43), scaled (43) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 35140, 35141, 35141, 35142, 35140 
    ## Resampling results:
    ## 
    ##   logLoss  
    ##   0.3832144

``` r
summary(poly_log)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Coefficients:
    ##                        Estimate Std. Error  z value Pr(>|z|)    
    ## (Intercept)           -2.036571   0.019239 -105.857  < 2e-16 ***
    ## HighBP1                0.307281   0.016219   18.946  < 2e-16 ***
    ## HighChol1              0.252589   0.014691   17.193  < 2e-16 ***
    ## CholCheck1             0.194673   0.025797    7.546 4.48e-14 ***
    ## `I(BMI^2)`            -0.823979   0.060731  -13.568  < 2e-16 ***
    ## BMI                    1.227791   0.062488   19.649  < 2e-16 ***
    ## Smoker1               -0.040645   0.014341   -2.834 0.004594 ** 
    ## Stroke1                0.035898   0.011567    3.103 0.001913 ** 
    ## HeartDiseaseorAttack1  0.079547   0.012199    6.521 7.01e-11 ***
    ## PhysActivity1         -0.023023   0.013889   -1.658 0.097393 .  
    ## Fruits1                0.001477   0.014392    0.103 0.918246    
    ## Veggies1              -0.007231   0.013891   -0.521 0.602650    
    ## HvyAlcoholConsump1    -0.175559   0.019425   -9.038  < 2e-16 ***
    ## AnyHealthcare1         0.026395   0.016423    1.607 0.108005    
    ## NoDocbcCost1          -0.011135   0.014422   -0.772 0.440053    
    ## GenHlth2               0.270449   0.037695    7.175 7.25e-13 ***
    ## GenHlth3               0.572550   0.038092   15.031  < 2e-16 ***
    ## GenHlth4               0.601966   0.032020   18.800  < 2e-16 ***
    ## GenHlth5               0.421714   0.023977   17.588  < 2e-16 ***
    ## `I(MentHlth^2)`        0.058925   0.050870    1.158 0.246728    
    ## MentHlth              -0.088830   0.052696   -1.686 0.091854 .  
    ## `I(PhysHlth^2)`       -0.067762   0.055074   -1.230 0.218550    
    ## PhysHlth               0.063151   0.058229    1.085 0.278131    
    ## DiffWalk1              0.061464   0.014693    4.183 2.87e-05 ***
    ## Sex1                   0.111163   0.014754    7.534 4.91e-14 ***
    ## Age2                  -0.034035   0.046234   -0.736 0.461637    
    ## Age3                   0.081829   0.044532    1.838 0.066132 .  
    ## Age4                   0.162237   0.045629    3.556 0.000377 ***
    ## Age5                   0.195336   0.048228    4.050 5.12e-05 ***
    ## Age6                   0.322780   0.056419    5.721 1.06e-08 ***
    ## Age7                   0.418311   0.067425    6.204 5.50e-10 ***
    ## Age8                   0.473859   0.073949    6.408 1.48e-10 ***
    ## Age9                   0.538992   0.073511    7.332 2.27e-13 ***
    ## Age10                  0.594306   0.071575    8.303  < 2e-16 ***
    ## Age11                  0.554323   0.067207    8.248  < 2e-16 ***
    ## Age12                  0.466280   0.060467    7.711 1.25e-14 ***
    ## Age13                  0.449362   0.064224    6.997 2.62e-12 ***
    ## Income2               -0.001004   0.017895   -0.056 0.955250    
    ## Income3               -0.024775   0.019799   -1.251 0.210819    
    ## Income4               -0.036888   0.021258   -1.735 0.082695 .  
    ## Income5               -0.082372   0.022494   -3.662 0.000250 ***
    ## Income6               -0.103460   0.023974   -4.315 1.59e-05 ***
    ## Income7               -0.134904   0.023847   -5.657 1.54e-08 ***
    ## Income8               -0.169517   0.025801   -6.570 5.02e-11 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 40925  on 43925  degrees of freedom
    ## Residual deviance: 33562  on 43882  degrees of freedom
    ## AIC: 33650
    ## 
    ## Number of Fisher Scoring iterations: 6

``` r
#apply the best model on the test set and merge the predicted results with the true response into one data frame
predicted3 <- data.frame(obs=test$diabetes_dx,
             pred=predict(poly_log, test),
             predict(poly_log, test, type="prob"))

#calculate the log-loss
c <- mnLogLoss(predicted3, lev = levels(predicted3$obs))
```

## Model selection for logistic regression

Now we have log-loss returned from 3 logistic regression models, we want
to pick the one with the lowest log-loss value. In below code, we will
compare their log-loss and pick the corresponding one with the lowest
log-loss value.

``` r
log_pick <- data.frame(logloss_value = c(a, b, c), model = c("Logistic regression model one", "Logistic regression model two", "Logistc regression model three"))
log_best <- log_pick[which.min(log_pick$logloss_value),]
log_best
```

    ## # A tibble: 1 × 2
    ##   logloss_value model                         
    ##           <dbl> <chr>                         
    ## 1         0.379 Logistc regression model three

From the result we can tell, the Logistc regression model three has the
lowest log-loss value (0.3791432), thus, the Logistc regression model
three is the best logistic regression model for predicting diabetes
diagnosis.

## Random forest

The next model we will fit is a tree based model called random forest.
This model is made up of multiple decision trees which are
non-parametric supervised learning method and used for classification
and regression. The goal of decision tree is to create a model that
predicts the value of a target variable by splitting the predictor into
regions with different predictions for each region. A random forest
utilizes the “bootstrap” method to takes repeatedly sampling with
replacement, fits multiple decision trees with a random subset of
predictors, then returns the average result from all the decision trees.

Random forests are generally more accurate than individual
classification trees because there is always a scope for over fitting
caused by the presence of variance in classification trees, while random
forests combine multiple trees and prevent over fitting. Random forests
also average the predicted results from classification trees and gives a
more accurate and precise prediction.

Now we can set up a random forest model for our data. we will still use
the `train` function but with method `rf`. The `tuneGrid` option is
where we tell the model how many predictor variables to grab per
bootstrap sample. The `trainControl` function within the `trControl`
option will be used for cross-validation with 5 folds for model
selection. After the best model is picked, use the `predict` function to
apply it on the test data set and merge the predicted results with the
test set’s true response. Lastly, use the `mnLogLoss` function to
compare their performance with a log-loss method.

``` r
ran_for <- train(diabetes_dx ~ HighBP + HighChol + CholCheck + BMI + Smoker + Stroke + HeartDiseaseorAttack 
                  + PhysActivity + Fruits + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth + MentHlth 
                  + PhysHlth + DiffWalk + Sex + Age + Income, 
                data = train,
                method = "rf",
                metric="logLoss",
                preProcess = c("center", "scale"),
                tuneGrid = data.frame(mtry = c(6:8)),
                trControl = trainControl(method = "cv", number = 5, classProbs=TRUE, summaryFunction=mnLogLoss))
ran_for
```

    ## Random Forest 
    ## 
    ## 43926 samples
    ##    20 predictor
    ##     2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (40), scaled (40) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 35140, 35140, 35142, 35141, 35141 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  logLoss  
    ##   6     0.4177474
    ##   7     0.4193669
    ##   8     0.4196101
    ## 
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 6.

``` r
#apply the best model on the test set and merge the predicted results with the true response into one data frame
predicted4 <- data.frame(obs=test$diabetes_dx,
             pred=predict(ran_for, test),
             predict(ran_for, test, type="prob"))

#calculate the log-loss

d <- mnLogLoss(predicted4, lev = levels(predicted4$obs))
```

## Logistic model tree

There is one classification model called logistic model tree, it
combines logistic regression and decision tree – instead of having
constants at leaves for prediction as the ordinary decision trees, a
logistic model tree has logistic regression models at its leaves to
provide prediction locally. The initial tree is built by creating a
standard classification tree, and afterwards building a logistic
regression model at every node trained on the set of examples at that
node. Then, we further split a node and want to build the logistic
regression function at one of the child nodes. Since we have already fit
a logistic regression at the parent node, it is reasonable to use it as
a basis for fitting the logistic regression at the child. We expect that
the parameters of the model at the parent node already encode ‘global’
influences of some attributes on the class variable; at the child node,
the model can be further refined by taking into account influences of
attributes that are only valid locally, i.e. within the set of training
examples associated with the child node.

The logistic model tree is fitted by the LogitBoost algorithm which
iteratively changes the logistic regression at chile node to improve the
fit to the data by changing one of the coefficients in the linear
function or introducing a new variable/coefficient pair. At some point,
adding more variables does not increase the accuracy of the model, but
splitting the instance space and refining the logistic models locally in
the two subdivisions created by the split might give a better model.
After splitting a node we can continue running LogitBoost iterations for
fitting the logsitc regression model to the response variables of the
training examples at the child node.

Thus, we tune logistic model tree by giving iteration number. For our
training set, we use the `LMT` method in `train` function and set the
iteration number to 1 to 3.

``` r
log_tr <- train(diabetes_dx ~ HighBP + HighChol + CholCheck + BMI + Smoker + Stroke + HeartDiseaseorAttack 
                  + PhysActivity + Fruits + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth + MentHlth 
                  + PhysHlth + DiffWalk + Sex + Age + Income, 
                data = train,
                method = "LMT",
                metric="logLoss",
                preProcess = c("center", "scale"),
                tuneGrid = data.frame(iter = c(1:3)),
                trControl = trainControl(method = "cv", number = 5, classProbs=TRUE, summaryFunction=mnLogLoss))
log_tr
```

    ## Logistic Model Trees 
    ## 
    ## 122 samples
    ##  20 predictor
    ##   2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (40), scaled (40) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 98, 98, 98, 97, 97 
    ## Resampling results across tuning parameters:
    ## 
    ##   iter  logLoss  
    ##   1     0.6404063
    ##   2     0.6628140
    ##   3     0.9038585
    ## 
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final value used for the model was iter = 1.

``` r
#apply the best model on the test set and merge the predicted results with the true response into one data frame
predicted5 <- data.frame(obs=test$diabetes_dx,
             pred=predict(log_tr, test),
             predict(log_tr, test, type="prob"))

#calculate the log-loss

e <- mnLogLoss(predicted5, lev = levels(predicted5$obs))
```

# Final model selection

Now, best models are chosen for each model type, we are going to compare
all six models log-loss values from running on the test set then pick
the final winner. Just like previous step, we will create a data frame
with all six log-loss values and their corresponding model names, then
return the row with the lowest log-loss value by using the `which.min`
function. Finally, we will insert the model name as an inline R code
into the conclusion.

``` r
final_pick <- data.frame(logloss_value = c(d, e), model = c("Random forest model", "Logistic model tree"))
#append with previous best logistic model
final_pick <- rbind(final_pick, log_best)
final_pick
```

    ## # A tibble: 3 × 2
    ##   logloss_value model                         
    ##           <dbl> <chr>                         
    ## 1         0.412 Random forest model           
    ## 2         0.574 Logistic model tree           
    ## 3         0.379 Logistc regression model three

``` r
#return the smallest log-loss model
final_best <- final_pick[which.min(final_pick$logloss_value),]
final_best
```

    ## # A tibble: 1 × 2
    ##   logloss_value model                         
    ##           <dbl> <chr>                         
    ## 1         0.379 Logistc regression model three

Finally, after comparing all six models’ log-loss values fitting on the
test set, the Logistc regression model three has the lowest log-loss
value (0.3791432), thus, the Logistc regression model three is the best
model for predicting diabetes diagnosis.
