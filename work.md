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
diabetes$Education  <- as.factor(diabetes$Education)
diabetes$Income <- as.factor(diabetes$Income)
diabetes              
```

    ## # A tibble: 253,680 × 23
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
    ## # ℹ 16 more variables: HeartDiseaseorAttack <fct>, PhysActivity <fct>,
    ## #   Fruits <fct>, Veggies <fct>, HvyAlcoholConsump <fct>, AnyHealthcare <fct>,
    ## #   NoDocbcCost <fct>, GenHlth <fct>, MentHlth <dbl>, PhysHlth <dbl>,
    ## #   DiffWalk <fct>, Sex <fct>, Age <fct>, Education <fct>, Income <fct>,
    ## #   diabetes_dx <fct>

``` r
# subset data set based on parameter in YAML header

#diabetes_sub <- diabetes %>% 

#                   filter(Education == params$Education)

diabetes_sub <- diabetes %>% 

                   filter(Education == "1")

diabetes_sub
```

    ## # A tibble: 174 × 23
    ##    Diabetes_binary HighBP HighChol CholCheck   BMI Smoker Stroke
    ##    <fct>           <fct>  <fct>    <fct>     <dbl> <fct>  <fct> 
    ##  1 1               1      1        1            42 0      0     
    ##  2 0               1      0        1            23 1      0     
    ##  3 0               0      0        1            22 0      0     
    ##  4 0               0      0        1            20 0      0     
    ##  5 1               1      1        1            29 0      0     
    ##  6 1               1      0        1            35 0      1     
    ##  7 1               0      1        1            28 1      0     
    ##  8 0               0      1        1            35 0      0     
    ##  9 0               1      1        1            52 0      0     
    ## 10 1               1      0        1            32 0      0     
    ## # ℹ 164 more rows
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
    ##   0   1 
    ## 127  47

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
    ##      0  1
    ##   0 74 53
    ##   1 12 35
    ## 
    ## [[2]]
    ##    0  1         0         1
    ## 0 74 53 0.5826772 0.4173228
    ## 1 12 35 0.2553191 0.7446809
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 14.707, df = 1, p-value = 0.0001256

``` r
chisq(diabetes_sub$HighChol)
```

    ## [[1]]
    ##    x
    ##      0  1
    ##   0 70 57
    ##   1 17 30
    ## 
    ## [[2]]
    ##    0  1         0         1
    ## 0 70 57 0.5511811 0.4488189
    ## 1 17 30 0.3617021 0.6382979
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 4.9265, df = 1, p-value = 0.02645

``` r
chisq(diabetes_sub$CholCheck)
```

    ## [[1]]
    ##    x
    ##       0   1
    ##   0   7 120
    ##   1   0  47
    ## 
    ## [[2]]
    ##   0   1          0         1
    ## 0 7 120 0.05511811 0.9448819
    ## 1 0  47 0.00000000 1.0000000
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 2.6991, df = 1, p-value = 0.1004

``` r
chisq(diabetes_sub$Smoker)
```

    ## [[1]]
    ##    x
    ##      0  1
    ##   0 78 49
    ##   1 30 17
    ## 
    ## [[2]]
    ##    0  1         0         1
    ## 0 78 49 0.6141732 0.3858268
    ## 1 30 17 0.6382979 0.3617021
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 0.084802, df = 1, p-value = 0.7709

``` r
chisq(diabetes_sub$Stroke)
```

    ## [[1]]
    ##    x
    ##       0   1
    ##   0 122   5
    ##   1  38   9
    ## 
    ## [[2]]
    ##     0 1         0          1
    ## 0 122 5 0.9606299 0.03937008
    ## 1  38 9 0.8085106 0.19148936
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 10.729, df = 1, p-value = 0.001055

``` r
chisq(diabetes_sub$HeartDiseaseorAttack)
```

    ## [[1]]
    ##    x
    ##       0   1
    ##   0 117  10
    ##   1  28  19
    ## 
    ## [[2]]
    ##     0  1         0          1
    ## 0 117 10 0.9212598 0.07874016
    ## 1  28 19 0.5957447 0.40425532
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 26.171, df = 1, p-value = 3.124e-07

``` r
chisq(diabetes_sub$PhysActivity)
```

    ## [[1]]
    ##    x
    ##      0  1
    ##   0 59 68
    ##   1 20 27
    ## 
    ## [[2]]
    ##    0  1         0         1
    ## 0 59 68 0.4645669 0.5354331
    ## 1 20 27 0.4255319 0.5744681
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 0.21087, df = 1, p-value = 0.6461

``` r
chisq(diabetes_sub$Fruits)
```

    ## [[1]]
    ##    x
    ##      0  1
    ##   0 51 76
    ##   1 13 34
    ## 
    ## [[2]]
    ##    0  1         0         1
    ## 0 51 76 0.4015748 0.5984252
    ## 1 13 34 0.2765957 0.7234043
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 2.3044, df = 1, p-value = 0.129

``` r
chisq(diabetes_sub$Veggies)
```

    ## [[1]]
    ##    x
    ##      0  1
    ##   0 35 92
    ##   1 12 35
    ## 
    ## [[2]]
    ##    0  1         0         1
    ## 0 35 92 0.2755906 0.7244094
    ## 1 12 35 0.2553191 0.7446809
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 0.071502, df = 1, p-value = 0.7892

``` r
chisq(diabetes_sub$HvyAlcoholConsump)
```

    ## [[1]]
    ##    x
    ##       0   1
    ##   0 120   7
    ##   1  47   0
    ## 
    ## [[2]]
    ##     0 1         0          1
    ## 0 120 7 0.9448819 0.05511811
    ## 1  47 0 1.0000000 0.00000000
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 2.6991, df = 1, p-value = 0.1004

``` r
chisq(diabetes_sub$AnyHealthcare)
```

    ## [[1]]
    ##    x
    ##      0  1
    ##   0 32 95
    ##   1  5 42
    ## 
    ## [[2]]
    ##    0  1         0         1
    ## 0 32 95 0.2519685 0.7480315
    ## 1  5 42 0.1063830 0.8936170
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 4.3428, df = 1, p-value = 0.03717

``` r
chisq(diabetes_sub$NoDocbcCost)
```

    ## [[1]]
    ##    x
    ##      0  1
    ##   0 96 31
    ##   1 37 10
    ## 
    ## [[2]]
    ##    0  1         0         1
    ## 0 96 31 0.7559055 0.2440945
    ## 1 37 10 0.7872340 0.2127660
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 0.18694, df = 1, p-value = 0.6655

``` r
chisq(diabetes_sub$GenHlth)
```

    ## [[1]]
    ##    x
    ##      1  2  3  4  5
    ##   0  8 26 37 41 15
    ##   1  1  0 12 15 19
    ## 
    ## [[2]]
    ##   1  2  3  4  5          1         2         3         4         5
    ## 0 8 26 37 41 15 0.06299213 0.2047244 0.2913386 0.3228346 0.1181102
    ## 1 1  0 12 15 19 0.02127660 0.0000000 0.2553191 0.3191489 0.4042553
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 25.31, df = 4, p-value = 4.358e-05

``` r
chisq(diabetes_sub$DiffWalk)
```

    ## [[1]]
    ##    x
    ##      0  1
    ##   0 94 33
    ##   1 20 27
    ## 
    ## [[2]]
    ##    0  1         0         1
    ## 0 94 33 0.7401575 0.2598425
    ## 1 20 27 0.4255319 0.5744681
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 15.031, df = 1, p-value = 0.0001058

``` r
chisq(diabetes_sub$Sex)
```

    ## [[1]]
    ##    x
    ##      0  1
    ##   0 72 55
    ##   1 30 17
    ## 
    ## [[2]]
    ##    0  1         0         1
    ## 0 72 55 0.5669291 0.4330709
    ## 1 30 17 0.6382979 0.3617021
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 0.72033, df = 1, p-value = 0.396

``` r
chisq(diabetes_sub$Income)
```

    ## [[1]]
    ##    x
    ##      1  2  3  4  5  6  7  8
    ##   0 23 15 19 13 19 16 12 10
    ##   1 14 10  9  5  3  2  1  3
    ## 
    ## [[2]]
    ##    1  2  3  4  5  6  7  8         1         2         3         4          5
    ## 0 23 15 19 13 19 16 12 10 0.1811024 0.1181102 0.1496063 0.1023622 0.14960630
    ## 1 14 10  9  5  3  2  1  3 0.2978723 0.2127660 0.1914894 0.1063830 0.06382979
    ##            6          7          8
    ## 0 0.12598425 0.09448819 0.07874016
    ## 1 0.04255319 0.02127660 0.06382979
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 11.586, df = 7, p-value = 0.115

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
    ## 1 0                29.0               7.03     49.4     28  24      32
    ## 2 1                31.8               7.62     58.1     29  26.5    35

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
    ## t = -2.1514, df = 76.704, p-value = 0.03459
    ## alternative hypothesis: true difference in means between group 0 and group 1 is not equal to 0
    ## 95 percent confidence interval:
    ##  -5.2806372 -0.2040336
    ## sample estimates:
    ## mean in group 0 mean in group 1 
    ##        29.02362        31.76596

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
val ues are coded as “0” and “1”.

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
test set’s true response. Lastly, use the `mnLogLoss` function to
compare their performance with a log-loss method.

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
    ## 122 samples
    ##  20 predictor
    ##   2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (40), scaled (40) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 98, 97, 97, 99, 97 
    ## Resampling results:
    ## 
    ##   logLoss 
    ##   8.566273

``` r
summary(full_log)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Coefficients:
    ##                         Estimate Std. Error z value Pr(>|z|)  
    ## (Intercept)             -8.48524  714.25805  -0.012   0.9905  
    ## HighBP1                  0.22703    0.47089   0.482   0.6297  
    ## HighChol1                0.12634    0.43168   0.293   0.7698  
    ## CholCheck1               4.35733 1050.82922   0.004   0.9967  
    ## BMI                      0.38402    0.48482   0.792   0.4283  
    ## Smoker1                 -1.11111    0.63121  -1.760   0.0784 .
    ## Stroke1                  0.46451    0.44278   1.049   0.2941  
    ## HeartDiseaseorAttack1    0.66317    0.47235   1.404   0.1603  
    ## PhysActivity1            0.03967    0.44184   0.090   0.9285  
    ## Fruits1                  0.76868    0.52426   1.466   0.1426  
    ## Veggies1                 0.14393    0.40478   0.356   0.7222  
    ## HvyAlcoholConsump1      -2.26260 1220.00808  -0.002   0.9985  
    ## AnyHealthcare1           0.62352    0.63012   0.990   0.3224  
    ## NoDocbcCost1            -0.35793    0.56212  -0.637   0.5243  
    ## GenHlth2                -5.77725 1148.63246  -0.005   0.9960  
    ## GenHlth3                 2.13947    1.27344   1.680   0.0929 .
    ## GenHlth4                 1.21288    1.21084   1.002   0.3165  
    ## GenHlth5                 2.77698    1.25587   2.211   0.0270 *
    ## MentHlth                -0.88173    0.54722  -1.611   0.1071  
    ## PhysHlth                -0.18824    0.65066  -0.289   0.7723  
    ## DiffWalk1                0.10750    0.63309   0.170   0.8652  
    ## Sex1                     0.75312    0.54469   1.383   0.1668  
    ## Age2                    -0.08394 1631.57886   0.000   1.0000  
    ## Age3                    -4.17442 1190.79657  -0.004   0.9972  
    ## Age4                    -4.66224 1222.21595  -0.004   0.9970  
    ## Age5                    -5.55840 1166.71669  -0.005   0.9962  
    ## Age6                    -1.50949    1.04595  -1.443   0.1490  
    ## Age7                    -1.27042    0.95433  -1.331   0.1831  
    ## Age8                    -0.27795    0.91994  -0.302   0.7626  
    ## Age9                    -0.74675    1.15180  -0.648   0.5168  
    ## Age10                   -0.82413    0.97488  -0.845   0.3979  
    ## Age11                   -0.38343    0.96094  -0.399   0.6899  
    ## Age12                   -0.41293    0.97379  -0.424   0.6715  
    ## Age13                   -1.55336    0.94572  -1.643   0.1005  
    ## Income2                  0.30384    0.52795   0.576   0.5649  
    ## Income3                  0.04880    0.48994   0.100   0.9207  
    ## Income4                  0.08258    0.60041   0.138   0.8906  
    ## Income5                 -0.14612    0.52143  -0.280   0.7793  
    ## Income6                 -0.68655    0.53630  -1.280   0.2005  
    ## Income7                 -1.15446    0.71175  -1.622   0.1048  
    ## Income8                 -0.51427    0.46543  -1.105   0.2692  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 142.434  on 121  degrees of freedom
    ## Residual deviance:  62.193  on  81  degrees of freedom
    ## AIC: 144.19
    ## 
    ## Number of Fisher Scoring iterations: 19

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
    ## 122 samples
    ##  20 predictor
    ##   2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (48), scaled (48) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 97, 98, 99, 97, 97 
    ## Resampling results:
    ## 
    ##   logLoss 
    ##   13.39856

``` r
summary(inter_log)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Coefficients:
    ##                         Estimate Std. Error z value Pr(>|z|)  
    ## (Intercept)            -10.40660  678.83338  -0.015   0.9878  
    ## HighBP1                  5.57090    4.14541   1.344   0.1790  
    ## HighChol1                4.71616    3.37935   1.396   0.1628  
    ## CholCheck1               3.27181 1075.34359   0.003   0.9976  
    ## BMI                     23.73232   57.33426   0.414   0.6789  
    ## Smoker1                 -2.22027    1.11556  -1.990   0.0466 *
    ## Stroke1                  0.51169    0.64650   0.791   0.4287  
    ## HeartDiseaseorAttack1    1.15111    0.91649   1.256   0.2091  
    ## PhysActivity1           -2.26370    2.88099  -0.786   0.4320  
    ## Fruits1                  1.28489    0.84765   1.516   0.1296  
    ## Veggies1                 0.17141    0.57057   0.300   0.7639  
    ## HvyAlcoholConsump1      -1.61495 1220.30371  -0.001   0.9989  
    ## AnyHealthcare1          -0.64518    0.99904  -0.646   0.5184  
    ## NoDocbcCost1            -0.37881    0.86746  -0.437   0.6623  
    ## GenHlth2                30.16512 6341.03794   0.005   0.9962  
    ## GenHlth3                50.69781  114.32488   0.443   0.6574  
    ## GenHlth4                40.58450  114.67132   0.354   0.7234  
    ## GenHlth5                36.23737   93.14900   0.389   0.6973  
    ## MentHlth                -1.73296    0.97627  -1.775   0.0759 .
    ## PhysHlth                 0.02479    0.94575   0.026   0.9791  
    ## DiffWalk1               -0.47826    1.24836  -0.383   0.7016  
    ## Sex1                     0.97595    0.80655   1.210   0.2263  
    ## Age2                    -0.33547 1643.69632   0.000   0.9998  
    ## Age3                    -5.51754 1084.62837  -0.005   0.9959  
    ## Age4                    -5.47748 1265.74294  -0.004   0.9965  
    ## Age5                    -7.67809 1066.74790  -0.007   0.9943  
    ## Age6                    -3.58472    3.83586  -0.935   0.3500  
    ## Age7                    -3.83291    3.83746  -0.999   0.3179  
    ## Age8                    -1.57327    3.98348  -0.395   0.6929  
    ## Age9                    -1.74459    4.53018  -0.385   0.7002  
    ## Age10                   -2.79640    4.01446  -0.697   0.4861  
    ## Age11                   -1.37209    4.04603  -0.339   0.7345  
    ## Age12                   -1.92879    3.96927  -0.486   0.6270  
    ## Age13                   -2.33436    3.21988  -0.725   0.4685  
    ## Income2                  0.18681    0.69982   0.267   0.7895  
    ## Income3                 -0.89796    1.10238  -0.815   0.4153  
    ## Income4                  0.18770    0.89366   0.210   0.8336  
    ## Income5                 -0.86367    0.79864  -1.081   0.2795  
    ## Income6                 -1.79642    0.99574  -1.804   0.0712 .
    ## Income7                 -2.84565    1.49982  -1.897   0.0578 .
    ## Income8                 -0.12144    0.58414  -0.208   0.8353  
    ## `HighBP1:HighChol1`     -3.68251    1.72971  -2.129   0.0333 *
    ## `HighBP1:BMI`           -2.89863    4.20908  -0.689   0.4910  
    ## `HighChol1:BMI`         -1.56475    3.48229  -0.449   0.6532  
    ## `BMI:PhysActivity1`      2.02444    2.74963   0.736   0.4616  
    ## `BMI:GenHlth2`         -29.46338 6550.60534  -0.004   0.9964  
    ## `BMI:GenHlth3`         -48.42817  115.71756  -0.419   0.6756  
    ## `BMI:GenHlth4`         -37.00119  108.21560  -0.342   0.7324  
    ## `BMI:GenHlth5`         -34.73492   96.86088  -0.359   0.7199  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 142.434  on 121  degrees of freedom
    ## Residual deviance:  45.934  on  73  degrees of freedom
    ## AIC: 143.93
    ## 
    ## Number of Fisher Scoring iterations: 19

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
    ## 122 samples
    ##  20 predictor
    ##   2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (43), scaled (43) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 98, 97, 97, 98, 98 
    ## Resampling results:
    ## 
    ##   logLoss 
    ##   10.10324

``` r
summary(poly_log)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Coefficients:
    ##                         Estimate Std. Error z value Pr(>|z|)  
    ## (Intercept)            -10.82825  661.47278  -0.016   0.9869  
    ## HighBP1                  1.15026    0.76221   1.509   0.1313  
    ## HighChol1                0.73976    0.63241   1.170   0.2421  
    ## CholCheck1               4.55441  924.00008   0.005   0.9961  
    ## `I(BMI^2)`             -10.81510    5.61832  -1.925   0.0542 .
    ## BMI                     12.74196    6.29849   2.023   0.0431 *
    ## Smoker1                 -2.20158    1.40532  -1.567   0.1172  
    ## Stroke1                  0.60018    0.79907   0.751   0.4526  
    ## HeartDiseaseorAttack1    1.02658    1.00042   1.026   0.3048  
    ## PhysActivity1           -0.32008    0.72059  -0.444   0.6569  
    ## Fruits1                  2.10006    1.03454   2.030   0.0424 *
    ## Veggies1                 1.02849    0.75254   1.367   0.1717  
    ## HvyAlcoholConsump1       0.25023 1338.47468   0.000   0.9999  
    ## AnyHealthcare1           0.05971    0.75681   0.079   0.9371  
    ## NoDocbcCost1            -1.55311    0.99609  -1.559   0.1189  
    ## GenHlth2                -5.04554 1006.05813  -0.005   0.9960  
    ## GenHlth3                 4.13078    2.29390   1.801   0.0717 .
    ## GenHlth4                 2.10428    1.90964   1.102   0.2705  
    ## GenHlth5                 4.80937    2.00377   2.400   0.0164 *
    ## `I(MentHlth^2)`         -6.09461    5.53890  -1.100   0.2712  
    ## MentHlth                 4.24175    5.13130   0.827   0.4084  
    ## `I(PhysHlth^2)`        -12.61604    5.53029  -2.281   0.0225 *
    ## PhysHlth                12.23013    5.51733   2.217   0.0266 *
    ## DiffWalk1               -2.27557    1.63688  -1.390   0.1645  
    ## Sex1                     1.72484    0.99087   1.741   0.0817 .
    ## Age2                     0.36552 1625.48237   0.000   0.9998  
    ## Age3                    -5.24928 1027.38325  -0.005   0.9959  
    ## Age4                    -4.77073 1251.05760  -0.004   0.9970  
    ## Age5                    -6.62259 1084.68478  -0.006   0.9951  
    ## Age6                    -2.40955    1.43322  -1.681   0.0927 .
    ## Age7                    -1.96566    1.44860  -1.357   0.1748  
    ## Age8                     0.82710    1.28887   0.642   0.5211  
    ## Age9                    -0.35301    1.31253  -0.269   0.7880  
    ## Age10                   -0.13126    1.14219  -0.115   0.9085  
    ## Age11                    0.85624    1.11033   0.771   0.4406  
    ## Age12                    0.63938    1.31363   0.487   0.6264  
    ## Age13                   -0.75043    1.22212  -0.614   0.5392  
    ## Income2                  0.70646    0.80778   0.875   0.3818  
    ## Income3                 -0.45885    0.70362  -0.652   0.5143  
    ## Income4                  1.70968    1.05962   1.613   0.1066  
    ## Income5                 -1.12007    0.99498  -1.126   0.2603  
    ## Income6                 -2.59103    1.33057  -1.947   0.0515 .
    ## Income7                 -3.06783    1.51831  -2.021   0.0433 *
    ## Income8                 -1.78161    1.13581  -1.569   0.1167  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 142.434  on 121  degrees of freedom
    ## Residual deviance:  43.962  on  78  degrees of freedom
    ## AIC: 131.96
    ## 
    ## Number of Fisher Scoring iterations: 19

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
    ## 1          1.05 Logistic regression model one

From the result we can tell, the Logistic regression model one has the
lowest log-loss value (1.0520725), thus, the Logistic regression model
one is the best logistic regression model for predicting diabetes
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
                tuneGrid = data.frame(mtry = c(18:20)),
                trControl = trainControl(method = "cv", number = 5, classProbs=TRUE, summaryFunction=mnLogLoss))
ran_for
```

    ## Random Forest 
    ## 
    ## 122 samples
    ##  20 predictor
    ##   2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (40), scaled (40) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 97, 99, 97, 98, 97 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  logLoss  
    ##   18    0.5853624
    ##   19    0.5862596
    ##   20    0.5971445
    ## 
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 18.

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
