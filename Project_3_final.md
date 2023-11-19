Project 3
================
Vivi Feathers & Siyuan Su
2023-11-02

- [Introduction](#introduction)
- [Data](#data)
- [Summarizations](#summarizations)
  - [The response: “Diabetes_binary”](#the-response-diabetes_binary)
  - [Contigency table and Chi-square](#contigency-table-and-chi-square)
  - [“HighBP”](#highbp)
  - [“HighChol”](#highchol)
  - [“BMI”](#bmi)
  - [Grid of bar and density plots](#grid-of-bar-and-density-plots)
  - [Correlation among numeric
    variables](#correlation-among-numeric-variables)
- [Modeling](#modeling)
  - [Training and test set](#training-and-test-set)
  - [Log-Loss](#log-loss)
  - [Logistic regression](#logistic-regression)
  - [LASSO logistic regression](#lasso-logistic-regression)
  - [Classifcation tree model](#classifcation-tree-model)
  - [Random forest](#random-forest)
  - [Logistic model tree](#logistic-model-tree)
  - [Linear discriminant analysis](#linear-discriminant-analysis)
- [Final model selection](#final-model-selection)

# Introduction

Diabetes are one of the most prevalent chronic disease that are facing
people in the United States as well as in the world. It is estimated
that as large as 11.6% of the US population have diabetes, more than 1/3
of the US population have pre-diabetes. Diabetes is a serious disease in
which individuals lose the ability to effectively regulate levels of
glucose in the blood, and could eventually contribute to complications
like heart disease, vision loss, lower-limb amputation and kidney
diseases. According to the CDC website
(<https://www.cdc.gov/diabetes/>), At age 50, life expectancy is 6 years
shorter for people with type 2 diabetes than for people without it.

Although there are multiple treatment regime for diabetic individual to
effectively control their blood sugar level, there is no cure for the
disease. To make things worse, about 25% of diabetic individuals were
not diagnosed. So there are urgent unmet medical needs to 1. Find
effective preventative measurement to prevent individuals from
developing diabetes and 2. Predict potential pre-diabetic/diabetic
individuals from their life-style related information before they get
diagnosed at a hospital. 3. Find effective measures to help individuals
with diabetic to live a quality life. Making meaningful insight by
exploring data analysis and making prediction using large cohort
comprehensive health data set could help provide solution to the needs
mentioned above.

The data set we’ll be exploring is from The Behavioral Risk Factor
Surveillance System (BRFSS), a health-related telephone survey that is
collected annually by the CDC. The response is a binary variable that
has 2 classes. 0 is for no diabetes, and 1 is for pre-diabetes or
diabetes. There are also 21 variables collected in the data set that
might have some association with the response.

In order to learn the distribution of those variables and their
relationship with each other, we will conduct a series of data
exploratory analysis and generate contingency tables, summary tables,
chi-square tests, bar plots, density plots and correlation chart.

After getting some general understanding about the data, we are planning
on spitting the data set into training and test set, fitting some
prediction models on the training set with different model types and
turning parameters, also utilizing the cross-validation method for model
selection, and log-loss for model performance evaluation. Our final goal
is to find the model that returns the lowest log-loss value when applied
on the test set for prediction.

The model types that we will investigate are:

1.  Logistic regression model.  
2.  LASSO logistic regression model.  
3.  Classification tree model.  
4.  Random forest model.  
5.  Logistic model tree.  
6.  Linear discriminant analysis.

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
library(gridExtra)
library(readr)
library(scales)
library(MASS)
diabetes <- read_csv(file = "diabetes_binary_health_indicators_BRFSS2015.csv")

# for log-loss purpose, create a new variable and assign value as "YES" for the records have Diabetes_binary = 1 and 'NO' otherwise.
diabetes <- diabetes %>%
            mutate(diabetes_dx = as.factor(if_else(Diabetes_binary == 1, "Yes", "No")))
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
diabetes_sub <- diabetes %>% 
                   filter(Education == params$Education)
```

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
    ##    0    1 
    ## 2987 1230

``` r
g <- ggplot(data = diabetes_sub, aes(x = Diabetes_binary, fill = Diabetes_binary))
g + geom_bar(alpha = 0.6) +
  scale_x_discrete(breaks=c("0","1"),
        labels=c("No", "Yes")) +
  scale_fill_discrete(name = "Diabetes Diagnosis", labels = c("No", "Yes")) +
  labs(x = "Diabetes Diagnosis", y = "Subject count", title = "Bar plot of subject with diabetes and subject withou diabetes count") 
```

![](Project_3_final_files/figure-gfm/one-1.png)<!-- -->

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
    ##        0    1
    ##   0 1472 1515
    ##   1  268  962
    ## 
    ## [[2]]
    ##      0    1         0         1
    ## 0 1472 1515 0.4928021 0.5071979
    ## 1  268  962 0.2178862 0.7821138
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 271.69, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$HighChol)
```

    ## [[1]]
    ##    x
    ##        0    1
    ##   0 1611 1376
    ##   1  354  876
    ## 
    ## [[2]]
    ##      0    1         0         1
    ## 0 1611 1376 0.5393371 0.4606629
    ## 1  354  876 0.2878049 0.7121951
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 221.51, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$CholCheck)
```

    ## [[1]]
    ##    x
    ##        0    1
    ##   0  111 2876
    ##   1    8 1222
    ## 
    ## [[2]]
    ##     0    1           0         1
    ## 0 111 2876 0.037161031 0.9628390
    ## 1   8 1222 0.006504065 0.9934959
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 29.86, df = 1, p-value = 4.645e-08

``` r
chisq(diabetes_sub$Smoker)
```

    ## [[1]]
    ##    x
    ##        0    1
    ##   0 1575 1412
    ##   1  616  614
    ## 
    ## [[2]]
    ##      0    1         0         1
    ## 0 1575 1412 0.5272849 0.4727151
    ## 1  616  614 0.5008130 0.4991870
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 2.4459, df = 1, p-value = 0.1178

``` r
chisq(diabetes_sub$Stroke)
```

    ## [[1]]
    ##    x
    ##        0    1
    ##   0 2777  210
    ##   1 1080  150
    ## 
    ## [[2]]
    ##      0   1         0          1
    ## 0 2777 210 0.9296953 0.07030465
    ## 1 1080 150 0.8780488 0.12195122
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 29.763, df = 1, p-value = 4.883e-08

``` r
chisq(diabetes_sub$HeartDiseaseorAttack)
```

    ## [[1]]
    ##    x
    ##        0    1
    ##   0 2557  430
    ##   1  853  377
    ## 
    ## [[2]]
    ##      0   1         0         1
    ## 0 2557 430 0.8560429 0.1439571
    ## 1  853 377 0.6934959 0.3065041
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 148.76, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$PhysActivity)
```

    ## [[1]]
    ##    x
    ##        0    1
    ##   0 1230 1757
    ##   1  591  639
    ## 
    ## [[2]]
    ##      0    1         0         1
    ## 0 1230 1757 0.4117844 0.5882156
    ## 1  591  639 0.4804878 0.5195122
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 16.761, df = 1, p-value = 4.239e-05

``` r
chisq(diabetes_sub$Fruits)
```

    ## [[1]]
    ##    x
    ##        0    1
    ##   0 1247 1740
    ##   1  538  692
    ## 
    ## [[2]]
    ##      0    1         0         1
    ## 0 1247 1740 0.4174757 0.5825243
    ## 1  538  692 0.4373984 0.5626016
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 1.4166, df = 1, p-value = 0.234

``` r
chisq(diabetes_sub$Veggies)
```

    ## [[1]]
    ##    x
    ##        0    1
    ##   0  889 2098
    ##   1  406  824
    ## 
    ## [[2]]
    ##     0    1         0         1
    ## 0 889 2098 0.2976230 0.7023770
    ## 1 406  824 0.3300813 0.6699187
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 4.3136, df = 1, p-value = 0.03781

``` r
chisq(diabetes_sub$HvyAlcoholConsump)
```

    ## [[1]]
    ##    x
    ##        0    1
    ##   0 2888   99
    ##   1 1217   13
    ## 
    ## [[2]]
    ##      0  1         0          1
    ## 0 2888 99 0.9668564 0.03314362
    ## 1 1217 13 0.9894309 0.01056911
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 17.173, df = 1, p-value = 3.412e-05

``` r
chisq(diabetes_sub$AnyHealthcare)
```

    ## [[1]]
    ##    x
    ##        0    1
    ##   0  546 2441
    ##   1  126 1104
    ## 
    ## [[2]]
    ##     0    1         0         1
    ## 0 546 2441 0.1827921 0.8172079
    ## 1 126 1104 0.1024390 0.8975610
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 41.992, df = 1, p-value = 9.166e-11

``` r
chisq(diabetes_sub$NoDocbcCost)
```

    ## [[1]]
    ##    x
    ##        0    1
    ##   0 2453  534
    ##   1 1010  220
    ## 
    ## [[2]]
    ##      0   1         0         1
    ## 0 2453 534 0.8212253 0.1787747
    ## 1 1010 220 0.8211382 0.1788618
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 4.5013e-05, df = 1, p-value = 0.9946

``` r
chisq(diabetes_sub$GenHlth)
```

    ## [[1]]
    ##    x
    ##        1    2    3    4    5
    ##   0  243  399 1010  924  411
    ##   1   32   62  255  510  371
    ## 
    ## [[2]]
    ##     1   2    3   4   5          1         2         3         4         5
    ## 0 243 399 1010 924 411 0.08135253 0.1335788 0.3381319 0.3093405 0.1375963
    ## 1  32  62  255 510 371 0.02601626 0.0504065 0.2073171 0.4146341 0.3016260
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 300.56, df = 4, p-value < 2.2e-16

``` r
chisq(diabetes_sub$DiffWalk)
```

    ## [[1]]
    ##    x
    ##        0    1
    ##   0 2039  948
    ##   1  575  655
    ## 
    ## [[2]]
    ##      0   1         0         1
    ## 0 2039 948 0.6826247 0.3173753
    ## 1  575 655 0.4674797 0.5325203
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 171.15, df = 1, p-value < 2.2e-16

``` r
chisq(diabetes_sub$Sex)
```

    ## [[1]]
    ##    x
    ##        0    1
    ##   0 1576 1411
    ##   1  707  523
    ## 
    ## [[2]]
    ##      0    1         0         1
    ## 0 1576 1411 0.5276197 0.4723803
    ## 1  707  523 0.5747967 0.4252033
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 7.8099, df = 1, p-value = 0.005196

``` r
chisq(diabetes_sub$Income)
```

    ## [[1]]
    ##    x
    ##       1   2   3   4   5   6   7   8
    ##   0 588 488 552 473 373 252 131 130
    ##   1 349 278 216 150 127  59  26  25
    ## 
    ## [[2]]
    ##     1   2   3   4   5   6   7   8         1         2         3         4
    ## 0 588 488 552 473 373 252 131 130 0.1968530 0.1633746 0.1848008 0.1583529
    ## 1 349 278 216 150 127  59  26  25 0.2837398 0.2260163 0.1756098 0.1219512
    ##           5          6          7          8
    ## 0 0.1248745 0.08436558 0.04385671 0.04352193
    ## 1 0.1032520 0.04796748 0.02113821 0.02032520
    ## 
    ## [[3]]
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  a
    ## X-squared = 100.56, df = 7, p-value < 2.2e-16

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

![](Project_3_final_files/figure-gfm/bar1-1.png)<!-- -->

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

![](Project_3_final_files/figure-gfm/bar2-1.png)<!-- -->

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
    ## 1 0                28.7               6.75     45.5     28    24    31
    ## 2 1                31.4               7.40     54.7     30    27    35

``` r
#generate a kernal density plot to show the distribution of BMI
i <- ggplot(data = diabetes_sub, aes(x = BMI, fill = Diabetes_binary)) 
i + geom_density(adjust = 1, color="#e9ecef", alpha=0.5, position = 'dodge') +
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  labs(x = "BMI", title = "Kernal density plot of BMI distribution across diagnosis group") 
```

![](Project_3_final_files/figure-gfm/bmi-1.png)<!-- -->

``` r
#conduct a two sample t test for BMI in both diagnosis groups
t.test(BMI ~ Diabetes_binary, data = diabetes_sub, alternative = "two.sided", var.equal = FALSE)
```

    ## 
    ##  Welch Two Sample t-test
    ## 
    ## data:  BMI by Diabetes_binary
    ## t = -11.049, df = 2113.9, p-value < 2.2e-16
    ## alternative hypothesis: true difference in means between group 0 and group 1 is not equal to 0
    ## 95 percent confidence interval:
    ##  -3.179240 -2.220791
    ## sample estimates:
    ## mean in group 0 mean in group 1 
    ##        28.67559        31.37561

The summary table gives us the center (mean and median) and spread
(standard_deviation, variance, q1 and q3) of BMI in each diagnosis
group. The density plot shows the distribution of BMI in each diagnosis
group, we can estimate the means and standard deviations, also visualize
their differences. Lastly, the two sample t-test returns the 95%
confidence interval of mean difference, and the t statistic, degree of
freedom and p-value as t-test result, we need to see if the p-value is
smaller than 0.05 to decide whether we will reject the null hypothesis
(true difference in means between two diagnosis is equal to 0).

## Grid of bar and density plots

Beside above 3 variables that we investigated individually, we still
want to have a brief concept about the distribution of other variables
that were not mentioned that often in scientific articles about their
association with diabetes diagnosis. Compared with contingency tables,
bar/density plots are often visually more appealing and easier to
understand.

In order to create bar plot for visualizing the distribution of diabetic
people in each category of variables, we need to use the `ggplot`
package, and the `geom_bar` function will be applied. For numeric
variable, the `geom_density` function will be used for displaying their
distributions. As previous, Function `labs` were used to input the label
for each plot, and `scale_x_discrete` function was used to create label
for each category on the x-axis, meanwhile `scale_fill_discrete`
function was used to provide label in the legend.

``` r
PHlthplot<-ggplot(diabetes_sub,aes(x=PhysHlth,fill=Diabetes_binary))+
  geom_density(adjust=1,alpha=0.5)+
  scale_fill_discrete(name="Diabetic?", label=c("No","Yes"))+
  labs(x="Number of days physical health not good")
MHlthplot<-ggplot(diabetes_sub,aes(x=MentHlth,fill=Diabetes_binary))+
  geom_density(adjust=1,alpha=0.5)+
  scale_fill_discrete(name="Diabetic?", label=c("No","Yes"))+
  labs(x="Number of days mental health not good")
Physplot<-ggplot(diabetes_sub,aes(x=PhysActivity))+
  geom_bar(aes(fill=(Diabetes_binary)))+
  labs(x="Physically Active?")+
  scale_x_discrete(labels=c("No","Yes"))+
  scale_fill_discrete(name="Diabetic?", label=c("No","Yes"))
Fruitplot<-ggplot(diabetes_sub,aes(x=Fruits))+
  geom_bar(aes(fill=(Diabetes_binary)))+
  labs(x="Fruit lover?")+
  scale_x_discrete(labels=c("No","Yes"))+
  scale_fill_discrete(name="Diabetic?", label=c("No","Yes"))
Veggieplot<-ggplot(diabetes_sub,aes(x=Veggies))+
  geom_bar(aes(fill=(Diabetes_binary)))+
  labs(x="Veggies eater?")+
  scale_x_discrete(labels=c("No","Yes"))+
  scale_fill_discrete(name="Diabetic?", label=c("No","Yes"))
GHlthplot<-ggplot(diabetes_sub,aes(x=GenHlth))+
  geom_bar(aes(fill=(Diabetes_binary)))+
  labs(x="General Health Condition")+
  scale_x_discrete(labels=c("E","VG", "G", "F", "P"))+
  scale_fill_discrete(name="Diabetic?", label=c("No","Yes"))
Incomeplot<-ggplot(diabetes_sub,aes(x=Income))+
  geom_bar(aes(fill=(Diabetes_binary)))+
  labs(x="Annual Household Income level")+
  scale_x_discrete(labels=c("1","2", "3","4", "5","6","7", "8"))+
  scale_fill_discrete(name="Diabetic?", label=c("No","Yes"))
Alcoholplot<-ggplot(diabetes_sub,aes(x=HvyAlcoholConsump))+
  geom_bar(aes(fill=(Diabetes_binary)))+
  labs(x="Heavy Drinker?")+
  scale_x_discrete(labels=c("No","Yes"))+
  scale_fill_discrete(name="Diabetic?", label=c("No","Yes"))
```

It would be very cumbersome to view each graph independently, so we are
putting them in a tile with the `grid.arrange` function.

``` r
grid.arrange(Physplot,Fruitplot,Veggieplot,GHlthplot,Alcoholplot,Incomeplot,PHlthplot,MHlthplot,ncol=3)
```

![](Project_3_final_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

The contrasting color red (no diabetes) and blue (have diabetes)
provides a straightforward way to visualize the how the ratio of people
who have diabetes are affected by the value of variables. Note that
different variables may have interactions, if interested we could
further split the value of a variable by another variable, just like
what will be shown in the next graph.

``` r
diabetes_Edu1_nofruit<-diabetes_sub%>%filter(Fruits=="0")
diabetes_Edu1_yesfruit<-diabetes_sub%>%filter(Fruits=="1")
diabetes_Edu1_noveggie<-diabetes_sub%>%filter(Veggies=="0")
diabetes_Edu1_yesveggie<-diabetes_sub%>%filter(Veggies=="1")
Physnofruitplot<-ggplot(diabetes_Edu1_nofruit,aes(x=PhysActivity))+
  geom_bar(aes(fill=(Diabetes_binary)))+
  ggtitle("Fruit hater")+
  labs(x="Physically Active?")+
  scale_x_discrete(labels=c("No","Yes"))+
  scale_fill_discrete(name="Diabetic?", label=c("No","Yes"))
Physyesfruitplot<-ggplot(diabetes_Edu1_yesfruit,aes(x=PhysActivity))+
  geom_bar(aes(fill=(Diabetes_binary)))+
  ggtitle("Fruit lover")+
  labs(x="Physically Active?")+
  scale_x_discrete(labels=c("No","Yes"))+
  scale_fill_discrete(name="Diabetic?", label=c("No","Yes"))
Physnoveggieplot<-ggplot(diabetes_Edu1_noveggie,aes(x=PhysActivity))+
  geom_bar(aes(fill=(Diabetes_binary)))+
  ggtitle("Veggie hater")+
  labs(x="Physically Active?")+
  scale_x_discrete(labels=c("No","Yes"))+
  scale_fill_discrete(name="Diabetic?", label=c("No","Yes"))
Physyesveggieplot<-ggplot(diabetes_Edu1_yesveggie,aes(x=PhysActivity))+
  geom_bar(aes(fill=(Diabetes_binary)))+
  ggtitle("Veggie lover")+
  labs(x="Physically Active?")+
  scale_x_discrete(labels=c("No","Yes"))+
  scale_fill_discrete(name="Diabetic?", label=c("No","Yes"))
```

``` r
grid.arrange(Physnofruitplot,Physyesfruitplot,Physnoveggieplot,Physyesveggieplot)
```

![](Project_3_final_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

This plot panel showed the effect of physical activity on diabetes
status in people who eat/do not eat fruits or veggies. Compared with
diabetes patients, is there higher ratio of physically active people in
the non-diabetes group that are also fruit & vegetable lovers?

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

![](Project_3_final_files/figure-gfm/graphics-1.png)<!-- -->

Let’s look for the pairs that have correlation coefficient higher than
0.4. If we do find such pairs, we need to refer to the correlations
between each variable and the response & other predictors, as well as
some background knowledge (may do a literature review), in order to
decide which variable will be include into the models.

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
    ## 2952 samples
    ##   20 predictor
    ##    2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (40), scaled (40) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 2362, 2361, 2362, 2361, 2362 
    ## Resampling results:
    ## 
    ##   logLoss  
    ##   0.5170152

``` r
summary(full_log)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Coefficients:
    ##                         Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)            -1.259913   2.087486  -0.604  0.54614    
    ## HighBP1                 0.340596   0.053528   6.363 1.98e-10 ***
    ## HighChol1               0.319992   0.049839   6.420 1.36e-10 ***
    ## CholCheck1              0.190495   0.074469   2.558  0.01053 *  
    ## BMI                     0.305487   0.046646   6.549 5.79e-11 ***
    ## Smoker1                -0.064281   0.049448  -1.300  0.19361    
    ## Stroke1                -0.006276   0.043122  -0.146  0.88428    
    ## HeartDiseaseorAttack1   0.198650   0.044175   4.497 6.90e-06 ***
    ## PhysActivity1          -0.070542   0.046600  -1.514  0.13008    
    ## Fruits1                 0.099742   0.048067   2.075  0.03798 *  
    ## Veggies1               -0.017848   0.046529  -0.384  0.70129    
    ## HvyAlcoholConsump1     -0.212436   0.064841  -3.276  0.00105 ** 
    ## AnyHealthcare1          0.066190   0.058212   1.137  0.25551    
    ## NoDocbcCost1           -0.043791   0.049812  -0.879  0.37933    
    ## GenHlth2                0.033417   0.093376   0.358  0.72044    
    ## GenHlth3                0.268769   0.121279   2.216  0.02668 *  
    ## GenHlth4                0.519132   0.123161   4.215 2.50e-05 ***
    ## GenHlth5                0.536903   0.110313   4.867 1.13e-06 ***
    ## MentHlth               -0.027667   0.049802  -0.556  0.57853    
    ## PhysHlth               -0.061232   0.056945  -1.075  0.28224    
    ## DiffWalk1               0.054229   0.052073   1.041  0.29769    
    ## Sex1                    0.040582   0.050070   0.811  0.41765    
    ## Age2                    1.354064  33.345248   0.041  0.96761    
    ## Age3                    2.060248  50.925524   0.040  0.96773    
    ## Age4                    2.391468  61.320305   0.039  0.96889    
    ## Age5                    2.482389  63.440749   0.039  0.96879    
    ## Age6                    3.203052  76.518530   0.042  0.96661    
    ## Age7                    3.833233  92.253330   0.042  0.96686    
    ## Age8                    3.805957  88.805971   0.043  0.96582    
    ## Age9                    4.033714  94.846816   0.043  0.96608    
    ## Age10                   4.425001 101.169644   0.044  0.96511    
    ## Age11                   4.311557  98.847093   0.044  0.96521    
    ## Age12                   4.153880  95.246085   0.044  0.96521    
    ## Age13                   4.835260 112.324476   0.043  0.96566    
    ## Income2                 0.004601   0.051849   0.089  0.92929    
    ## Income3                -0.048163   0.053514  -0.900  0.36812    
    ## Income4                -0.089272   0.055489  -1.609  0.10765    
    ## Income5                -0.037584   0.054699  -0.687  0.49202    
    ## Income6                -0.104560   0.054907  -1.904  0.05687 .  
    ## Income7                -0.051078   0.055871  -0.914  0.36060    
    ## Income8                -0.071873   0.058059  -1.238  0.21574    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 3563.9  on 2951  degrees of freedom
    ## Residual deviance: 2946.6  on 2911  degrees of freedom
    ## AIC: 3028.6
    ## 
    ## Number of Fisher Scoring iterations: 14

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
    ## 2952 samples
    ##   20 predictor
    ##    2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (48), scaled (48) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 2362, 2362, 2362, 2362, 2360 
    ## Resampling results:
    ## 
    ##   logLoss
    ##   0.51895

``` r
summary(inter_log)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Coefficients:
    ##                         Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)            -1.276304   2.070919  -0.616  0.53770    
    ## HighBP1                 0.502372   0.240489   2.089  0.03671 *  
    ## HighChol1               0.614836   0.232591   2.643  0.00821 ** 
    ## CholCheck1              0.188672   0.074718   2.525  0.01157 *  
    ## BMI                     0.314483   0.252555   1.245  0.21306    
    ## Smoker1                -0.065024   0.049575  -1.312  0.18965    
    ## Stroke1                -0.005827   0.043117  -0.135  0.89250    
    ## HeartDiseaseorAttack1   0.196949   0.044242   4.452 8.52e-06 ***
    ## PhysActivity1          -0.086381   0.201615  -0.428  0.66833    
    ## Fruits1                 0.105632   0.048229   2.190  0.02851 *  
    ## Veggies1               -0.020691   0.046618  -0.444  0.65715    
    ## HvyAlcoholConsump1     -0.208515   0.064659  -3.225  0.00126 ** 
    ## AnyHealthcare1          0.060683   0.058415   1.039  0.29888    
    ## NoDocbcCost1           -0.047962   0.049847  -0.962  0.33596    
    ## GenHlth2               -0.557287   0.427476  -1.304  0.19235    
    ## GenHlth3                0.340826   0.512766   0.665  0.50625    
    ## GenHlth4                0.319697   0.523994   0.610  0.54179    
    ## GenHlth5                0.271406   0.449341   0.604  0.54584    
    ## MentHlth               -0.029261   0.050023  -0.585  0.55858    
    ## PhysHlth               -0.055230   0.057042  -0.968  0.33293    
    ## DiffWalk1               0.054376   0.052138   1.043  0.29698    
    ## Sex1                    0.043035   0.050230   0.857  0.39158    
    ## Age2                    1.345701  33.079808   0.041  0.96755    
    ## Age3                    2.044991  50.520136   0.040  0.96771    
    ## Age4                    2.370320  60.832169   0.039  0.96892    
    ## Age5                    2.460402  62.935734   0.039  0.96882    
    ## Age6                    3.177246  75.909409   0.042  0.96661    
    ## Age7                    3.802571  91.518954   0.042  0.96686    
    ## Age8                    3.778893  88.099036   0.043  0.96579    
    ## Age9                    4.003912  94.091794   0.043  0.96606    
    ## Age10                   4.387749 100.364289   0.044  0.96513    
    ## Age11                   4.278685  98.060227   0.044  0.96520    
    ## Age12                   4.121848  94.487885   0.044  0.96520    
    ## Age13                   4.800666 111.430324   0.043  0.96564    
    ## Income2                 0.005860   0.051918   0.113  0.91013    
    ## Income3                -0.049075   0.053665  -0.914  0.36048    
    ## Income4                -0.087759   0.055571  -1.579  0.11429    
    ## Income5                -0.048532   0.055122  -0.880  0.37862    
    ## Income6                -0.107520   0.054997  -1.955  0.05058 .  
    ## Income7                -0.051576   0.055843  -0.924  0.35571    
    ## Income8                -0.071754   0.058265  -1.232  0.21813    
    ## `HighBP1:HighChol1`    -0.095013   0.103567  -0.917  0.35893    
    ## `HighBP1:BMI`          -0.119532   0.245319  -0.487  0.62608    
    ## `HighChol1:BMI`        -0.241861   0.232003  -1.042  0.29718    
    ## `BMI:PhysActivity1`     0.016541   0.200628   0.082  0.93429    
    ## `BMI:GenHlth2`          0.571769   0.402416   1.421  0.15536    
    ## `BMI:GenHlth3`         -0.065093   0.510856  -0.127  0.89861    
    ## `BMI:GenHlth4`          0.211510   0.535857   0.395  0.69305    
    ## `BMI:GenHlth5`          0.283480   0.466621   0.608  0.54351    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 3563.9  on 2951  degrees of freedom
    ## Residual deviance: 2938.7  on 2903  degrees of freedom
    ## AIC: 3036.7
    ## 
    ## Number of Fisher Scoring iterations: 14

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
    ## 2952 samples
    ##   20 predictor
    ##    2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (43), scaled (43) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 2361, 2361, 2362, 2362, 2362 
    ## Resampling results:
    ## 
    ##   logLoss  
    ##   0.5099902

``` r
summary(poly_log)
```

    ## 
    ## Call:
    ## NULL
    ## 
    ## Coefficients:
    ##                         Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)            -1.272754   2.071754  -0.614 0.538993    
    ## HighBP1                 0.334083   0.053696   6.222 4.92e-10 ***
    ## HighChol1               0.318969   0.049994   6.380 1.77e-10 ***
    ## CholCheck1              0.192451   0.074508   2.583 0.009796 ** 
    ## `I(BMI^2)`             -0.808533   0.215146  -3.758 0.000171 ***
    ## BMI                     1.106592   0.217928   5.078 3.82e-07 ***
    ## Smoker1                -0.063211   0.049701  -1.272 0.203431    
    ## Stroke1                -0.001688   0.043281  -0.039 0.968882    
    ## HeartDiseaseorAttack1   0.199447   0.044397   4.492 7.04e-06 ***
    ## PhysActivity1          -0.069196   0.046883  -1.476 0.139962    
    ## Fruits1                 0.093586   0.048228   1.941 0.052318 .  
    ## Veggies1               -0.019945   0.046690  -0.427 0.669248    
    ## HvyAlcoholConsump1     -0.203205   0.064147  -3.168 0.001536 ** 
    ## AnyHealthcare1          0.067013   0.058318   1.149 0.250523    
    ## NoDocbcCost1           -0.044595   0.050047  -0.891 0.372904    
    ## GenHlth2                0.030122   0.093791   0.321 0.748086    
    ## GenHlth3                0.262924   0.121865   2.158 0.030966 *  
    ## GenHlth4                0.511335   0.124057   4.122 3.76e-05 ***
    ## GenHlth5                0.536928   0.110760   4.848 1.25e-06 ***
    ## `I(MentHlth^2)`         0.076481   0.198402   0.385 0.699878    
    ## MentHlth               -0.106264   0.203766  -0.522 0.602018    
    ## `I(PhysHlth^2)`        -0.263992   0.208957  -1.263 0.206454    
    ## PhysHlth                0.210448   0.219187   0.960 0.336989    
    ## DiffWalk1               0.041779   0.052464   0.796 0.425843    
    ## Sex1                    0.038141   0.050381   0.757 0.449018    
    ## Age2                    1.361160  33.093597   0.041 0.967192    
    ## Age3                    2.066487  50.541195   0.041 0.967386    
    ## Age4                    2.385845  60.857528   0.039 0.968728    
    ## Age5                    2.476954  62.961969   0.039 0.968619    
    ## Age6                    3.205753  75.941053   0.042 0.966328    
    ## Age7                    3.825282  91.557105   0.042 0.966674    
    ## Age8                    3.805716  88.135762   0.043 0.965558    
    ## Age9                    4.031991  94.131017   0.043 0.965834    
    ## Age10                   4.425925 100.406128   0.044 0.964840    
    ## Age11                   4.304843  98.101105   0.044 0.964999    
    ## Age12                   4.157756  94.527274   0.044 0.964917    
    ## Age13                   4.844573 111.476776   0.043 0.965336    
    ## Income2                -0.002307   0.052090  -0.044 0.964676    
    ## Income3                -0.057440   0.053841  -1.067 0.286047    
    ## Income4                -0.089940   0.055665  -1.616 0.106150    
    ## Income5                -0.042002   0.054804  -0.766 0.443433    
    ## Income6                -0.105436   0.055148  -1.912 0.055892 .  
    ## Income7                -0.050929   0.055851  -0.912 0.361834    
    ## Income8                -0.076103   0.058322  -1.305 0.191935    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 3563.9  on 2951  degrees of freedom
    ## Residual deviance: 2929.6  on 2908  degrees of freedom
    ## AIC: 3017.6
    ## 
    ## Number of Fisher Scoring iterations: 14

``` r
#apply the best model on the test set and merge the predicted results with the true response into one data frame
predicted3 <- data.frame(obs=test$diabetes_dx,
             pred=predict(poly_log, test),
             predict(poly_log, test, type="prob"))

#calculate the log-loss
c <- mnLogLoss(predicted3, lev = levels(predicted3$obs))
```

### Model selection for logistic regression

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
    ## 1         0.532 Logistc regression model three

From the result we can tell, the Logistc regression model three has the
lowest log-loss value (0.5322047), thus, the Logistc regression model
three is the best logistic regression model for predicting diabetes
diagnosis.

## LASSO logistic regression

LASSO, standing for Least Absolute Shrinkage and Selection Operator, is
a popular technique used in statistical modeling and machine learning to
estimate the relationships between variables and make prediction.

The objective LASSO regression is to find the values of the coefficients
that minimize the sum of the squared differences between the predicted
values and the actual values, while also minimizing the L1
regularization term, which is an additional penalty term based on the
absolute values of the coefficients.

Compared with basic logistic regression, LASSO logistic regression model
help with the selection of variables for prediction without prior
knowledge.LASSO regression model uses a penalty based method to penalize
against using irrelevant predictors and also could prevent over-fitting
because LASSO could reduce the “weight” for each predictor. In addition,
it could be used to automate the variable selection process.

Next, we will use the `caret` package to fit a model to the training
set, we will assign `glmnet` as the model type and use log-loss to
evaluate the model performance.

``` r
lasso_log_reg<-train(diabetes_dx ~ HighBP + HighChol + CholCheck + BMI + Smoker + Stroke + HeartDiseaseorAttack 
                  + PhysActivity + Fruits + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth + MentHlth 
                  + PhysHlth + DiffWalk + Sex + Age + Income,
                     data=train,
                     method="glmnet",
                     metric="logLoss",
                     preProcess=c("center","scale"),
                     trControl=trainControl(method="cv", number = 5, classProbs=TRUE, summaryFunction=mnLogLoss),
                     tuneGrid = expand.grid(alpha = seq(0,1,by =0.1),
                                            lambda = seq (0,1,by=0.1)))
head(lasso_log_reg$results)
```

    ## # A tibble: 6 × 4
    ##   alpha lambda logLoss logLossSD
    ##   <dbl>  <dbl>   <dbl>     <dbl>
    ## 1     0    0     0.513   0.00784
    ## 2     0    0.1   0.517   0.00586
    ## 3     0    0.2   0.523   0.00488
    ## 4     0    0.3   0.529   0.00421
    ## 5     0    0.4   0.535   0.00370
    ## 6     0    0.5   0.539   0.00330

``` r
lasso_log_reg$bestTune
```

    ## # A tibble: 1 × 2
    ##   alpha lambda
    ##   <dbl>  <dbl>
    ## 1     0      0

``` r
#apply the best model on the test set and merge the predicted results with the true response into one data frame
predicted6 <- data.frame(obs=test$diabetes_dx,
             pred=predict(lasso_log_reg, test),
             predict(lasso_log_reg, test, type="prob"))

#calculate the log-loss
f <- mnLogLoss(predicted6, lev = levels(predicted6$obs))
```

## Classifcation tree model

classification tree is also a very straight-forward and easy to read
model. By fitting a classification tree models, we can separate the
predictors into different “spaces” and within each space, further split
could be made based off other predictors, to generate even smaller
region and so on. Classification tree model makes prediction for certain
test data by taking the higher voted value from the corresponding
region.

Compared with linear regression model or generalized linear model, tree
methods has several advantages:

1.  No variable selection is needed to perform manually.  
2.  Data preprocessing: no normalization/scaling needed.  
3.  A classification tree model is very intuitive and generally easier
    to understand.

On the other hand, there are also significant drawbacks:

1.  A single classification tree model is inadequate for accurately
    predicting a response.  
2.  Tree model takes a longer time to run compared with linear
    regression models.  
3.  A small change in the data set can sometimes cause quite big changes
    in the classification tree structure.

So there is no perfect prediction algorithm for all problems, we need to
decide based on the data set and the prediction goal. The response we
are looking at is a binary factor, so fitting a classification tree
model here is suitable.

``` r
class_tree<-train(diabetes_dx ~ HighBP + HighChol + CholCheck + BMI + Smoker + Stroke + HeartDiseaseorAttack 
                  + PhysActivity + Fruits + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth + MentHlth 
                  + PhysHlth + DiffWalk + Sex + Age + Income,
                  data=train,
                  method="rpart",
                  metric="logLoss",
                  preProcess=c("center","scale"),
                  trControl=trainControl(method="cv", number = 5, classProbs=TRUE, summaryFunction=mnLogLoss),
                  tuneGrid = data.frame(cp = seq(0,1,by = 0.1)))
class_tree
```

    ## CART 
    ## 
    ## 2952 samples
    ##   20 predictor
    ##    2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (40), scaled (40) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 2362, 2362, 2362, 2361, 2361 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp   logLoss  
    ##   0.0  0.8358985
    ##   0.1  0.6036376
    ##   0.2  0.6036376
    ##   0.3  0.6036376
    ##   0.4  0.6036376
    ##   0.5  0.6036376
    ##   0.6  0.6036376
    ##   0.7  0.6036376
    ##   0.8  0.6036376
    ##   0.9  0.6036376
    ##   1.0  0.6036376
    ## 
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 1.

``` r
#apply the best model on the test set and merge the predicted results with the true response into one data frame
predicted7 <- data.frame(obs=test$diabetes_dx,
             pred=predict(class_tree, test),
             predict(class_tree, test, type="prob"))

#calculate the log-loss
g <- mnLogLoss(predicted7, lev = levels(predicted7$obs))
```

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
                ntree = 100,
                metric="logLoss",
                preProcess = c("center", "scale"),
                tuneGrid = data.frame(mtry = c(5:15)),
                trControl = trainControl(method = "cv", number = 5, classProbs=TRUE, summaryFunction=mnLogLoss))
ran_for
```

    ## Random Forest 
    ## 
    ## 2952 samples
    ##   20 predictor
    ##    2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (40), scaled (40) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 2361, 2361, 2362, 2362, 2362 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  logLoss  
    ##    5    0.5337206
    ##    6    0.5292374
    ##    7    0.5280548
    ##    8    0.5592258
    ##    9    0.5606457
    ##   10    0.5450618
    ##   11    0.5669313
    ##   12    0.5345997
    ##   13    0.5479900
    ##   14    0.5456677
    ##   15    0.5603880
    ## 
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 7.

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
    ## 2952 samples
    ##   20 predictor
    ##    2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (40), scaled (40) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 2362, 2361, 2361, 2362, 2362 
    ## Resampling results across tuning parameters:
    ## 
    ##   iter  logLoss  
    ##   1     0.5450542
    ##   2     0.5368012
    ##   3     0.5370983
    ## 
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final value used for the model was iter = 2.

``` r
#apply the best model on the test set and merge the predicted results with the true response into one data frame
predicted5 <- data.frame(obs=test$diabetes_dx,
             pred=predict(log_tr, test),
             predict(log_tr, test, type="prob"))

#calculate the log-loss

e <- mnLogLoss(predicted5, lev = levels(predicted5$obs))
```

## Linear discriminant analysis

This is also another linear model available in the `caret` package
called Linear discriminant analysis (discriminant correspondence
analysis), it is a method for finding a linear combination of features
that characterizes or separates two or more classes of response. LDA is
very similar to logistic regression as it explains a categorical
variable by the values of continuous independent variables.It is also
closely related to principal component analysis which looks for linear
combinations of variables that best explain data. One difference is that
LDA explicitly attempts to model the difference between the classes of
data.

Since our response is binary data, We will use `lda` as model type to
fit a Linear discriminant analysis model as below:

``` r
class_CART<-train(diabetes_dx ~ HighBP + HighChol + CholCheck + BMI + Smoker + Stroke + HeartDiseaseorAttack 
                  + PhysActivity + Fruits + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth + MentHlth 
                  + PhysHlth + DiffWalk + Sex + Age + Income, 
                  data=train,
                  method="lda",
                  metric="logLoss",
                  preProcess=c("center","scale"),
                  trControl=trainControl(method="cv", number = 5, classProbs=TRUE, summaryFunction=mnLogLoss))
class_CART
```

    ## Linear Discriminant Analysis 
    ## 
    ## 2952 samples
    ##   20 predictor
    ##    2 classes: 'No', 'Yes' 
    ## 
    ## Pre-processing: centered (40), scaled (40) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 2362, 2362, 2362, 2361, 2361 
    ## Resampling results:
    ## 
    ##   logLoss  
    ##   0.5158826

``` r
#apply the best model on the test set and merge the predicted results with the true response into one data frame
predicted8 <- data.frame(obs=test$diabetes_dx,
             pred=predict(class_CART, test),
             predict(class_CART, test, type="prob"))

#calculate the log-loss

h <- mnLogLoss(predicted8, lev = levels(predicted8$obs))
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
final_pick <- data.frame(logloss_value = c(f, g, d, e, h), model = c("LASSO logistic regression model", "Classification tree model", "Random forest model", "Logistic model tree", "Linear discriminant analysis model"))
#append with previous best logistic model
final_pick <- rbind(final_pick, log_best)
final_pick
```

    ## # A tibble: 6 × 2
    ##   logloss_value model                             
    ##           <dbl> <chr>                             
    ## 1         0.526 LASSO logistic regression model   
    ## 2         0.604 Classification tree model         
    ## 3         0.546 Random forest model               
    ## 4         0.559 Logistic model tree               
    ## 5         0.526 Linear discriminant analysis model
    ## 6         0.532 Logistc regression model three

``` r
#return the smallest log-loss model
final_best <- final_pick[which.min(final_pick$logloss_value),]
final_best
```

    ## # A tibble: 1 × 2
    ##   logloss_value model                          
    ##           <dbl> <chr>                          
    ## 1         0.526 LASSO logistic regression model

Finally, after comparing all six models’ log-loss values fitting on the
test set, the LASSO logistic regression model has the lowest log-loss
value (0.5258203), thus, the LASSO logistic regression model is the best
model for predicting diabetes diagnosis.
