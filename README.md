### Table of contents
1. [Introduction](#1)
2. [Dataset background](#2)
3. [Assignment instructions](#3)
4. [Excercise development](#4)
    1. [Setting up R](#4.1)
    2. [Loading data](#4.2)
    3. [Partitioning data sets](#4.3)
    4. [Cleaning data](#4.4)
5. [Training models](#5)
    1. [Setting up parallel processing](#5.1)
    2. [trainControl](#5.2)
    3. [Fitting different models](#5.3)
        1. [Naive Bayes](#5.3.1)
        2. [Boosted Logistic Regression](#5.3.2)
        3. [Stochastic Gradient Boosting](#5.3.3)
        4. [CART](#5.3.4)
        5. [Random Forest](#5.3.5)
    4. [Stopping the cluster](#5.4)
6. [Visualising model performances](#6)
7. [Prediction on new dataset](#7)

Coursera Practical Machine Learning. Peer-graded Assignment: Prediction Assignment Writeup

Author: Riccardo Terzi

Date: 8 January 2017

### <a name="1"></a>Introduction
This document has been submitted as part of the prediction assigment for Coursera's [Practical Machine Learning](https://www.coursera.org/learn/practical-machine-learning).

---

### <a name="2"></a>Dataset background
> Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. 

> One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

> Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

---

### <a name="3"></a>Assignment instructions
> One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

> The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

---

### <a name="4"></a>Excercise development
#### <a name="4.1"></a>Setting up R
```R
library(caret)
library(parallel)
library(foreach)
library(doParallel)
set.seed(62094)
```

---

#### <a name="4.2"></a>Loading data
```R
dataset <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")

dim(dataset)
```
```
## [1] 19622   160
```
The data has 19,622 observations of 160 variables.

---

#### <a name="4.3"></a>Partitioning data sets
```R
inTrain <- createDataPartition(y = dataset$classe,
                               p = 0.7,
                               list = F)

training <- dataset[inTrain, ]; testing <- dataset[-inTrain, ]
```
Using 70% of the data for the training set and 30% for the test set.

```R
dim(training); dim(testing)
```
```
## [1] 13737   160
## [1] 5885  160
```

---

#### <a name="4.4"></a>Cleaning data
```R
training <- training[ , colSums(is.na(training)) == 0] # selecting only columns that do not have NAs
testing <- testing[ , colSums(is.na(testing)) == 0]

training <- training[, -nearZeroVar(training)] # removing columns with near zero variance
testing <- testing[, -nearZeroVar(testing)]

training <- training[ , -c(1:5)] # removing variables for row number, username, and timestamp
testing <- testing[ , -c(1:5)]
```
Variables have been reduced:
```R
dim(training); dim(testing)
```
```
## [1] 13737    54
## [1] 5885   54
```

---

#### <a name="5"></a>Training models
##### <a name="5.1"></a>Setting up parallel processing
Will use 3 out of 4 CPU cores
```R
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl, cores = detectCores() -1)
```

---

##### <a name="5.2"></a>trainControl
Setting arguments for traincontrol with 4-fold cross-validation and enabling parallel processing
```R
tr.ctrl <- trainControl(method = "cv",
                        number = 4,
                        allowParallel = TRUE,
                        verboseIter = TRUE)
```

---

##### <a name="5.3"></a>Fitting different models
Will use the following models: Naive Bayes, Boosted Logistic Regression, Stochastic Gradient Boosting, CART, and Random Forest.
I expect at least one of these - specifically, the random forest model - to perform with an accuracy > 95%

---

##### <a name="5.3.1"></a>Naive Bayes
```R
modFit.nb <- train(classe ~ .,
                   data = training,
                   method = "nb",
                   trControl = tr.ctrl)

saveRDS(modFit.nb, "modfit.nb.rds") # saving the model to a file

pred.nb <- predict(modFit.nb, testing)
acc.nb <- confusionMatrix(pred.nb, testing$classe)$overall['Accuracy']
acc.nb
```
```
## Accuracy 
## 0.7607477 
```

---

##### <a name="5.3.2"></a>Boosted Logistic Regression
```R
modFit.logbst <- train(classe ~ .,
                       data = training,
                       method = "LogitBoost",
                       trControl = tr.ctrl)

saveRDS(modFit.logbst, "modfit.logbst.rds") # saving the model to a file

pred.logbst <- predict(modFit.logbst, testing)
acc.logbst <- confusionMatrix(pred.logbst, testing$classe)$overall['Accuracy']
acc.logbst
```
```
# Accuracy 
# 0.928681 
```

---

##### <a name="5.3.3"></a>Stochastic Gradient Boosting
```R
  modFit.gbm <- train(classe ~ .,
                     data = training,
                     method = "gbm",
                     trControl = tr.ctrl)

saveRDS(modFit.gbm, "modfit.gbm.rds")  # saving the model to a file

pred.gbm <- predict(modFit.gbm, testing)
acc.gbm <- confusionMatrix(pred.gbm, testing$classe)$overall['Accuracy']
acc.gbm
```
```
# Accuracy 
# 0.9882753 
```

---

##### <a name="5.3.4"></a>CART
```R
modFit.rpart <- train(classe ~ .,
              data = training,
              method = "rpart",
              trControl = tr.ctrl)

saveRDS(modFit.rpart, "modfit.rpart.rds") # saving the model to a file

pred.rpart <- predict(modFit.rpart, testing)
acc.rpart <- confusionMatrix(pred.rpart, testing$classe)$overall['Accuracy']
acc.rpart
```
```
## Accuracy 
## 0.4971963 
```

---

##### <a name="5.3.5"></a>Random Forest
```R
  modFit.rf <- train(classe ~ .,
                     data = training,
                     method = "rf",
                     trControl = tr.ctrl)

saveRDS(modFit.rf, "modfit.rf.rds") # saving the model to a file

pred.rf <- predict(modFit.rf, testing)
acc.rf <- confusionMatrix(pred.rf, testing$classe)$overall['Accuracy']
acc.rf
```
```
# Accuracy 
# 0.9969414 
```

---

#### <a name="5.4"></a>Stopping the cluster
```R
stopCluster(cl)
```

---

### <a name="6"></a>Visualising model performances
```R
acc.values <- c(acc.nb, acc.logbst, acc.gbm, acc.rpart, acc.rf)
mod.names <- c("Naive Bayes", "Boosted Logistic Regression", "Stochastic Gradient Boosting", "CART", "Random Forest")
x <- data.frame(Model = mod.names,
                Accuracy = acc.values)

ggplot(x, aes(x = Model, y = Accuracy)) + 
  geom_bar(stat = "identity", aes(fill = Model)) +
  theme_bw() + theme(legend.position = "none")
```
![](accuracy.png)

Random forest is the best performing model, followed by Stochastic Gradient Boosting.
Will use the Random Forest model for the last part of the assignment.

---

### <a name="7"></a>Prediction on new dataset
Applying the Random Forest model to a different dataset
```R
test.part2 <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

dim(test.part2)
```
```
## [1]  20 160
```
```R
predVal <- predict(modFit.rf, test.part2)
```
```
## [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```