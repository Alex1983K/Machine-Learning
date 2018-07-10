# Physical Activity Analysis and Prediction Using Supervised Machine Learning Algorithms

Alexander Kuznetsov

07/08/2018

## Introduction
The purpose of this project is to analyze data from accelerometers installed on participants permorming physical exersices with dumbells. Trained fitness expert observed the manner in which exercises were performed and classified them in 5 different categories: A, B, C, D and E. Category A corresponds to the correct way of doing the exercise with dumbell. Other categories are defined depending on mistakes made by subjects. More information about this dataset is available [here](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har). Article detailing results of the study, dataset and literature overview are available in [open access (1)](http://web.archive.org/web/20161125212224/http://groupware.les.inf.puc-rio.br:80/work.jsf?p1=10335). The ultimate goal of the project is to build a predictive model that can identify the manner in which exercise with dumbell is done by analyzing accelerometers output. This report disscusses in detail models based on supervised machine learning algorithms such as random forest and support vector machines (SVM).  

## Cleaning Data
Training and testing datasets can be read into R using *read.csv* command. Initial look at data revealed large numbers of NA, missing and #DIV/0! values. Columns with these values can be easily identified with *summary* and *str* commands. Selecting appropriate values for *na.strings* option in *read.csv* allows filling all problematic cells with NAs, which makes it easier for future work with data. 
```{r}
library(knitr)
training <- read.csv("pml-training.csv", header=TRUE, na.strings=c("NA", "", "#DIV/0!"))
testing <- read.csv("pml-testing.csv", header=TRUE, na.strings=c("NA", "", "#DIV/0!"))
```
At this point all NA, missing and #DIV/0! values in both datasets are simply substituted with NAs. Call to *colSums* function as shown in the following code removes columns with NA values.
```{r}
training1 <- training[, colSums(is.na(training))==0]
testing1 <- testing[, colSums(is.na(testing))==0]
setdiff(colnames(training1), colnames(testing1))
setdiff(colnames(testing1), colnames(training1))
```
Last two lines of the code above compare column names of new datasets with each other to show that same columns were removed from training and testing sets. Only columns identified are "classe" and "problem_id". Both columns are unique for **training1** and **testing1** datasets, which means that all problematic columns are similar in both datasets, and they were successfully removed. Remaining columns are the same in both sets and, therefore, can be used to build and validate the model. 
```{r}
colnames(testing1)
```
First seven columns do not contain any relevant information recorded from accelerometers and can be removed from training and testing datasets. Furthermore, "problem_id" column in testing set is also useless for our task.
```{r}
colnames(testing1)[60]
```
We can arrive to our final datasets by omitting columns 1 to 7 and column 60 in **testing1**. 
```{r}
training2 <- training1[, -c(1:7)]
testing2 <- testing1[, -c(1:7, 60)]
```
Final training set contains 52 classifiers used to build model and variable "classe". Final testing set has 52 variables, which are the same as classifiers in training set.  
```{r}
dim(training2)
dim(testing2)
```

## Random Forest Approach to Classification Problem
Random forest is one of the most accurate algorithms used for classification problems such as this one. However, this method consumes significant computer time. Method "rf" in *train* function of the "caret" package performs random forest algorithm by doing its own cross validation for each bootstrap step. Usually 2/3 of the dataset is used for training the model with 1/3 used for validation and accuracy estimates (aka OOB error). Nevertheless, it is important to carry out our own cross validation and determine the error. Therefore, 70% of training set (**training2**) will be randomly selected for training the model and 30% for validation.
```{r}
set.seed(19031983)
library(caret)
inTrain <- createDataPartition(y=training2$classe, p=0.7, list=FALSE)
training3 <- training2[inTrain,]
testing3 <- training2[-inTrain,]
```
Because of concerns with computational time, parallel calculations will be used in this report by building a cluster with 3 cores on 4 core processor. Excellent detailed description of parallel calculations can be found on [Len Greski's GitHub page](https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md). 
```{r}
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)

pt1 <- proc.time()
modelRF <- train(classe~., data=training3, method="rf", importance=TRUE, trControl=trainControl(allowParallel=TRUE))
proc.time()-pt1

stopCluster(cluster)
registerDoSEQ()
```
Random forest model takes more than hour to train (elapsed time) even with parallel computing. However, main advantage of random forest model is its high accuracy.
```{r}
modelRF
```
Algorithm used different number of variables, selected randomly, at each split as shown in **mtry** column of the output above. Highest accuracy was achieved when only certain number of variables were used at each tree split. Kappa is a measure of accuracy including chance and coincidence, and its value is always less than accuracy.
As mentioned above, *train* function performs cross validation as random forest model is being built. Therefore, out-of-bag (OOB) estimate is a good measure of error rate for the model. OOB was found to be quite small, as most of the categories were identified correctly (as shown by confusion matrix below). OOB error is plotted for all "classe" categories below. Overall OOB estimate is shown as black line. Error declines fast as number of trees increases. 
```{r}
modelRF$finalModel
plot(modelRF$finalModel)
```

Original training dataset was split into training set used to buil a model and testing set for validation. Therefore, it is interesting to validate the model with the testing set and compare results with accuracy reported above in the call for *modelRF*. 
```{r}
predRF <- predict(modelRF, testing3)
confusionMatrix(predRF, testing3$classe)
```
Accuracy achieved on testing set is even higher than estimated accuracy of the model determined by resampling of the training set when creating the forest. Most of the observations are placed under correct categories. Specificity and sensitivity metrics are also close to 1 as reported for each category.
Finally, it is beneficial to look at importance of variables for the model. 
```{r}
varImp(modelRF)
```

## Support Vector Machines Approach to Classification Problem
Support Vector Machines (SVM) method is another supervised machine learning algorithm that can be used to solve classification problems. Many resources explaining this method are available [online](https://en.wikipedia.org/wiki/Support_vector_machine). Although, this algorithm requires less computing time than random forest, it is also less accurate. SVM can be performed using parallel computations on a cluster using "parallelSVM" package. However model built with this package is less accurate than model built with *svm* function from "e1071" package. Therefore, *svm* function will be used in this project. 
Accuracy of SVM method can be improved by using larger training set. Original set was split so that 70% of data were in training set and 30% - in testing set. Next, we are going to split data in such a way that 90% is used for training model and 10% for validation. 
```{r}
set.seed(3191983)
inTrain1 <- createDataPartition(y=training2$classe, p=0.9, list=FALSE)
training4 <- training2[inTrain1,]
testing4 <- training2[-inTrain1,]

library(e1071)

pt2 <- proc.time()
modelSVM <- svm(classe~., data=training4)
proc.time()-pt2
```
Computation time for SVM approach is seconds versus hours in the case of random forest. As mentioned above, accuracy of former method is also lower than for random forest.
```{r}
predSVM <- predict(modelSVM, testing4)
confusionMatrix(predSVM, testing4$classe)
```

## Comparing Random Forest with SVM
In conclusion, we can compare both approaches by trying to predict outcomes for original testing set assigned for this project (**testing2**). 
```{r}
RF <- predict(modelRF, testing2)
SVM <- predict(modelSVM, testing2)
RF
SVM
table(RF, SVM)
```
Therefore, both methods predict the same values for categories in "classe" variable. Larger training set is beneficial for SVM model. Random forest method is more accurate than SVM, but also much more computationally expensive.

## Reference
1. Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. [Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements.](http://web.archive.org/web/20161125212224/http://groupware.les.inf.puc-rio.br:80/work.jsf?p1=10335) Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6. 
