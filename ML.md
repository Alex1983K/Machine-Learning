# Physical Activity Analysis and Prediction Using Supervised Machine Learning Algorithms

Alexander Kuznetsov

07/08/2018

## Introduction
The purpose of this project is to analyze data from accelerometers installed on participants permorming physical exersices with dumbells. Trained fitness expert observed the manner in which exercises were performed and classified them in 5 different categories: A, B, C, D and E. Category A corresponds to the correct way of doing the exercise with dumbell. Other categories are defined depending on mistakes made by subjects. More information about this dataset is available [here](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har). Article detailing results of the study, dataset and literature overview are available in [open access (1)](http://web.archive.org/web/20161125212224/http://groupware.les.inf.puc-rio.br:80/work.jsf?p1=10335). The ultimate goal of the project is to build a predictive model that can identify the manner in which exercise with dumbell is done by analyzing accelerometers output. This report disscusses in detail models based on supervised machine learning algorithms such as random forest and support vector machines (SVM).  

## Cleaning Data
Training and testing datasets can be read into R using *read.csv* command. Initial look at data revealed large numbers of NA, missing and #DIV/0! values. Columns with these values can be easily identified with *summary* and *str* commands. Selecting appropriate values for *na.strings* option in *read.csv* allows filling all problematic cells with NAs, which makes it easier for future work with data. 

```r
library(knitr)
training <- read.csv("pml-training.csv", header=TRUE, na.strings=c("NA", "", "#DIV/0!"))
testing <- read.csv("pml-testing.csv", header=TRUE, na.strings=c("NA", "", "#DIV/0!"))
```
At this point all NA, missing and #DIV/0! values in both datasets are simply substituted with NAs. Call to *colSums* function as shown in the following code removes columns with NA values.

```r
training1 <- training[, colSums(is.na(training))==0]
testing1 <- testing[, colSums(is.na(testing))==0]
setdiff(colnames(training1), colnames(testing1))
```

```
## [1] "classe"
```

```r
setdiff(colnames(testing1), colnames(training1))
```

```
## [1] "problem_id"
```
Last two lines of the code above compare column names of new datasets with each other to show that same columns were removed from training and testing sets. Only columns identified are "classe" and "problem_id". Both columns are unique for **training1** and **testing1** datasets, which means that all problematic columns are similar in both datasets, and they were successfully removed. Remaining columns are the same in both sets and, therefore, can be used to build and validate the model. 

```r
colnames(testing1)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
## [13] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [16] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [19] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [22] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [25] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [28] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [31] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [34] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [37] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [40] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [43] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [46] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [49] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [52] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [55] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [58] "magnet_forearm_y"     "magnet_forearm_z"     "problem_id"
```
First seven columns do not contain any relevant information recorded from accelerometers and can be removed from training and testing datasets. Furthermore, "problem_id" column in testing set is also useless for our task.

```r
colnames(testing1)[60]
```

```
## [1] "problem_id"
```
We can arrive to our final datasets by omitting columns 1 to 7 and column 60 in **testing1**. 

```r
training2 <- training1[, -c(1:7)]
testing2 <- testing1[, -c(1:7, 60)]
```
Final training set contains 52 classifiers used to build model and variable "classe". Final testing set has 52 variables, which are the same as classifiers in training set.  

```r
dim(training2)
```

```
## [1] 19622    53
```

```r
dim(testing2)
```

```
## [1] 20 52
```

## Random Forest Approach to Classification Problem
Random forest is one of the most accurate algorithms used for classification problems such as this one. However, this method consumes significant computer time. Method "rf" in *train* function of the "caret" package performs random forest algorithm by doing its own cross validation for each bootstrap step. Usually 2/3 of the dataset is used for training the model with 1/3 used for validation and accuracy estimates (aka OOB error). Nevertheless, it is important to carry out our own cross validation and determine the error. Therefore, 70% of training set (**training2**) will be randomly selected for training the model and 30% for validation.

```r
set.seed(19031983)
library(caret)
inTrain <- createDataPartition(y=training2$classe, p=0.7, list=FALSE)
training3 <- training2[inTrain,]
testing3 <- training2[-inTrain,]
```
Because of concerns with computational time, parallel calculations will be used in this report by building a cluster with 3 cores on 4 core processor. Excellent detailed description of parallel calculations can be found on [Len Greski's GitHub page](https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md). 

```r
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)

pt1 <- proc.time()
modelRF <- train(classe~., data=training3, method="rf", importance=TRUE, trControl=trainControl(allowParallel=TRUE))
proc.time()-pt1
```

```
##    user  system elapsed 
##  105.45    2.50 5230.26
```

```r
stopCluster(cluster)
registerDoSEQ()
```
Random forest model takes more than hour to train (elapsed time) even with parallel computing. However, main advantage of random forest model is its high accuracy.

```r
modelRF
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9887879  0.9858167
##   27    0.9887053  0.9857140
##   52    0.9781341  0.9723412
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```
Algorithm used different number of variables, selected randomly, at each split as shown in **mtry** column of the output above. Highest accuracy was achieved when only certain number of variables were used at each tree split. Kappa is a measure of accuracy including chance and coincidence, and its value is always less than accuracy.
As mentioned above, *train* function performs cross validation as random forest model is being built. Therefore, out-of-bag (OOB) estimate is a good measure of error rate for the model. OOB was found to be quite small, as most of the categories were identified correctly (as shown by confusion matrix below). OOB error is plotted for all "classe" categories below. Overall OOB estimate is shown as black line. Error declines fast as number of trees increases. 

```r
modelRF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, importance = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.67%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3905    1    0    0    0 0.0002560164
## B   17 2631   10    0    0 0.0101580135
## C    0   18 2373    5    0 0.0095993322
## D    0    0   35 2215    2 0.0164298401
## E    0    0    1    3 2521 0.0015841584
```

```r
plot(modelRF$finalModel)
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10-1.png)

Original training dataset was split into training set used to buil a model and testing set for validation. Therefore, it is interesting to validate the model with the testing set and compare results with accuracy reported above in the call for *modelRF*. 

```r
predRF <- predict(modelRF, testing3)
confusionMatrix(predRF, testing3$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    7    0    0    0
##          B    0 1127   13    0    0
##          C    0    5 1012   23    0
##          D    0    0    1  939    1
##          E    0    0    0    2 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9912          
##                  95% CI : (0.9884, 0.9934)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9888          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9895   0.9864   0.9741   0.9991
## Specificity            0.9983   0.9973   0.9942   0.9996   0.9996
## Pos Pred Value         0.9958   0.9886   0.9731   0.9979   0.9982
## Neg Pred Value         1.0000   0.9975   0.9971   0.9949   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1915   0.1720   0.1596   0.1837
## Detection Prevalence   0.2856   0.1937   0.1767   0.1599   0.1840
## Balanced Accuracy      0.9992   0.9934   0.9903   0.9868   0.9993
```
Accuracy achieved on testing set is even higher than estimated accuracy of the model determined by resampling of the training set when creating the forest. Most of the observations are placed under correct categories. Specificity and sensitivity metrics are also close to 1 as reported for each category.
Finally, it is beneficial to look at importance of variables for the model. 

```r
varImp(modelRF)
```

```
## rf variable importance
## 
##   variables are sorted by maximum importance across the classes
##   only 20 most important variables shown (out of 52)
## 
##                       A      B     C     D     E
## pitch_belt        70.43 100.00 74.88 73.49 67.74
## yaw_belt          86.67  82.40 67.45 95.64 57.74
## roll_belt         62.41  88.44 82.64 90.89 57.20
## magnet_dumbbell_y 48.88  59.94 76.39 53.11 45.61
## magnet_dumbbell_z 75.64  63.89 75.23 65.96 60.49
## roll_arm          44.15  68.67 67.76 62.78 43.01
## accel_dumbbell_y  49.43  68.56 65.60 64.54 57.30
## accel_belt_z      32.42  52.65 67.25 40.18 34.18
## magnet_dumbbell_x 35.57  39.24 64.18 41.11 26.95
## accel_arm_y       23.27  61.13 31.43 43.67 35.93
## magnet_belt_y     47.21  59.31 59.74 56.30 38.48
## pitch_forearm     42.71  59.05 54.90 54.65 51.82
## yaw_arm           31.73  59.02 47.09 49.89 38.22
## gyros_dumbbell_y  37.16  43.10 58.60 43.12 35.39
## gyros_belt_z      32.48  57.21 52.78 46.69 47.29
## magnet_belt_z     41.84  55.38 54.30 57.14 52.38
## gyros_dumbbell_z  20.89  56.92 46.18 39.19 36.36
## roll_dumbbell     32.05  38.36 56.85 49.20 37.05
## magnet_forearm_z  30.61  50.92 43.09 55.49 40.02
## accel_dumbbell_z  35.06  50.93 53.96 45.75 52.65
```

## Support Vector Machines Approach to Classification Problem
Support Vector Machines (SVM) method is another supervised machine learning algorithm that can be used to solve classification problems. Many resources explaining this method are available [online](https://en.wikipedia.org/wiki/Support_vector_machine). Although, this algorithm requires less computing time than random forest, it is also less accurate. SVM can be performed using parallel computations on a cluster using "parallelSVM" package. However model built with this package is less accurate than model built with *svm* function from "e1071" package. Therefore, *svm* function will be used in this project. 
Accuracy of SVM method can be improved by using larger training set. Original set was split so that 70% of data were in training set and 30% - in testing set. Next, we are going to split data in such a way that 90% is used for training model and 10% for validation. 

```r
set.seed(3191983)
inTrain1 <- createDataPartition(y=training2$classe, p=0.9, list=FALSE)
training4 <- training2[inTrain1,]
testing4 <- training2[-inTrain1,]

library(e1071)

pt2 <- proc.time()
modelSVM <- svm(classe~., data=training4)
proc.time()-pt2
```

```
##    user  system elapsed 
##   86.64    0.29   87.56
```
Computation time for SVM approach is seconds versus hours in the case of random forest. As mentioned above, accuracy of former method is also lower than for random forest.

```r
predSVM <- predict(modelSVM, testing4)
confusionMatrix(predSVM, testing4$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 557  21   1   0   0
##          B   0 355  14   0   0
##          C   1   2 324  29   3
##          D   0   0   1 292   5
##          E   0   1   2   0 352
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9592          
##                  95% CI : (0.9495, 0.9675)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9483          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9367   0.9474   0.9097   0.9778
## Specificity            0.9843   0.9911   0.9784   0.9963   0.9981
## Pos Pred Value         0.9620   0.9621   0.9025   0.9799   0.9915
## Neg Pred Value         0.9993   0.9849   0.9888   0.9826   0.9950
## Prevalence             0.2847   0.1934   0.1745   0.1638   0.1837
## Detection Rate         0.2842   0.1811   0.1653   0.1490   0.1796
## Detection Prevalence   0.2954   0.1883   0.1832   0.1520   0.1811
## Balanced Accuracy      0.9913   0.9639   0.9629   0.9530   0.9880
```

## Comparing Random Forest with SVM
In conclusion, we can compare both approaches by trying to predict outcomes for original testing set assigned for this project (**testing2**). 

```r
RF <- predict(modelRF, testing2)
SVM <- predict(modelSVM, testing2)
RF
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
SVM
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
table(RF, SVM)
```

```
##    SVM
## RF  A B C D E
##   A 7 0 0 0 0
##   B 0 8 0 0 0
##   C 0 0 1 0 0
##   D 0 0 0 1 0
##   E 0 0 0 0 3
```
Therefore, both methods predict the same values for categories in "classe" variable. Larger training set is beneficial for SVM model. Random forest method is more accurate than SVM, but also much more computationally expensive.

## Reference
1. Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. [Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements.](http://web.archive.org/web/20161125212224/http://groupware.les.inf.puc-rio.br:80/work.jsf?p1=10335) Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6. 
