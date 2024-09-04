#Učitavanje sređenog dataseta
preprocessed_dataset<- readRDS("preprocessed_dataset.RDS")
dataset <- preprocessed_dataset
str(dataset)

#Učitavanje funkcije 
source("funkcija_matrica.R")

#Instaliranje paketa i učitavanje paketa u okviru biblioteke
install.packages("bnlearn")
install.packages("naivebayes")
library(caret)
library(ROSE)
library(pROC)
library(bnlearn)
library(randomForest)
library(themis)
install.packages("randomForest")
install.packages("themis")

#Podela na test i trening 
set.seed(100) 
indexes <- createDataPartition(dataset$Churn, p = 0.8, list = FALSE)
trainSet<- dataset[indexes, ]
testSet <- dataset[-indexes, ]

#--------------------PRVI MODEL--------------------

rf1 <- randomForest(Churn~.,
                    data = trainSet)

rf1.pred <- predict(object = rf1, newdata = testSet, type = "class")

rf1.cm <- table(true = testSet$Churn, predicted = rf1.pred)
rf1.cm
#predicted
#true  Yes  No
#Yes 182 191
#No  102 932

rf1.eval <- getEvaluationMetrics(rf1.cm)
rf1.eval
# Accuracy Precision    Recall        F1 
#0.7917555 0.6408451 0.4879357 0.5540335 

#--------------------DRUGI MODEL--------------------
#control
ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     sampling = "down")

#downSample
set.seed(1)
down_inside_rf <- train(x = trainSet[,-17], 
                     y = trainSet$Churn,
                     method = "rf",
                     metric = "ROC",
                     trControl = ctrl)

#up
ctrl$sampling <- "up"
set.seed(1)
up_inside_rf <- train(x = trainSet[,-17], 
                   y = trainSet$Churn,
                   method = "rf",
                   metric = "ROC",
                   trControl = ctrl)

#rose
ctrl$sampling <- "rose"
set.seed(1)
rose_inside_rf <- train(x = trainSet[,-17], 
                     y = trainSet$Churn,
                     method = "rf",
                     metric = "ROC",
                     trControl = ctrl)

#original
ctrl$sampling <- NULL
set.seed(1)
orig_fit_rf <- train(x = trainSet[,-17], 
                  y = trainSet$Churn, 
                  method = "rf",
                  metric = "ROC",
                  trControl = ctrl)


inside_models_rf <- list(original = orig_fit_rf,
                      down = down_inside_rf,
                      up = up_inside_rf,
                      ROSE = rose_inside_rf)

inside_resampling_rf <- resamples(inside_models_rf)
summary(inside_resampling_rf, metric = "ROC")
#Najbolja je down
#ROC 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#original 0.7889533 0.8268144 0.8405346 0.8375218 0.8513174 0.8791707    0
#down     0.7973188 0.8320149 0.8457345 0.8415796 0.8525030 0.8777617    0
#up       0.7986715 0.8271437 0.8454449 0.8414000 0.8523591 0.8810789    0
#ROSE     0.7953221 0.8234440 0.8361071 0.8341266 0.8460165 0.8732287    0

rf2.pred <- predict(down_inside_rf$finalModel, newdata = testSet, type = "class")

rf2.cm <- table(true = testSet$Churn, predicted = rf2.pred)
rf2.cm

rf2.eval <- getEvaluationMetrics(rf2.cm)
rf2.eval
# Accuracy Precision    Recall        F1 
#0.7377399 0.5033445 0.8069705 0.6199794

#--------------------TREĆI MODEL--------------------

grid <- expand.grid(mtry = c(2, 4, 6, 8, 10))

train_control <- trainControl(method = "cv",
                     number = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

model_rf <- train(x = trainSet[,-17],
                 y = trainSet$Churn,
                 method = "rf",
                 metric = "ROC",
                 tuneGrid = grid,
                 trControl = train_control)

model_rf$finalModel
#Confusion matrix:
#Yes   No class.error
#Yes 772  724   0.4839572
#No  414 3726   0.1000000
best_mtry <- model_rf$bestTune$mtry
#2
rf3 <- randomForest(Churn~., data = trainSet, mtry = best_mtry)

rf3.pred <- predict(object = rf3, newdata = testSet, type = "class")

rf3.cm <- table(true = testSet$Churn, predicted = rf3.pred)
rf3.cm
#     predicted
#true  Yes  No
#Yes 194 179
#No  105 929
rf3.eval <- getEvaluationMetrics(rf3.cm)
rf3.eval
# Accuracy Precision    Recall        F1 
#0.7981521 0.6488294 0.5201072 0.5773810  

data.frame(rbind(rf1.eval, rf2.eval, rf3.eval), row.names = c("prvi", "drugi", "treći"))
