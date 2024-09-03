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

rf1 <- randomForest(Churn~., data = trainSet)

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
#control
ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     sampling = "down")

#downSample
set.seed(100)
down_inside_rf <- train(x = trainSet[,-17], 
                     y = trainSet$Churn,
                     method = "rf",
                     metric = "ROC",
                     trControl = ctrl)

#up
ctrl$sampling <- "up"

set.seed(100)
up_inside_rf <- train(x = trainSet[,-17], 
                   y = trainSet$Churn,
                   method = "rf",
                   metric = "ROC",
                   trControl = ctrl)

#rose
ctrl$sampling <- "rose"

set.seed(1010)
rose_inside_rf <- train(x = trainSet[,-17], 
                     y = trainSet$Churn,
                     method = "rf",
                     metric = "ROC",
                     trControl = ctrl)

#original
ctrl$sampling <- NULL

set.seed(1010)
orig_fit_rf <- train(x = trainSet[,-17], 
                  y = trainSet$Churn, 
                  method = "rf",
                  metric = "ROC",
                  trControl = ctrl)


inside_models <- list(original = orig_fit_rf,
                      down = down_inside_rf,
                      up = up_inside_rf)
                      #ROSE = rose_inside_rf)

inside_resampling <- resamples(inside_models)
summary(inside_resampling, metric = "ROC")



rf2.pred <- predict(orig_fit_rf$finalModel, newdata = testSet, type = "class")

rf2.cm <- table(true = testSet$Churn, predicted = rf2.pred)
rf2.cm

rf2.eval <- getEvaluationMetrics(rf2.cm)
rf2.eval
# Accuracy Precision    Recall        F1 
#0.8002843 0.6523179 0.5281501 0.5837037 

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
best_mtry <- model_rf$bestTune$mtry

rf3 <- randomForest(Churn~., data = trainSet, mtry = best_mtry)

rf3.pred <- predict(object = rf3, newdata = testSet, type = "class")

rf3.cm <- table(true = testSet$Churn, predicted = rf3.pred)
rf3.cm

rf3.eval <- getEvaluationMetrics(rf3.cm)
rf3.eval
# Accuracy Precision    Recall        F1 
#0.8009950 0.6587031 0.5174263 0.5795796 

data.frame(rbind(rf1.eval, rf2.eval, rf3.eval), row.names = c("prvi", "drugi", "treći"))
