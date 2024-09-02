#Učitavanje sređenog dataseta
preprocessed_dataset<- readRDS("preprocessed_dataset.RDS")
dataset <- preprocessed_dataset

#Instaliranje paketa i učitavanje paketa u okviru biblioteke
install.packages("rpart.plot")
library(caret)
library(rpart)
library(rpart.plot)
library(e1071)
library(ROSE)
library(pROC)

#Učitavanje funkcije 
source("funkcija_matrica.R")

#Kreiranje train i test dataseta
set.seed(100) 
indexes <- createDataPartition(dataset$Churn, p = 0.8, list = FALSE)
trainSet<- dataset[indexes, ]
testSet <- dataset[-indexes, ]

#--------------------PRVI MODEL--------------------

#Kreiranje nebalansiranog stabla
tree1 <- rpart(Churn ~ ., 
           data = trainSet,
           method = "class")
tree1

rpart.plot(tree1, extra = 104)

#Kreiranje predikcije za prvi model
tree1.prediction<-predict(tree1,
                          newdata=testSet,
                          type="class")

#Kreiranje matrice konfuzije za prvi model
tree1.cm<-table(true=testSet$Churn,
              predicted=tree1.prediction)
tree1.cm

#Razmatranje evalucionih metrika
eval.tree1 <- getEvaluationMetrics(tree1.cm)
eval.tree1
#Accuracy Precision    Recall        F1 
#0.7867804 0.6308244 0.4718499 0.5398773 

#--------------------DRUGI MODEL--------------------

tree2 <- rpart(Churn~.,
               data = trainSet,
               method = "class",
               control = rpart.control(cp = 0.001))



tree2.pred <- predict(tree2,
                      newdata = testSet,
                      type = "class")

tree2.cm <- table(true = testSet$Churn,
                  predicted = tree2.pred)
tree2.cm

eval.tree2 <- getEvaluationMetrics(tree2.cm)
eval.tree2
#Accuracy Precision    Recall        F1 
#0.7768301 0.6096654 0.4396783 0.5109034 

#--------------------TREĆI MODEL--------------------

#Krosvalidacija i određvanje optimalne vrednosti cp 

numFolds <- trainControl(method = "cv", number = 10)
cpGrid <- expand.grid(.cp=seq(from = 0.001,
                              to = 0.05,
                              by = 0.001))
set.seed(100)
crossvalidation <- train(x = trainSet[,-17],
                         y = trainSet$Churn,
                         method = "rpart",
                         control = rpart.control(minsplit = 10),
                         trControl = numFolds,
                         tuneGrid = cpGrid)

crossvalidation
plot(crossvalidation)
cpValue <- crossvalidation$bestTune$cp

#cp ima vrednost 0.004

#Kreiranje stabla nakon krosvalidacije
tree3 <- rpart(Churn~.,
               data = trainSet,
               method = "class",
               control = rpart.control(cp = cpValue))



tree3.pred <- predict(tree3,
                      newdata = testSet,
                      type = "class")

tree3.cm <- table(true = testSet$Churn,
                  predicted = tree3.pred)
tree3.cm

eval.tree3 <- getEvaluationMetrics(tree3.cm)
eval.tree3

#Accuracy Precision    Recall        F1 
#0.7860697 0.6046512 0.5576408 0.5801953 

data.frame(rbind(tree1.cm, tree2.cm,tree3.cm))

data.frame(rbind(eval.tree1, eval.tree2,eval.tree3), row.names = c("prvi","drugi","treći"))

#Accuracy Precision    Recall        F1
#prvi  0.7867804 0.6308244 0.4718499 0.5398773
#drugi 0.7768301 0.6096654 0.4396783 0.5109034
#treći 0.7860697 0.6046512 0.5576408 0.5801953


#-----------------------------------------------------------------
#control
ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     sampling = "down")
#downSample
set.seed(1)
down_inside <- train(x = trainSet[,-17], 
                     y = trainSet$Churn,
                     method = "rpart",
                     metric = "ROC",
                     trControl = ctrl)
#up
ctrl$sampling <- "up"
set.seed(1)
up_inside <- train(x = trainSet[,-17], 
                   y = trainSet$Churn,
                   method = "rpart",
                   metric = "ROC",
                   trControl = ctrl)
#rose
ctrl$sampling <- "rose"
set.seed(1)
rose_inside <- train(x = trainSet[,-17], 
                     y = trainSet$Churn,
                     method = "rpart",
                     metric = "ROC",
                     trControl = ctrl)


#orig fit
ctrl$sampling <- NULL
set.seed(1)
orig_fit <- train(x = trainSet[,-17], 
                  y = trainSet$Churn, 
                  method = "rpart",
                  metric = "ROC",
                  trControl = ctrl)

inside_models <- list(original = orig_fit,
                      down = down_inside,
                      up = up_inside,
                      ROSE = rose_inside)

inside_resampling <- resamples(inside_models)
summary(inside_resampling, metric = "ROC")
#Zakljucujem da je original fit najbolji

