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

#Krosvalidacija i određvanje optimalne vrednosti cp 

numFolds <- trainControl(method = "cv", number = 10)
cpGrid <- expand.grid(.cp=seq(from = 0.001,
                              to = 0.05,
                              by = 0.001))
set.seed(100)
crossvalidation <- train(x = trainSet[,-17],
                         y = trainSet$Churn,
                         method = "rpart",
                         trControl = numFolds,
                         tuneGrid = cpGrid)

crossvalidation
plot(crossvalidation)
cpValue <- crossvalidation$bestTune$cp

#cp ima vrednost 0.004

#Kreiranje stabla nakon krosvalidacije
tree2 <- rpart(Churn~.,
               data = trainSet,
               method = "class",
               control = rpart.control(cp = cpValue))



tree2.pred <- predict(tree2,
                      newdata = testSet,
                      type = "class")

tree2.cm <- table(true = testSet$Churn,
                  predicted = tree2.pred)
tree2.cm

eval.tree2 <- getEvaluationMetrics(tree2.cm)
eval.tree2

#Accuracy Precision    Recall        F1 
#0.7860697 0.6046512 0.5576408 0.5801953 

#--------------------TREĆI MODEL--------------------

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
                     trControl = ctrl,
                     tuneGrid = cpGrid)
cpValue_down<-down_inside$bestTune$cp
cpValue_down
tree_down.prediction<-predict(down_inside$finalModel,
                          newdata=testSet,
                          type="class")
tree_down.cm<-table(true=testSet$Churn,
                predicted=tree_down.prediction)

eval.tree_down <- getEvaluationMetrics(tree_down.cm)
eval.tree_down
# Accuracy Precision    Recall        F1 
#0.7178394 0.4797297 0.7613941 0.5886010 

#up
ctrl$sampling <- "up"
set.seed(1)
up_inside <- train(x = trainSet[,-17], 
                   y = trainSet$Churn,
                   method = "rpart",
                   metric = "ROC",
                   trControl = ctrl,
                   tuneGrid = cpGrid)
cpValue_up<-up_inside$bestTune$cp
cpValue_up
tree_up.prediction<-predict(up_inside$finalModel,
                              newdata=testSet,
                              type="class")
tree_up.cm<-table(true=testSet$Churn,
                    predicted=tree_up.prediction)

eval.tree_up <- getEvaluationMetrics(tree_up.cm)
eval.tree_up
# Accuracy Precision    Recall        F1 
#0.7405828 0.5072202 0.7533512 0.6062567 

#rose
ctrl$sampling <- "rose"
set.seed(1)
rose_inside <- train(x = trainSet[,-17], 
                     y = trainSet$Churn,
                     method = "rpart",
                     metric = "ROC",
                     trControl = ctrl,
                     tuneGrid = cpGrid)
cpValue_rose<-rose_inside$bestTune$cp
cpValue_rose
tree_rose.prediction<-predict(rose_inside$finalModel,
                            newdata=testSet,
                            type="class")
tree_rose.cm<-table(true=testSet$Churn,
                  predicted=tree_rose.prediction)

eval.tree_rose <- getEvaluationMetrics(tree_rose.cm)
eval.tree_rose
#Accuracy Precision    Recall        F1 
#0.2835821 0.1123321 0.2466488 0.1543624 

#orig fit
ctrl$sampling <- NULL
set.seed(1)
orig_fit <- train(x = trainSet[,-17], 
                  y = trainSet$Churn, 
                  method = "rpart",
                  metric = "ROC",
                  trControl = ctrl,
                  tuneGrid = cpGrid)
cpValue_orig_fit<-orig_fit$bestTune$cp
cpValue_orig_fit
tree_orig_fit.prediction<-predict(orig_fit$finalModel,
                              newdata=testSet,
                              type="class")
tree_orig_fit.cm<-table(true=testSet$Churn,
                    predicted=tree_orig_fit.prediction)

eval.tree_orig_fit <- getEvaluationMetrics(tree_orig_fit.cm)
eval.tree_orig_fit
# Accuracy Precision    Recall        F1 
#0.7917555 0.6342282 0.5067024 0.5633383 

inside_models <- list(original = orig_fit,
                      down = down_inside,
                      up = up_inside,
                      ROSE = rose_inside)
inside_resampling <- resamples(inside_models)
summary(inside_resampling, metric = "ROC")
#Zakljucujem da je original fit najbolji


data.frame(rbind(tree1.cm, tree2.cm,tree3.cm))
data.frame(rbind(eval.tree1, eval.tree2,eval.tree_orig_fit), row.names = c("prvi","drugi","treći"))
