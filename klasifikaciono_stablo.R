#Učitavanje sređenog dataseta
preprocessed_dataset<- readRDS("preprocessed_dataset.RDS")
dataset <- preprocessed_dataset
#Instaliranje paketa i učitavanje paketa u okviru biblioteke
#install.packages("rpart.plot")
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


png("stablo_odlucivanja1.png", width = 1200, height = 800, res = 150)
rpart.plot(tree1, 
           extra = 106,   
           cex = 0.8,         
           main = "Stablo odlučivanja") 
dev.off()
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


png("stablo_odlucivanja.png", width = 1200, height = 800, res = 150)
rpart.plot(tree2, 
           extra = 106,   
           cex = 0.8,         
           main = "Stablo odlučivanja") 
dev.off()



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

#up
ctrl$sampling <- "up"
set.seed(1)
up_inside <- train(x = trainSet[,-17], 
                   y = trainSet$Churn,
                   method = "rpart",
                   metric = "ROC",
                   trControl = ctrl,
                   tuneGrid = cpGrid)

#rose
ctrl$sampling <- "rose"
set.seed(1)
rose_inside <- train(x = trainSet[,-17], 
                     y = trainSet$Churn,
                     method = "rpart",
                     metric = "ROC",
                     trControl = ctrl,
                     tuneGrid = cpGrid)

#orig fit
ctrl$sampling <- NULL
set.seed(1)
orig_fit <- train(x = trainSet[,-17], 
                  y = trainSet$Churn, 
                  method = "rpart",
                  metric = "ROC",
                  trControl = ctrl,
                  tuneGrid = cpGrid)

inside_models <- list(original = orig_fit,
                      down = down_inside,
                      up = up_inside,
                      ROSE = rose_inside)
inside_resampling <- resamples(inside_models)
summary(inside_resampling, metric = "ROC")

#Najbolji je up
# ROC 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# original 0.7666245 0.7966941 0.8134756 0.8128483 0.8259276 0.8591385    0
# down     0.7757595 0.8037017 0.8134914 0.8134851 0.8235183 0.8504026    0
# up       0.7508916 0.7998792 0.8152295 0.8145790 0.8308718 0.8552184    0
# ROSE     0.7631472 0.7963205 0.8084797 0.8099266 0.8228967 0.8461272    0

png("stablo_odlucivanja3.png", width = 1700, height = 1200, res = 150)
rpart.plot(up_inside$finalModel, 
           extra = 106,   
           cex = 0.8,         
           main = "Stablo odlučivanja") 
dev.off()
tree3.pred <- predict(up_inside$finalModel,
                      newdata = testSet,
                      type = "class")

tree3.cm <- table(true = testSet$Churn,
                  predicted = tree3.pred)
tree3.cm

eval.tree3 <- getEvaluationMetrics(tree3.cm)
eval.tree3
# Accuracy Precision    Recall        F1 
#0.7405828 0.5072202 0.7533512 0.6062567 

data.frame(rbind(tree1.cm, tree2.cm, tree3.cm))
data.frame(rbind(eval.tree1, eval.tree2, eval.tree3), row.names = c("prvi","drugi","treći"))
