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
library(rpart)
library(e1071)
library(ROSE)
library(pROC)
library(naivebayes)
library(bnlearn)
install.packages("pROC")

# Instaliraj recipes i tidymodels
install.packages("recipes")
install.packages("tidymodels")

# Učitaj recipes paket
library(recipes)


#Ponovna provera da li numeričke varijable imaju normalnu raspodelu
shapiro.test(sample(dataset$tenure,size=5000))
shapiro.test(sample(dataset$MonthlyCharges,size=5000))
shapiro.test(sample(dataset$TotalCharges,size=5000))
#Pošto je p value manja od 0.05, promenljiva nema normalnu raspodelu i diskretizovaćemo je

#tenure diskretizacija
dataset$tenure<-as.numeric(dataset$tenure)
tenure.df<-as.data.frame(dataset$tenure)
discretized_tenure<-discretize(tenure.df,
                        method = "quantile",
                        breaks = c(5))
summary(discretized_tenure)

#MonthlyCharges diskretizacija
dataset$MonthlyCharges<-as.numeric(dataset$MonthlyCharges)
MonthlyCharges.df<-as.data.frame(dataset$MonthlyCharges)
discretized_MonthlyCharges<-discretize(MonthlyCharges.df,
                               method = "quantile",
                               breaks = c(5))
summary(discretized_MonthlyCharges)

#TotalCharges diskretizacija
dataset$TotalCharges<-as.numeric(dataset$TotalCharges)
TotalCharges.df<-as.data.frame(dataset$TotalCharges)
discretized_TotalCharges<-discretize(TotalCharges.df,
                                       method = "quantile",
                                       breaks = c(5))
summary(discretized_TotalCharges)

#Kreiranje novog dataseta sa diskretizovanim vrednostima
newData<-as.data.frame(cbind(discretized_tenure,discretized_MonthlyCharges,
                       discretized_TotalCharges,dataset[,c(1,2,3,5:14,17)]))
print(colnames(newData))
colnames(newData)[colnames(newData) == "dataset$tenure"] <- "tenure"
colnames(newData)[colnames(newData) == "dataset$MonthlyCharges"] <- "MonthlyCharges"
colnames(newData)[colnames(newData) == "dataset$TotalCharges"] <- "TotalCharges"

dataset<-newData

#Podela na test i trening 
set.seed(100) 
indexes <- createDataPartition(dataset$Churn, p = 0.8, list = FALSE)
trainSet<- dataset[indexes, ]
testSet <- dataset[-indexes, ]

#--------------------PRVI MODEL--------------------

nb1 <- naiveBayes(Churn ~., data = trainSet)
nb1

nb1.pred <- predict(nb1, newdata = testSet, type = "class")

nb1.cm <- table(true = testSet$Churn, predicted = nb1.pred)
nb1.cm
#     predicted
#true  Yes  No
#Yes 288  85
#No  317 717
eval.nb1 <- getEvaluationMetrics(nb1.cm)
eval.nb1
# Accuracy Precision    Recall        F1 
#0.7142857 0.4760331 0.7721180 0.5889571

#--------------------DRUGI MODEL--------------------
#control
ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     sampling = "down")
grid <- expand.grid(laplace = 1, usekernel= FALSE, adjust = 1)

#downSample
set.seed(1)
down_inside_nb <- train(x = trainSet[,-17], 
                     y = trainSet$Churn,
                     method = "naive_bayes",
                     metric = "ROC",
                     trControl = ctrl,
                     tuneGrid = grid)
cpValue_down<-down_inside_nb$bestTune$cp
cpValue_down
nb_down.prediction<-predict(down_inside_nb$finalModel,
                            newdata=testSet,
                            type="class")
nb_down.cm<-table(true=testSet$Churn,
                  predicted=nb_down.prediction)

eval.nb_down <- getEvaluationMetrics(nb_down.cm)
eval.nb_down
# Accuracy Precision    Recall        F1 
#0.6922530 0.4544073 0.8016086 0.5800194 

#up
ctrl$sampling <- "up"
set.seed(1)
up_inside_nb <- train(x = trainSet[,-17], 
                   y = trainSet$Churn,
                   method = "naive_bayes",
                   metric = "ROC",
                   trControl = ctrl,
                   tuneGrid = grid)
cpValue_up<-up_inside_nb$bestTune$cp
cpValue_up
nb_up.prediction<-predict(up_inside_nb$finalModel,
                          newdata=testSet,
                          type="class")
nb_up.cm<-table(true=testSet$Churn,
                predicted=nb_up.prediction)

eval.nb_up <- getEvaluationMetrics(nb_up.cm)
eval.nb_up
# Accuracy Precision    Recall        F1 
#0.6837242 0.4465875 0.8069705 0.5749761 

#rose
ctrl$sampling <- "rose"
set.seed(1)
rose_inside_nb <- train(x = trainSet[,-17], 
                     y = trainSet$Churn,
                     method = "naive_bayes",
                     metric = "ROC",
                     trControl = ctrl,
                     tuneGrid = grid)
cpValue_rose<-rose_inside_nb$bestTune$cp
cpValue_rose
nb_rose.prediction<-predict(rose_inside_nb$finalModel,
                            newdata=testSet,
                            type="class")
nb_rose.cm<-table(true=testSet$Churn,
                  predicted=nb_rose.prediction)

eval.nb_rose <- getEvaluationMetrics(nb_rose.cm)
eval.nb_rose
#Accuracy Precision    Recall        F1 
#0.31982942 0.09444444 0.18230563 0.12442818 

#orig fit
ctrl$sampling <- NULL
set.seed(1)
orig_fit_nb <- train(x = trainSet[,-17], 
                  y = trainSet$Churn, 
                  method = "naive_bayes",
                  metric = "ROC",
                  trControl = ctrl,
                  tuneGrid = grid)
cpValue_orig_fit<-orig_fit_nb$bestTune$cp
cpValue_orig_fit
nb_orig_fit.prediction<-predict(orig_fit_nb$finalModel,
                                newdata=testSet,
                                type="class")
nb_orig_fit.cm<-table(true=testSet$Churn,
                      predicted=nb_orig_fit.prediction)

eval.nb_orig_fit <- getEvaluationMetrics(nb_orig_fit.cm)
eval.nb_orig_fit
# Accuracy Precision    Recall        F1 
#0.7178394 0.4801325 0.7774799 0.5936540 

#--------------------TREĆI MODEL--------------------

nb2.pred.prob <- predict(nb1, newdata = testSet, type = "raw")
nb2.pred.prob

nb2.roc <- roc(response = as.integer(testSet$Churn),
               predictor = nb2.pred.prob[,1],
               levels = c(2,1))
plot.roc(nb2.roc)
nb2.roc$auc
#Area under the curve: 0.8066

plot.roc(nb2.roc, print.thres = TRUE, print.thres.best.method = "youden")

nb2.coords <- coords(nb2.roc,
                            ret = c("accuracy","spec","sens","threshold"),
                            x = "best",
                            best.method = "youden")
# accuracy specificity sensitivity threshold
#threshold 0.7562189   0.7717602   0.7131367 0.7714423

threshold <- nb2.coords.youden[1,'threshold']
#0.7714423

nb2.pred <- ifelse(test = nb2.pred.prob[,1]>= threshold,
                   yes = "Yes", no = "No")
nb2.pred <- factor(nb2.pred, levels = c("Yes","No"))

nb2.cm<- table(true = testSet$Churn, predicted = nb2.pred)
nb2.cm
#     predicted
#true  Yes  No
#Yes 266 107
#No  236 798

eval.nb2 <- getEvaluationMetrics(nb2.cm)
eval.nb2
# Accuracy Precision    Recall        F1 
#0.7562189 0.5298805 0.7131367 0.6080000 





data.frame(rbind(nb1.cm, nb2.cm,nb_down.cm))
data.frame(rbind(eval.nb1, eval.nb_orig_fit,eval.nb2), row.names = c("prvi","drugi","treći"))
