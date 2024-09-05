#Učitavanje sređenog dataseta
preprocessed_dataset<- readRDS("preprocessed_dataset.RDS")
dataset <- preprocessed_dataset
str(dataset)

#Učitavanje funkcije 
source("funkcija_matrica.R")

#Instaliranje paketa i učitavanje paketa u okviru biblioteke
#install.packages("bnlearn")
#install.packages("naivebayes")
#install.packages("pROC")
library(caret)
library(rpart)
library(e1071)
library(ROSE)
library(pROC)
library(naivebayes)
library(bnlearn)


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


threshold <- nb2.coords[1,'threshold']
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



#--------------------TREĆI MODEL--------------------


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

#up
ctrl$sampling <- "up"
set.seed(1)
up_inside_nb <- train(x = trainSet[,-17], 
                      y = trainSet$Churn,
                      method = "naive_bayes",
                      metric = "ROC",
                      trControl = ctrl,
                      tuneGrid = grid)

#rose
ctrl$sampling <- "rose"
set.seed(1)
rose_inside_nb <- train(x = trainSet[,-17], 
                        y = trainSet$Churn,
                        method = "naive_bayes",
                        metric = "ROC",
                        trControl = ctrl,
                        tuneGrid = grid)

#orig fit
ctrl$sampling <- NULL
set.seed(1)
orig_fit_nb <- train(x = trainSet[,-17], 
                     y = trainSet$Churn, 
                     method = "naive_bayes",
                     metric = "ROC",
                     trControl = ctrl,
                     tuneGrid = grid)

inside_models_nb <- list(original = orig_fit_nb,
                         down = down_inside_nb,
                         up = up_inside_nb,
                         ROSE = rose_inside_nb)
inside_resampling_nb <- resamples(inside_models_nb)
summary(inside_resampling_nb, metric = "ROC")
#Najbolja je up
#ROC 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#original 0.7666245 0.7966941 0.8134756 0.8128483 0.8259276 0.8591385    0
#down     0.7757595 0.8037017 0.8134914 0.8134851 0.8235183 0.8504026    0
#up       0.7508916 0.7998792 0.8152295 0.8145790 0.8308718 0.8552184    0
#ROSE     0.7631472 0.7963205 0.8084797 0.8099266 0.8228967 0.8461272    0

nb3.pred <- predict(up_inside_nb$finalModel, newdata = testSet[,-17], type = "class")

nb3.cm <- table(true = testSet$Churn, predicted = nb3.pred)
nb3.cm
#     predicted
#true  Yes  No
#Yes 304  69
#No  370 664
eval.nb3 <- getEvaluationMetrics(nb3.cm)
eval.nb3
# Accuracy Precision    Recall        F1 
#0.6879886 0.4510386 0.8150134 0.5807068 

data.frame(rbind(nb1.cm, nb2.cm,nb3.cm))
data.frame(rbind(eval.nb1,eval.nb2,eval.nb3), row.names = c("prvi","drugi","treći"))
#       Accuracy Precision    Recall        F1
#prvi  0.7142857 0.4760331 0.7721180 0.5889571
#drugi 0.7562189 0.5298805 0.7131367 0.6080000
#treći 0.6879886 0.4510386 0.8150134 0.5807068