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
remove.packages("bnlearn")
install.packages("bnlearn")

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

columns <- unlist(lapply(dataset, is.numeric))
datasetNumericVar <- dataset[,columns]
str(datasetNumericVar)

apply(datasetNumericVar,2,as.numeric)
datasetNumericDataFrame <- as.data.frame(datasetNumericVar)
library(bnlearn)
discretized <- discretize(
  datasetNumericDataFrame$tenure,
  cuts = 2,
  labels = NULL,
  prefix = "bin",
  keep_na = TRUE,
  infs = FALSE
)
discretized
