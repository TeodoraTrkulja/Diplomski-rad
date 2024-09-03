dataset<-read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv",stringsAsFactors = F)
str(dataset)

#Instaliranje paketa i učitavanje paketa u okviru biblioteke
install.packages("dplyr")
install.packages("ggplot2")
library(ggplot2)
library(dplyr)

#provera nedostajućih vrednosti
apply(dataset,MARGIN = 2,FUN = function(x) sum(is.na(x)))
#TotalCharges ima 11 NA
apply(dataset,MARGIN = 2,FUN = function(x) sum(x==""))
apply(dataset,MARGIN = 2,FUN = function(x) sum(x==" "))
apply(dataset,MARGIN = 2,FUN = function(x) sum(x=="-"))

#Sređivanje TotalCharges,p<0.5 pa NA vrednost menjam medijanom
shapiro.test(sample(dataset$TotalCharges,size=5000))
medianTotalCharges<-median(dataset$TotalCharges,na.rm = T)
dataset$TotalCharges[is.na(dataset$TotalCharges)]<-medianTotalCharges

#isključivanje opisnih varijabli i varijabli sa prevelikim brojem različitih vrednosti
dataset$customerID<-NULL

#provera vrednosti char i num promenljivih
summary(dataset$gender)
summary(dataset$SeniorCitizen)
summary(dataset$Partner)
summary(dataset$Dependents)
summary(dataset$tenure)
summary(dataset$PhoneService)
summary(dataset$MultipleLines)
summary(dataset$InternetService)
summary(dataset$OnlineSecurity)
summary(dataset$OnlineBackup)
summary(dataset$DeviceProtection)
summary(dataset$TechSupport)
summary(dataset$StreamingTV)
summary(dataset$StreamingMovies)
summary(dataset$Contract)
summary(dataset$PaperlessBilling)
summary(dataset$PaymentMethod)
summary(dataset$MonthlyCharges)
summary(dataset$TotalCharges)
summary(dataset$Churn)

#transformisanje char varijable u factor
dataset$gender<-as.factor(dataset$gender)
dataset$Contract<-as.factor(dataset$Contract)
dataset$SeniorCitizen<-as.factor(dataset$SeniorCitizen)
dataset$Partner<-as.factor(dataset$Partner)
dataset$PhoneService<-as.factor(dataset$PhoneService)
dataset$MultipleLines<-as.factor(dataset$MultipleLines)
dataset$InternetService<-as.factor(dataset$InternetService)
dataset$Dependents<-as.factor(dataset$Dependents)
dataset$OnlineSecurity<-as.factor(dataset$OnlineSecurity)
dataset$OnlineBackup<-as.factor(dataset$OnlineBackup)
dataset$DeviceProtection<-as.factor(dataset$DeviceProtection)
dataset$TechSupport<-as.factor(dataset$TechSupport)
dataset$StreamingTV<-as.factor(dataset$StreamingTV)
dataset$StreamingMovies<-as.factor(dataset$StreamingMovies)
dataset$PaperlessBilling<-as.factor(dataset$PaperlessBilling)
dataset$PaymentMethod<-as.factor(dataset$PaymentMethod)
dataset$Churn <- factor(dataset$Churn, levels = c("Yes", "No"))

#Ispitivanje značajnosti varijabli preko plotova

#Density plots (geom_density) za numeričke varijable
numeric_vars <- c("tenure", "MonthlyCharges", "TotalCharges")

for (var in numeric_vars) {
  print(
    ggplot(dataset, aes_string(x = var, fill = "Churn", color = "Churn")) +
      geom_density(alpha = 0.65) +
      labs(title = paste("Density plot for", var, "vs Churn"),
           x = var,
           y = "Density",
           fill = "Churn",
           color = "Churn")
  )
}
#TotalCharges, što je vrednost manja, veći je broj izlazne varijable sa vrednošću Yes (varijabla je značajna)
#MonthlyCharges,negativna klasa No najviše obuhvata niske vrednosti ove varijable, dok Yes vrednosti 75-100 (varijabla je značajna)
#tenure, negativna klasa No najviše obuhvata visoke vrednosti ove varijable >60, dok Yes vrednosti <20 (varijabla je značajna)

#Za kategoričke varijable, bar graf 
categorical_vars <- c( "SeniorCitizen", "Partner", "Dependents",
                       "InternetService",
                      "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                      "TechSupport", "StreamingTV", "StreamingMovies",
                      "Contract", "PaperlessBilling", "PaymentMethod")


for (var in categorical_vars) {
  print(
    ggplot(dataset, aes_string(x = var, fill = "Churn")) +
      geom_bar(position = "fill", width = 0.6) +
      labs(title = paste("Bar plot for", var, "vs Churn"),
           x = var,
           y = "Proportion",
           fill = "Churn")+
      theme_minimal()
  )
}
#PaymentMethod,najviše vrednosti pozitivne klase Yes obuhvata Electronic check dok su ostale slične (varijabla je za sad značajna,ostavljam je u razmatranju)
#PaperlessBilling,više vrednosti Yes obuhvata kada postoje PaperlessBilling (varijabla je značajna)
#Contract,kada je vrednost Month-to-month vrednost klase Yes je najveća dok što je veća dužina ugovora, klasa Yes je znatno manja (varijabla je značajna)
#StreamingMovies,razlike između vrndnosti varijable StreamingMovies kada je Yes i kada je No gotovo da nema, dok kada je vrednost jednaka No internet service Churn je velikom većinom negativan (varijabla je za sad značajna,ostavljam je u razmatranju)
#StreamingTV,isto kao kod StreamingMovies
#TechSupport,slično kao kod StreamingMovies samo je veća razlike između Yes i No kategorija kod te varijable (varijabla je značajna)
#DeviceProtection,isto kao kod StreamingMovies
#OnlineBackup,isto kao kod StreamingMovies
#OnlineSecurity,slično kao kod TechSupport
#InternetService, vrednosti DSL i No ove varijable imaju slične vrednosti dok za Fiber optic mnogo je veća zastupljenost Yes klase varijable Churn nego kod ostale dve (varijabla je za sad značajna)
#MultipleLines, vrednosti Yes i No varijable Churn su u istom procentu zastupljene kod sve tri vrednosti varijable MultipleLines (varijabla nije značajna,isključujem je iz daljeg razmatranja)
#PhoneService, vrednosti Yes i No varijable Churn su u istom procentu zastupljene kod sve tri vrednosti varijable MultipleLines (varijabla nije značajna,isključujem je iz daljeg razmatranja)
#Dependents,Partner,SeniorCitizen postoji razlika između Yes i No 
#Gender, vrednosti Yes i No varijable Churn su u istom procentu zastupljene kod sve tri vrednosti varijable MultipleLines (varijabla nije značajna,isključujem je iz daljeg razmatranja)

dataset$MultipleLines<-NULL
dataset$PhoneService<-NULL
dataset$gender<-NULL

saveRDS(dataset,file = "preprocessed_dataset.RDS")

