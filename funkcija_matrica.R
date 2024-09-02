getEvaluationMetrics <- function(cm) {
  TP <- cm[1,1]
  TN <- cm[2,2]
  FP <- cm[2,1]
  FN <- cm[1,2]
  
  accuracy = sum(diag(cm)) / sum(cm)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  F1 <- (2 * precision * recall) / (precision + recall)
  
  c(Accuracy = accuracy, 
    Precision = precision, 
    Recall = recall, 
    F1 = F1)
  
}
