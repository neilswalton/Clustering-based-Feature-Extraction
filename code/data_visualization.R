library(pracma)
library(ggplot2)

mushroom <- read.csv("~/Documenten/Clustering-based-Feature-Extraction/code/mushroom_results_02_12_2018:13:06_5_0.005.csv", skip=1, header=T,sep=",")

mushroom_mean <- list()
j <- 1
for (i in 1:nrow(mushroom)){
  if(strcmp(as.character(mushroom[i,]$score_type), "accuracy")){
    #temp <- list()
    #temp[1] <- mushroom[i,]$mean
    #temp[2] <- as.character(mushroom[i,]$method)
    #temp[3] <- as.character(mushroom[i,]$classifier)
    mushroom_mean[[j]] <- c("score" = as.numeric(mushroom[i,]$mean), "method" = as.character(mushroom[i,]$method), "classifier" = as.character(mushroom[i,]$classifier))
    j <- j+1
  }
}

mushroom_mean <- as.data.frame(do.call(rbind, mushroom_mean))
mushroom_mean$score <- as.numeric(levels(mushroom_mean$score))[mushroom_mean$score]
barplot(as.numeric(mushroom_mean$score), col = ifelse(,'red','green'), ylim=c(0,1), names.arg = mushroom_mean$method)

ggplot(mushroom_mean, aes(factor(method), score, fill = classifier))+geom_bar(stat="identity", position = "dodge")+scale_fill_brewer(palette = "Set1")+labs(x = "Feature Extraction Method", y = "Accuracy") 
