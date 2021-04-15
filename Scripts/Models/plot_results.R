##################################################################################################
# Set Working Directory and Load Libraries
#Server path
# path = "/home/javidih/ChildhoodObesity/Scripts/Models/results/integrated_results/"
#=================================================================================
#local path

path = "/Volumes/javidih/ChildhoodObesity/Scripts/Models/results/integrated_results/"


print(path)
setwd(path)

library('plyr')
library('hash')
library('ggplot2')
library('data.table')



# if (!require("devtools")) {
#   install.packages("devtools")
# }
# 
# devtools::install_github("b0rxa/scmamp")
# install.packages("../../scmamp_0.2.55.tar.gz", reps=NULL)
library('scmamp')
#=================================================================================================
# Initialization

cohort_types = c('mag', 'shape')

# Create Dictionary
dict = hash()
dict[[cohort_types[1]]] = c(0, 0.1, 0.25, 0.5)
dict[[cohort_types[2]]] = c(0, 0.1, 0.25, 0.5)


#=================================================================================================
# Part 1: Load Mag cohorts Results
#=================================================================================================


raw_res = fread(paste0(path, "MAG_Metrics_All_Classifiers_All_Cohorts.csv"), sep = ',', data.table = FALSE)

mean_res = fread(paste0(path, "MAG_Mean_Metrics_All_Classifiers_All_Cohorts.csv"), sep = ',', data.table = FALSE)

correct_prob_dist = fread(paste0(path, "MAG_Class_Prob_All_Classifiers_All_Cohorts.csv"), sep = ',', data.table = FALSE)
#=================================================================================================
# Plot density of classifiers' decision confidence
print("Part 1 ...")
# correct_prob_dist <- as.data.frame(correct_prob_dist)
# correct_prob_dist[, 2] = as.numeric(as.character(correct_prob_dist[,2]))
correct_prob_dist$classifier <- factor(correct_prob_dist$classifier, levels = c('MLP', 'TSF_MLP', 'GAF_CNN', 'FCNN', 'ResNet', 'RNN', 'CNN_RNN', 'CNN_MLP_Autoencoder', 'RNN_Autoencoder'))
mu <- ddply(correct_prob_dist, "classifier", summarise, grp.median=median(probability))
p<-ggplot(correct_prob_dist, aes(x=probability, fill=classifier)) + #alpha=0.4
  geom_density(alpha=0.4) + 
  facet_grid(classifier ~ ., scales = "free_y") + theme_classic() +
  coord_cartesian(xlim = c(0.9, 1))+
  geom_vline(data=mu, aes(xintercept=grp.median, color=classifier), linetype="dashed") + 
  theme(strip.text.y = element_text(size = 4, colour = "blue", angle = 90))+
  theme(axis.text=element_text(size=6), axis.title=element_text(size=6,face="bold"))+
  labs(x="Probability of Correctly Predicted Classes", y = "Density") +
  theme(legend.position="none") #+geom_histogram(aes(y=..density..), alpha=0.5, position="identity")
ggsave(filename="Mag_Density_plot.jpg", height=4, width=6, units="in",  device="jpeg", dpi=300, path = "../plot")

length(which(correct_prob_dist$classifier =='GAF_CNN'))
#=================================================================================================
# Critical Difference Diagram
data_CDiff = raw_res[which(raw_res$classifier=='MLP'), 5]
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='TSF_MLP'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='GAF_CNN'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='FCNN'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='ResNet'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='RNN'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='CNN_RNN'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='CNN_MLP_Autoencoder'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='RNN_Autoencoder'), 5])
data_CDiff = as.data.frame(data_CDiff)
colnames(data_CDiff) = c('MLP', 'TSF_MLP', 'GAF_CNN', 'FCNN', 'ResNet', 'RNN', 'CNN_RNN', 'CNN_MLP_Autoencoder', 'RNN_Autoencoder')
row.names(data_CDiff) <- raw_res[which(raw_res$classifier=='MLP'), 3]
plotCD(data_CDiff, alpha=0.05)


#=================================================================================================
# Part 2: Load Shape cohorts Results
#=================================================================================================
print("Part 2 ...")
raw_res = fread(paste0(path, "Shape_Metrics_All_Classifiers_All_Cohorts.csv"), sep = ',', data.table = FALSE)

mean_res = fread(paste0(path, "Shape_Mean_Metrics_All_Classifiers_All_Cohorts.csv"), sep = ',', data.table = FALSE)

correct_prob_dist = fread(paste0(path, "Shape_Class_Prob_All_Classifiers_All_Cohorts.csv"), sep = ',', data.table = FALSE)
#=================================================================================================
# Plot density of classifiers' decision confidence

# correct_prob_dist <- as.data.frame(correct_prob_dist)
# correct_prob_dist[, 2] = as.numeric(as.character(correct_prob_dist[,2]))
correct_prob_dist$classifier <- factor(correct_prob_dist$classifier, levels = c('MLP', 'TSF_MLP', 'GAF_CNN', 'FCNN', 'ResNet', 'RNN', 'CNN_RNN', 'CNN_MLP_Autoencoder', 'RNN_Autoencoder'))
mu <- ddply(correct_prob_dist, "classifier", summarise, grp.median=median(probability))
p<-ggplot(correct_prob_dist, aes(x=probability, fill=classifier)) + #alpha=0.4
  geom_density(alpha=0.4)+ 
  facet_grid(classifier ~ ., scales = "free_y") + theme_classic() +
  geom_vline(data=mu, aes(xintercept=grp.median, color=classifier), linetype="dashed") +
  theme(strip.text.y = element_text(size = 4, colour = "blue", angle = 90))+
  theme(axis.text=element_text(size=6), axis.title=element_text(size=6,face="bold"))+
  labs(x="Probability of Correctly Predicted Classes", y = "Density") +
  theme(legend.position="none") #+geom_histogram(aes(y=..density..), alpha=0.5, position="identity")
ggsave(filename="Shape_Density_plot.jpg", height=4, width=6, units="in",  device="jpeg", dpi=300, path = "../plot")

length(which(correct_prob_dist$classifier =='GAF_CNN'))
#=================================================================================================
# Critical Difference Diagram

data_CDiff = raw_res[which(raw_res$classifier=='MLP'), 5]
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='TSF_MLP'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='GAF_CNN'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='FCNN'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='ResNet'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='RNN'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='CNN_RNN'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='CNN_MLP_Autoencoder'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='RNN_Autoencoder'), 5])
data_CDiff = as.data.frame(data_CDiff)
colnames(data_CDiff) = c('MLP', 'TSF_MLP', 'GAF_CNN', 'FCNN', 'ResNet', 'RNN', 'CNN_RNN', 'CNN_MLP_Autoencoder', 'RNN_Autoencoder')
row.names(data_CDiff) <- raw_res[which(raw_res$classifier=='MLP'), 3]
plotCD(data_CDiff, alpha=0.05)


#==================================================================================================
#Plot overall
#==================================================================================================

raw_res1 = fread(paste0(path, "MAG_Metrics_All_Classifiers_All_Cohorts.csv"), sep = ',', data.table = FALSE)
mean_res1 = fread(paste0(path, "MAG_Mean_Metrics_All_Classifiers_All_Cohorts.csv"), sep = ',', data.table = FALSE)
correct_prob_dist1 = fread(paste0(path, "MAG_Class_Prob_All_Classifiers_All_Cohorts.csv"), sep = ',', data.table = FALSE)

raw_res2 = fread(paste0(path, "Shape_Metrics_All_Classifiers_All_Cohorts.csv"), sep = ',', data.table = FALSE)
mean_res2 = fread(paste0(path, "Shape_Mean_Metrics_All_Classifiers_All_Cohorts.csv"), sep = ',', data.table = FALSE)
correct_prob_dist2 = fread(paste0(path, "Shape_Class_Prob_All_Classifiers_All_Cohorts.csv"), sep = ',', data.table = FALSE)

raw_res = rbind(raw_res1, raw_res2)
mean_res = rbind(mean_res1, mean_res2)
correct_prob_dist = rbind(correct_prob_dist1,correct_prob_dist2)

correct_prob_dist$classifier <- factor(correct_prob_dist$classifier, levels = c('MLP', 'TSF_MLP', 'GAF_CNN', 'FCNN', 'ResNet', 'RNN', 'CNN_RNN', 'CNN_MLP_Autoencoder', 'RNN_Autoencoder'))
mu <- ddply(correct_prob_dist, "classifier", summarise, grp.median=median(probability))
p<-ggplot(correct_prob_dist, aes(x=probability, fill=classifier)) + #alpha=0.4
  geom_density(alpha=0.4)+ 
  facet_grid(classifier ~ ., scales = "free_y") + theme_classic() +
  geom_vline(data=mu, aes(xintercept=grp.median, color=classifier), linetype="dashed") +
  theme(strip.text.y = element_text(size = 3, colour = "blue", angle = 90))+
  theme(axis.text=element_text(size=4), axis.title=element_text(size=4,face="bold"))+
  labs(x="Probability of Correctly Predicted Classes", y = "Density") +
  theme(legend.position="none") #+geom_histogram(aes(y=..density..), alpha=0.5, position="identity")
ggsave(filename="Overall_Density_plot.jpg", height=3, width=6, units="in",  device="jpeg", dpi=300, path = "../plot")

# Critical Difference Diagram for accuracy
data_CDiff = raw_res[which(raw_res$classifier=='MLP'), 5]
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='TSF_MLP'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='GAF_CNN'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='FCNN'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='ResNet'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='RNN'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='CNN_RNN'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='CNN_MLP_Autoencoder'), 5])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='RNN_Autoencoder'), 5])
data_CDiff = as.data.frame(data_CDiff)
colnames(data_CDiff) = c('MLP', 'TSF_MLP', 'GAF_CNN', 'FCNN', 'ResNet', 'RNN', 'CNN_RNN', 'CNN_MLP_Autoencoder', 'RNN_Autoencoder')
row.names(data_CDiff) <- raw_res[which(raw_res$classifier=='MLP'), 3]
plotCD(data_CDiff, alpha=0.05)

# Critical Difference Diagram for traning time
data_CDiff = raw_res[which(raw_res$classifier=='MLP'), 7]
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='TSF_MLP'), 7])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='GAF_CNN'), 7])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='FCNN'), 7])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='ResNet'), 7])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='RNN'), 7])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='CNN_RNN'), 7])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='CNN_MLP_Autoencoder'), 7])
data_CDiff = cbind(data_CDiff, raw_res[which(raw_res$classifier=='RNN_Autoencoder'), 7])
data_CDiff = as.data.frame(data_CDiff)
colnames(data_CDiff) = c('MLP', 'TSF_MLP', 'GAF_CNN', 'FCNN', 'ResNet', 'RNN', 'CNN_RNN', 'CNN_MLP_Autoencoder', 'RNN_Autoencoder')
row.names(data_CDiff) <- raw_res[which(raw_res$classifier=='MLP'), 3]
plotCD(data_CDiff, alpha=0.05)
