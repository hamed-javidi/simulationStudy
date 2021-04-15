setwd("~/simulationStudy/scripts/results/Variational_Autoencoder")

library('data.table')
library(ggplot2)

vae = fread('mag_cohorts_NEW_Simulation_Cohort_34_18323_2_3000_4.5_0.5VAE_generated_dataset.csv', sep = ",", data.table = FALSE)
colnames(vae) = c("ID", "SBP", "Time", "Class", "Gen_ID")

original = fread('~/simulationStudy/dataMarch2020/data_subset_test/mag_cohorts/Simulation_Cohort_34_18323/_Simulated_Cohort_2_3000_4.5_0.5.csv', sep = ",", data.table = FALSE)

p1<-ggplot()+geom_line(data=vae,aes(x=Time,y=SBP,color=as.factor(Class),group=interaction(ID,Class,Gen_ID)))
p1
#===========================================================================================================
id = 1
indx = which(vae[, 1] == id)
vae_subset = vae[indx, ]
colnames(vae_subset) = c("ID", "SBP", "Time", "Class", "Gen_ID")

indx = which(original[, 1] == id)
original_subset = cbind(original[indx, c(1:3, 5)], rep("org", length(indx)))
colnames(original_subset) = c("ID", "SBP", "Time", "Class", "Gen_ID")

df = rbind(original_subset, vae_subset)
df$Gen_ID <- factor(df$Gen_ID, levels = c("orig", "mu 0", "mu 1", "mu 2", "mu 3", "mu 4"))

# The palette with black:
# cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
p1<-ggplot()+geom_line(data=df,aes(x=Time,y=SBP,color=as.factor(Gen_ID),group=interaction(ID,Class,Gen_ID)))+
  theme(legend.position = "bottom",panel.background = element_rect(fill="white"),axis.line = element_line(color = "black"))+
  scale_color_viridis_d()  # scale_color_manual(values="set1") #
p1
#===========================================================================================================