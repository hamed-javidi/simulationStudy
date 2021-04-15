library(data.table)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(cowplot) 

LogReg <- function(df, Code){
     mylogit <- glm(Class ~ Percentile,  data = df, family = "binomial")
     # print(summary(fit)
     # newdata1 <- with(df, data.frame(BMI = seq(1,80,1)))
     # newdata1$Code <- predict(mylogit, newdata = newdata1, type = "response")
     # print(head(newdata1))
     # base <- ggplot(data=newdata1)+ ylab("Probability") +
          # geom_line(aes(x=BMI, y=Code))
     # ggsave(filename=paste(Code,".jpeg"), height=4, width=6, units="in",  device="jpeg", dpi=300, path = "../plot/codes")
     return(coef(summary(mylogit))[,c('Estimate','Std. Error', 'z value', 'Pr(>|z|)')])
}


# icd9_file <- fread('../Output/LogisticReg_resultsICD9Code.csv')
# icd10_file <- fread('../Output/LogisticReg_resultsICD10Code.csv')
# icd9_pvalues <- icd9_file$'BMI-Pr(>|z|'
# icd10_pvalues <- icd10_file$"BMI-Pr(>|z|"
# icd9_file$icd9_adjusted_pvalues = p.adjust(icd9_pvalues, method = "fdr", n= length(icd9_pvalues))
# write.csv(icd9_file, file = paste0( "../Output/LogisticReg_resultsICD9Code_v1.csv"), row.names=FALSE)
# 
# icd10_file$icd10_adjusted_pvalues = p.adjust(icd10_pvalues, method = "fdr", n= length(icd10_pvalues))
# write.csv(icd10_file, file = paste0( "../Output/LogisticReg_resultsICD10Code_v1.csv"), row.names=FALSE)
# print(icd9_adjusted_pvalues)
# print(LogReg(dt,'278'))