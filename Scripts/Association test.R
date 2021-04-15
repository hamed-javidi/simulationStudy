library(vroom)
library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
print("Current path is: ")
print(getw)

diag <- vroom('../Output/final_diag.csv')
bmi <- vroom('../Output/final_BMI_data.csv')
being_patient <- vroom('../Output/being_patient_list.csv', delim ='\n' )
icd9_codes <- vroom('../Input/ICD9Code_T2D.txt', delim = '\t')
icd10_codes <- vroom('../Input/ICD10Code_T2D.txt', delim = '\t')
age_limit = 35


# 1- apply age limit on diag 
age_limit = age_limit * 12
diag = subset(diag, subset = age <= age_limit)

# 2- For each code 
for (i in range(2)){
  if(i == 1){
    codes = icd9_codes
    col_name = 'ICD9Code'
  }
  else{
    codes = icd10_codes
    col_name = 'ICD10Code'
  }
  for(j in codes$code){
    #    a. Find detection date
    diag_class_1 = subset(diag, subset = col_name == j)
    diag_class_1 %>% 
      group_by(PatientID) %>% 
      summarise(DetectionDate = min(ContactDate))
    
    #    b. Truncate trajectory for class 1 + keep whole trajectory for class 0
    
    
    #    c. Keep class1 + check class 0 to be in being_patient list
    
    #    d. find the max percentile for each individual
    
    #    e. perform logreg
  }
}

mylogit <- glm(Class ~ Percentile,  data = df, family = "binomial")
