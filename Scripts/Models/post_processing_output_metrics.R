#=================================================================================================
# Metric: Accuracy and Success Indicator
ACC_SUCC = function(y_true, y_pred){
  ACC = length(which(y_true == y_pred))/length(y_true)
  threshold = 0.99        # Success threshold
  SUCC = 0
  if (ACC >= threshold){
    SUCC = 1
  } 
  output = list(ACC, SUCC)
  return(output)
}
#=================================================================================================
# Extract noise level from data set name
noise_level = function(cohort_name){
  l = nchar(cohort_name)
  underline_locations = gregexpr(pattern ='_', cohort_name)[[1]]
  start_pos = underline_locations[length(underline_locations) - 1] + 1
  end_pos = tail(underline_locations, n= 1) - 1
  level = as.numeric(substr(cohort_name, start_pos , end_pos))
  return(level)
}
#=================================================================================================
# Compute class probability
max_prob = function(prob_each_class){
  splitted = strsplit(prob_each_class, "[|]")[[1]]
  out = max(as.numeric(unlist(splitted)))
  return(out)
}

#=================================================================================================
# Compute Metrics
ClassificationComputeMetrics = function(cohort_index, classifier_names){
  # Initialization
  
  
  cohort_types = c('mag', 'shape')
  
  # Create Dictionary
  dict = hash()
  dict[[cohort_types[1]]] = c(0, 0.1, 0.25, 0.5)
  dict[[cohort_types[2]]] = c(0, 0.1, 0.25, 0.5)
  raw_res = vector()
  mean_res = vector()
  correct_prob_dist = vector()
  cohort_type = cohort_types[cohort_index]
  
  for (classifier in classifier_names){ # Loop over different classifiers
    print(classifier)
    output = fread(paste0( classifier, '_outputs.csv'), sep = ',', data.table = FALSE)
    cohort_names = unique(output$cohort_name)
    f = first( which(grepl(cohort_type, cohort_names) == TRUE) )
    l = last( which(grepl(cohort_type, cohort_names) == TRUE) )
    level = dict[[cohort_type]]
    sub_cohort_names = cohort_names[f:l]
    noise = lapply(sub_cohort_names, noise_level)
    for (group_level in level){ # Loop over cohort's sub-groups based on dataset's noise level
      print(paste0('group level = ', group_level))
      index_grouped_cohort = which( noise == group_level)
      noise_level_res = vector()
      for (index in index_grouped_cohort){
        print(paste0('cohort name = ', sub_cohort_names[index]))
        splitted = strsplit(sub_cohort_names[index], "/")[[1]]
        file_name = paste0(splitted[2], '_', splitted[3], substr(splitted[4], 18, nchar(splitted[4])-4), 
                           '_', classifier, '_history.csv')
        hist = fread(paste0(path, 'models/', file_name), sep = ',', data.table = FALSE)
        
        f_pattern = first(which(output$cohort_name == sub_cohort_names[index]))
        l_pattern = last(which(output$cohort_name == sub_cohort_names[index]))
        sub_output = output[f_pattern:l_pattern, ]
        # Compute Metrics
        y_true = sub_output$original_class[which(sub_output$train_test == 'Test')]
        y_pred = sub_output$predicted_class[which(sub_output$train_test == 'Test')]
        acc_sc = ACC_SUCC(y_true, y_pred)
        ACC = acc_sc[[1]]                             # Accuracy
        SF = acc_sc[[2]]                              # Success flag
        TRT = sub_output$training_time[1]             # Training Time
        ITR = nrow(hist)                              # Training iterations
        ACC_ITR = ACC / ITR                           # Accuracy per iteration
        metrics = c(ACC, SF, TRT, ITR, ACC_ITR)
        noise_level_res = rbind(noise_level_res, metrics)
        raw_res = rbind(raw_res, c(classifier, cohort_type, sub_cohort_names[index], group_level, metrics))
        
        probs = sub_output$prob_each_class[which(sub_output$original_class == sub_output$predicted_class & sub_output$train_test == 'Test')]
        correct_class_probs = as.numeric(as.character(lapply(probs, max_prob)))
        correct_prob_dist <- rbind(correct_prob_dist, cbind(rep(classifier, length(correct_class_probs)), correct_class_probs))
      }
      mean_res = rbind(mean_res, c(classifier, cohort_type, group_level, colMeans(noise_level_res))) 
    }
  }
  return(list(raw_res, mean_res, correct_prob_dist))
}

#=================================================================================================
ClusteringComputeMetrics = function(cohort_index, clustering_names, clustering_methods){
  # Initialization
  cohort_types = c('mag', 'shape')
  
  # Create Dictionary
  dict = hash()
  dict[[cohort_types[1]]] = c(0, 0.1, 0.25, 0.5)
  dict[[cohort_types[2]]] = c(0, 0.1, 0.25, 0.5)
  raw_res = vector()
  mean_res = vector()
  correct_prob_dist = vector()
  cohort_type = cohort_types[cohort_index]
  
  for (classifier in clustering_names) { # Loop over different clustering
    for (clustering_method in clustering_methods){ # Loop over different clustering_method
      print(paste(classifier, clustering_method))
      output = fread(paste0( classifier,'_',clustering_method, '_outputs.csv'), sep = ',', data.table = FALSE)
      cohort_names = unique(output$cohort_name)
      f = first( which(grepl(cohort_type, cohort_names) == TRUE) )
      l = last( which(grepl(cohort_type, cohort_names) == TRUE) )
      level = dict[[cohort_type]]
      sub_cohort_names = cohort_names[f:l]
      noise = lapply(sub_cohort_names, noise_level)
      for (group_level in level){ # Loop over cohort's sub-groups based on dataset's noise level
        print(paste0('group level = ', group_level))
        index_grouped_cohort = which( noise == group_level)
        noise_level_res = vector()
        for (index in index_grouped_cohort){
          # print(paste0('cohort name = ', sub_cohort_names[index]))
          # splitted = strsplit(sub_cohort_names[index], "/")[[1]]
          # file_name = paste0(splitted[2], '_', splitted[3], substr(splitted[4], 18, nchar(splitted[4])-4), 
          #                    '_', classifier, '_', clustering_method, '_Clustering.csv')
          # hist = fread(paste0(path, 'models/', file_name), sep = ',', data.table = FALSE)
          
          f_pattern = first(which(output$cohort_name == sub_cohort_names[index]))
          l_pattern = last(which(output$cohort_name == sub_cohort_names[index]))
          sub_output = output[f_pattern:l_pattern, ]
          # Compute Metrics
          orig_class = as.numeric(mean(sub_output$predicted_cluster[which(sub_output$original_class == "Original")]) >= 0.5)
          derived_class = as.numeric(mean(sub_output$predicted_cluster[which(sub_output$original_class == "Derived")]) >= 0.5)
          y_true = ifelse (sub_output$original_class == "Original", orig_class, derived_class )
          y_pred = sub_output$predicted_cluster
          acc_sc = ACC_SUCC(y_true, y_pred)
          ACC = acc_sc[[1]]                             # Accuracy
          SF = acc_sc[[2]]                              # Success flag
          TRT = sub_output$clustering_time[1]           # Training Time
          ITR = sub_output$clustering_iteration[1]         # Training iterations
          ACC_ITR = ACC / ITR                           # Accuracy per iteration
          metrics = c(ACC, SF, TRT, ITR, ACC_ITR)
          noise_level_res = rbind(noise_level_res, metrics)
          raw_res = rbind(raw_res, c(classifier, clustering_method, cohort_type, sub_cohort_names[index], group_level, metrics))
        }
        mean_res = rbind(mean_res, c(classifier, clustering_method, cohort_type, group_level, colMeans(noise_level_res))) 
      }
    }
  }
  return(list(raw_res, mean_res))
}
#=================================================================================================
#=================================================================================================
#=================================================================================================
#=================================================================================================
#=================================================================================================
##################################################################################################
# Set Working Directory and Load Libraries
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
path = paste0(getwd(),'/results/')
# path = "/Users/javidih/OneDrive - Cleveland State University/Cleveland Clinic/ChildhoodObesity/Output/"
setwd(path)


library('data.table')
library('hash')
library(doFuture)
registerDoFuture()
plan(multiprocess)

classifier_names = c('MLP', 'TSF_MLP', 'GAF_CNN', 'FCNN', 'ResNet', 'RNN', 'CNN_RNN', 'CNN_MLP_Autoencoder', 'RNN_Autoencoder')

params = ClassificationComputeMetrics(1, classifier_names)
raw_res = params[[1]]
mean_res = params[[2]]
correct_prob_dist = params[[3]]
colnames(raw_res) <- c("classifier", "cohort_type", "data_set", "noise_level", "accuracy", "success_rate", "training_time", "training_iterations", "accuracy_per_iteration")
write.csv(raw_res, file = paste0(path, "integrated_results/MAG_Metrics_All_Classifiers_All_Cohorts.csv"), row.names=FALSE)
colnames(mean_res) <- c("classifier", "cohort_type", "noise_level", "accuracy", "success_rate", "training_time", "training_iterations", "accuracy_per_iteration")
write.csv(mean_res, file = paste0(path, "integrated_results/MAG_Mean_Metrics_All_Classifiers_All_Cohorts.csv"), row.names=FALSE)
colnames(correct_prob_dist) <- c("classifier", "probability")
write.csv(correct_prob_dist, file = paste0(path, "integrated_results/MAG_Class_Prob_All_Classifiers_All_Cohorts.csv"), row.names=FALSE)

params = ClassificationComputeMetrics(2, classifier_names)
raw_res = params[[1]]
mean_res = params[[2]]
correct_prob_dist = params[[3]]
colnames(raw_res) <- c("classifier", "cohort_type", "data_set", "noise_level", "accuracy", "success_rate", "training_time", "training_iterations", "accuracy_per_iteration")
write.csv(raw_res, file = paste0(path, "integrated_results/Shape_Metrics_All_Classifiers_All_Cohorts.csv"), row.names=FALSE)
colnames(mean_res) <- c("classifier", "cohort_type", "noise_level", "accuracy", "success_rate", "training_time", "training_iterations", "accuracy_per_iteration")
write.csv(mean_res, file = paste0(path, "integrated_results/Shape_Mean_Metrics_All_Classifiers_All_Cohorts.csv"), row.names=FALSE)
colnames(correct_prob_dist) <- c("classifier", "probability")
write.csv(correct_prob_dist, file = paste0(path, "integrated_results/Shape_Class_Prob_All_Classifiers_All_Cohorts.csv"), row.names=FALSE)
#=================================================================================================
#=================================================================================================
#=================================================================================================

# clustering_names = c('RNN', 'CNN_MLP')
# clustering_method = c('kmeans', 'GMM')
# print("Mag cogort...")
# params = ClusteringComputeMetrics(1, clustering_names, clustering_method)
# raw_res = params[[1]]
# mean_res = params[[2]]
# colnames(raw_res) <- c("model", "method", "cohort_type", "data_set", "noise_level", "accuracy", "success_rate", "clustering_time", "clustering_iterations", "accuracy_per_iteration")
# write.csv(raw_res, file = paste0(path, "integrated_results/MAG_Metrics_All_Clustering_All_Cohorts.csv"), row.names=FALSE)
# colnames(mean_res) <- c("model", "method", "cohort_type", "noise_level", "accuracy", "success_rate", "clustering_time", "clustering_iterations", "accuracy_per_iteration")
# write.csv(mean_res, file = paste0(path, "integrated_results/MAG_Mean_Metrics_All_Clustering_All_Cohorts.csv"), row.names=FALSE)
# 
# print("Shape cogort...")
# params = ClusteringComputeMetrics(2, clustering_names, clustering_method)
# raw_res = params[[1]]
# mean_res = params[[2]]
# colnames(raw_res) <- c("model", "method", "cohort_type", "data_set", "noise_level", "accuracy", "success_rate", "clustering_time", "clustering_iterations", "accuracy_per_iteration")
# write.csv(raw_res, file = paste0(path, "integrated_results/Shape_Metrics_All_Clustering_All_Cohorts.csv"), row.names=FALSE)
# colnames(mean_res) <- c("model", "method", "cohort_type", "noise_level", "accuracy", "success_rate", "clustering_time", "clustering_iterations", "accuracy_per_iteration")
# write.csv(mean_res, file = paste0(path, "integrated_results/Shape_Mean_Metrics_All_Clustering_All_Cohorts.csv"), row.names=FALSE)


