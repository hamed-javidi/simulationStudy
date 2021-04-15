#setwd("~/Desktop/MacDown/Downloads/HypertensionSimulationStudy")
library(ggplot2)
library(dplyr)
library(reshape2)
library(gridExtra)
args= commandArgs(TRUE)
sample_nums<-as.numeric(args[2])
noise_mags<-seq(as.numeric(args[5])/10,as.numeric(args[6])/10,.1)
magnitude_changes<-seq(as.numeric(args[3]),as.numeric(args[4]),5)
num_classes<-as.numeric(args[1])

df<-read.csv("data/HealthySBP80.csv",stringsAsFactors = F)
head(df)

df$imputed_3<-FALSE
df$imputed_3[is.na(df$V2)]<-TRUE
table(df$imputed_3)

colnames(df)<-c("id","sbp","date","observed_3","num","max_sbp","imputed_3")
df$imp_sbp<-df$sbp
df$imp_date<-df$date
for(i in seq(1,nrow(df))){
  if(is.na(df$imp_sbp[i])){
    df$imp_sbp[i]<-mean(c(df$imp_sbp[i-1],df$imp_sbp[i+1]),na.rm = TRUE)
    df$imp_date[i]<-round(mean(c(df$imp_date[i-1],df$imp_date[i+1]),na.rm = TRUE))
  }
}
head(df)


df$median_at_t<-NA
for (i in unique(df$num)){
  df$median_at_t[df$num==i]<-(median(df$imp_sbp[df$num==i],na.rm=TRUE))
}
df$median_cen<-df$imp_sbp-df$median_at_t

generate_sim<-function(type_traj,sample_num,noise_mag,magnitude_change){
  sample_cohort<-NULL
  for(i in seq(1,sample_num)){
    temp<-type_traj+sample(seq(-1*(noise_mag*magnitude_change),(noise_mag*magnitude_change),noise_mag/10),1)
    #+sample(seq(-1*(noise_mag*),noise_mag,1),1)
    #+(((type_traj)-mean(type_traj))*sample(seq(-1*noise_mag,noise_mag,noise_mag/100),1))
    sample_cohort<-rbind(sample_cohort,cbind(i,temp,seq(1,length(temp))*3))
  }
  return(sample_cohort)
}

random_seed<-sample(1:100000,1)
set.seed(random_seed)
i<-sample(unique(df$id),1)

original_id<-i
sub_df<-df[df$id==i,]
mean(sub_df$median_cen)
sd(sub_df$median_cen)
date_track<-as.numeric(Sys.Date())
new_root<-paste("data/mag_cohorts/Simulation_Cohort_",original_id,"_",date_track,"/",sep="")
new_root2<-paste("data/mag_cohort_images/Simulation_Cohort_",original_id,"_",date_track,"/",sep="")
if(!dir.exists(new_root)){
  dir.create(new_root)
}
if(!dir.exists(new_root2)){
  dir.create(new_root2)
}

master_file<-paste("data/","simulation_magnitude_only_master.txt",sep="")
if(!file.exists(master_file)){
  file.create(master_file)
}

master_update<-NULL
for(num_class in num_classes){
  for(magnitude_change in magnitude_changes){
   for(sample_num in sample_nums){ 
     for(noise_mag in noise_mags){
     for(i in seq(1, num_class-1)){
       sub_df[,paste("raised_median",i,sep="_")]<-sub_df$median_cen+(magnitude_change*i)
     }

   cohorts_columns<-grep("median",colnames(sub_df))[-1]


   sub_df2<-melt(sub_df,"num",cohorts_columns) 
   p1<-ggplot()+geom_line(data=sub_df2,aes(y=value,x=num*3,color=variable))+xlab("Time (months)")+ylab("SBP")+theme(legend.position = "none",panel.background = 
                                                                                                                   element_rect(fill="white"),axis.line = element_line(color = "black"))
   final_df<-NULL 
   j<-1
 for(i in cohorts_columns){
  a<-cbind(as.data.frame(generate_sim(sub_df[,i],sample_num,noise_mag,magnitude_change)),paste("Class",j))
  colnames(a)<-c("ID","SBP","Time","Class")
  j<-j+1
  final_df<-rbind(final_df,a)
   }
   final_df<-cbind(final_df,original_id)
   p2<-ggplot()+geom_line(data=final_df,aes(x=Time,y=SBP,color=as.factor(Class),group=interaction(ID,Class)))+theme(legend.position = "none",panel.background = 
                                                                              element_rect(fill="white"),axis.line = element_line(color = "black"))
   write.csv(final_df,paste(paste(new_root,"Simulated_Cohort",num_class,sample_num,magnitude_change,noise_mag,sep="_"),".csv",sep=""),row.names = F)
   jpeg(paste(paste(new_root2,"Simulated_Cohort",num_class,sample_num,magnitude_change,noise_mag,sep="_"),".jpg",sep=""),res = 1000,width = 5,height = 5,units = "in")
   gridExtra::grid.arrange(p1,p2)
   graphics.off()
   master_update<-rbind(master_update,cbind(random_seed,original_id,num_class,sample_num,magnitude_change,noise_mag,date_track))
       }
     }
    }
}
write.table(master_update,master_file,sep="\t",append = TRUE,row.names = FALSE,col.names = FALSE)

