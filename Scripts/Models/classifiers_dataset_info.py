# method_dict = {'MLP': {'params': '#_dense_layers | #_hidden_nodes | L1_dropout | L2_dropout | L3_dropuot | L1_L3_activation | L4_activation | batch_size | #_epochs | loss_function | optimizer | lr | lr_factor | min_lr | lr_reduce | stop_metric',
#                       'values': '4 | 500 | 0.1 | 0.2 | 0.3 | Relu | Softmax | 50 | 500 | categorical_crossentropy | Adadelta | 1.0 | 0.92 | 0.1 | no_loss_change_for_3_epochs | no_val_loss_change_for_15_epochs'},
#               'TSF_MLP': {'params': '#_dense_layers | #_hidden_nodes | L1_dropout | L2_dropout | L3_dropuot | L1_L3_activation | L4_activation | batch_size | #_epochs | loss_function | optimizer | lr | lr_factor | min_lr | lr_reduce | stop_metric | min_subseries_length | #_intervals',
#                           'values': '4 | 500 | 0.1 | 0.2 | 0.3 | Relu | Softmax | 50 | 500 | categorical_crossentropy | Adadelta | 1.0 | 0.92 | 0.1 | no_loss_change_for_3_epochs | no_val_loss_change_for_15_epochs | series_length/5 | square_root(series_length)'},
#               'GAF_CNN':{'params': '#_conv_layers | #_filters | conv_kernel_size | conv_stride | conv_padding | conv_activation | #_pooling_layer | pooling_type | pooling_size | #_global_pooling | global_pooling_type | #_dense_layer | dense_activation | batch_size | #_epochs | loss_function | optimizer | lr | lr_factor | min_lr | lr_reduce | stop_metric',
#                          'values': '5 | 32_32_32_32_64 | (3,3) | (1,1) | yes | Relu | 2 | max | (2,2) | 1 | average | 1 | softmax | 50 | 500 | categorical_crossentropy | Adam | 0.001 | 0.92 | 0.0001 | no_loss_change_for_3_epochs | no_val_loss_change_for_15_epochs'},
#               'FCNN':{'params': '#_conv_layers | #_filters | conv_kernel_size | conv_stride | conv_padding | conv_activation | batch_normalization | #_global_pooling | global_pooling_type | #_dense_layer | dense_activation | batch_size | #_epochs | loss_function | optimizer | lr | lr_factor | min_lr | lr_reduce | stop_metric',
#                          'values': '3 | 128_256_128 | 8_5_3 | 1 | yes | Relu | yes | 1 | average | 1 | softmax | 50 | 500 | categorical_crossentropy | Adam | 0.001 | 0.92 | 0.0001 | no_loss_change_for_3_epochs | no_val_loss_change_for_15_epochs'},
#               'ResNet':{'params': '#_conv_layers | #_filters | conv_kernel_size | conv_stride | conv_padding | conv_activation | batch_normalization | #_global_pooling | global_pooling_type | #_dense_layer | dense_activation | batch_size | #_epochs | loss_function | optimizer | lr | lr_factor | min_lr | lr_reduce | stop_metric',
#                          'values': '10 | 64 | 8_5_3_1 | 1 | yes | Relu | yes | 1 | average | 1 | softmax | 50 | 500 | categorical_crossentropy | Adam | 0.001 | 0.92 | 0.0001 | no_loss_change_for_3_epochs | no_val_loss_change_for_15_epochs'},
#               'RNN':{'params': '#_LSTM_layers | #_units | dropout_rate | recurrent_dropout_rate | #_dense_layer | dense_activation | batch_size | #_epochs | loss_function | optimizer | lr | lr_factor | min_lr | lr_reduce | stop_metric',
#                          'values': '3 | 16_32_64 | 0.1_0.1_0.1 | 0.2_0.3_0.5 | 1 | softmax | 50 | 500 | categorical_crossentropy | Adam | 0.001 | 0.92 | 0.0001 | no_loss_change_for_3_epochs | no_val_loss_change_for_15_epochs'},
#               'CNN_RNN':{'params': '#_conv_layers | #_filters | conv_kernel_size | conv_stride | conv_padding | conv_activation | #_pooling_layer | pooling_type | pooling_size | #_LSTM_layers | #_units | dropout_rate | recurrent_dropout_rate | #_dense_layer | dense_activation | batch_size | #_epochs | loss_function | optimizer | lr | lr_factor | min_lr | lr_reduce | stop_metric',
#                          'values': '2 | 32_64 | 5_3 | 1 | yes | Relu | 1 | max | 2 | 1 | 64 | 0.1 | 0.5 | 1 | softmax | 50 | 500 | categorical_crossentropy | Adam | 0.001 | 0.92 | 0.0001 | no_loss_change_for_3_epochs | no_val_loss_change_for_15_epochs'}
#               }

dataset_dict = {
    # 'mag_cohorts':{'nb_classes':5, 'n_indv_per_class':3000, 'n_channels':1, 'path':'/home/khademg/simulationStudy/data/mag_cohorts/', 'class_order': 'Class
    # 1 | Class 2 | Class 3 | Class 4 | Class 5', 'class_labels':['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'] }, 'shape_cohorts':{'nb_classes':2,
    # 'n_indv_per_class':3000, 'n_channels':1, 'path':'/home/khademg/simulationStudy/data/shape_cohorts/', 'class_order': 'original | derived',
    # 'class_labels':['original', 'derived']}, 'mag_cohorts_NEW':{'nb_classes':2, 'n_indv_per_class':3000, 'n_channels':1,
    # 'path':'/home/khademg/simulationStudy/dataMarch2020/data_subset_test/mag_cohorts/', 'class_order': 'Original | Derived', 'class_labels':['Original',
    # 'Derived']}, 'shape_cohorts_NEW':{'nb_classes':2, 'n_indv_per_class':3000, 'n_channels':1,
    # 'path':'/home/khademg/simulationStudy/dataMarch2020/data_subset_test/shape_cohorts/', 'class_order': 'original | derived', 'class_labels':['original',
    # 'derived']}, 'mag_cohorts_Sep':{'nb_classes':2, 'n_indv_per_class':3000, 'n_channels':1,
    # 'path':'/mnt/isilon/data/RotroffDLab/projects/simulationStudy/dataSep2020/mag_cohorts/', 'class_order': 'Original | Derived', 'class_labels':[
    # 'Original', 'Derived']}, 'shape_cohorts_Sep':{'nb_classes':2, 'n_indv_per_class':3000, 'n_channels':1,
    # 'path':'/mnt/isilon/data/RotroffDLab/projects/simulationStudy/dataSep2020/shape_cohorts/', 'class_order': 'original | derived', 'class_labels':[
    # 'original', 'derived']}, 'mag_cohorts_CO': {'nb_classes': 2, 'n_indv_per_class': 3000, 'n_channels': 1,
    # 'path': '/mnt/isilon/data/RotroffDLab/projects/childhood_obesity/dataSimulation/mag_cohorts/', 'class_order': 'Original | Derived', 'class_labels': [
    # 'Original', 'Derived']}, 'shape_cohorts_CO': {'nb_classes': 2, 'n_indv_per_class': 3000, 'n_channels': 1,
    # 'path': '/mnt/isilon/data/RotroffDLab/projects/childhood_obesity/dataSimulation/shape_cohorts/', 'class_order': 'original | derived', 'class_labels': [
    # 'original', 'derived']}, 'mag_cohorts_2021': {'nb_classes': 2, 'n_indv_per_class': 3000, 'n_channels': 1,
    # 'path': '/mnt/isilon/data/RotroffDLab/projects/simulationStudy/dataSimulation2021/mag_cohorts/', 'class_order': 'Original | Derived', 'class_labels': [
    # 'Original', 'Derived']}, 'shape_cohorts_2021': {'nb_classes': 2, 'n_indv_per_class': 3000, 'n_channels': 1,
    # 'path': '/mnt/isilon/data/RotroffDLab/projects/simulationStudy/dataSimulation2021/shape_cohorts/', 'class_order': 'original | derived', 'class_labels':
    # ['original', 'derived']},
    #
    #                 'mag_cohorts_2021_v2': {'nb_classes': 2, 'n_indv_per_class': 3000, 'n_channels': 1,
    #                                     'path': '/mnt/isilon/data/RotroffDLab/projects/simulationStudy/dataSimulation2021v2/mag_cohorts/',
    #                                     'class_order': 'Original | Derived', 'class_labels': ['Original', 'Derived']},
    #                 'shape_cohorts_2021_v2': {'nb_classes': 2, 'n_indv_per_class': 3000, 'n_channels': 1,
    #                                       'path': '/mnt/isilon/data/RotroffDLab/projects/simulationStudy/dataSimulation2021v2/shape_cohorts/',
    #                                       'class_order': 'original | derived', 'class_labels': ['original', 'derived']},
    #
    #                 'mag_cohorts_2021_v3': {'nb_classes': 2, 'n_indv_per_class': 3000, 'n_channels': 1,
    #                                     'path': '/mnt/isilon/data/RotroffDLab/projects/simulationStudy/dataSimulation2021v3/mag_cohorts/',
    #                                     'class_order': 'Original | Derived', 'class_labels': ['Original', 'Derived']},
    #                 'shape_cohorts_2021_v3': {'nb_classes': 2, 'n_indv_per_class': 3000, 'n_channels': 1,
    #                                       'path': '/mnt/isilon/data/RotroffDLab/projects/simulationStudy/dataSimulation2021v3/shape_cohorts/',
    #                                       'class_order': 'original | derived', 'class_labels': ['original', 'derived']},

    '2022v1': {'nb_classes': 2, 'n_channels': 1,
               'path': '../../Dataset/dataSimulation2022v1/',
               'class_order': 'Original | Derived', 'class_labels': ['Original', 'Derived']},

    # 'local_mag_cohorts_2021': {'nb_classes': 2, 'n_indv_per_class': 3000, 'n_channels': 1,
    #                     'path': '/Users/javidih/OneDrive - Cleveland State University/Cleveland Clinic/dataSimulation2021v3/mag_cohorts/',
    #                     'class_order': 'Original | Derived', 'class_labels': ['Original', 'Derived']},
    # 'local_shape_cohorts_2021': {'nb_classes': 2, 'n_indv_per_class': 3000, 'n_channels': 1,
    #                       'path': '/Users/javidih/OneDrive - Cleveland State University/Cleveland Clinic/dataSimulation2021v3/shape_cohorts/',
    #                       'class_order': 'original | derived', 'class_labels': ['original', 'derived']},
    'local_2022v1': {'nb_classes': 2, 'n_channels': 1,
                     'path': '/Users/javidih/OneDrive - Cleveland State University/Cleveland Clinic/dataSimulation2022v1/',
                     'class_order': 'Original | Derived', 'class_labels': ['Original', 'Derived']}

}
