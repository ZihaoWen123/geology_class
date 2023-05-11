# geology_class0
Our programming is in Python, and here is a note on all the code files and data files:
File “Igneous_result” is to show the effect of igneous rocks and ore deposits classification models, such as confusion matrix, five-fold cross-validation results, F1-score and feature importance.
File “Yilgarn_zircon_chem” contains 30 zircon composition databases from 30 different igneous rocks in Yilgarn Craton, Western Australia. We use them to do a case study on the igneous rock classification model.
File “machine_learn” is a Python file containing algorithms of decision tree, random forest and XGBoost.
File “parameter” records all the results of Bayesian optimization of the Random Forest algorithm, including all features and 8 features of the igneous rocks and ore deposits classification model. pkl.file is the trained igneous rocks and ore deposits classification model based on the 8 important features.
Document “Zirocn_Composition_Database” is a zircon trace element database containing 7173 zircon samples from different igneous rocks and 3831 zircon samples from different ore deposits, which we use to train classification models for igneous rocks and ore deposits.
Document “Whole-Rock Geochemistry of 30 Igneous Rocks in the Yilgarn Craton” is the whole-rock geochemistry database that containing 30 igneous rocks from the Yilgarn Craton in Western Australia. It is used to identify the rock types and to compare the results with zircon classification of igneous rocks.
Document “data_process_train” is a Python file to pre-process the code when training the model.
Document “data_process_apply” is a Python file that save the code for pre-processing the data when applying the model.
Document “train_model_apply” is a Python file to save the code for the model application.
Document “infer” is a Python file, which is the code for the case study.
Document “utils” is a Python file, which is the code for drawing.
If you are interested in our code and research, please refer to paper Zi-Hao Wen, Lin Li, Christopher Kirkland, Sheng-Rong Li, Xiao-Jie Sun, Jia-Li Lei, Bo Xu and Zeng-Qian Hou, 2022, A machine learning approach to discrimination of igneous rocks and ore deposits by zircon trace elements, and feel free to cite it.
