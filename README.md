Our programming is in Python, and here is a note on all the code files and data files:

File “Deposit_case_test” is zircon composition data file for the deposit case study.
File “Igneous_result” and “Deposit_result” are to show the effect of igneous rocks and ore deposits classification models, such as confusion matrix, five-fold cross-validation results, F1-score and feature importance.
File “Yilgarn_zircon_chem” contains 30 zircon composition databases from 30 different igneous rocks in Yilgarn Craton, Western Australia. We use them to do a case study on the igneous rock classification model.
File “machine_learn” is a Python file containing algorithms of decision tree, random forest and XGBoost.
File “parameter” is a Python file that records all the parameters of the optimal model (random forest).
File “Zirocn_Composition_Database” is a zircon trace element database containing 7173 zircon samples from different igneous rocks and 3831 zircon samples from different ore deposits, which we use to train classification models for igneous rocks and ore deposits.
File “Zircon_Composition_Database_10_elements” is the zircon database that retains only the 10 most important features in the classification model for igneous rocks and ore deposits. We refined and optimized the models with these 10 features.
File “data_process” is a Python file to pre-process the data before training the model.
File “data_process2” is a Python file to pre-process the data before case analysis.
File “infer” is a Python file, which is the code for the case study.
File “utils” is a Python file, which is the code for drawing.

If you are interested in our code and research, please refer to paper：
Zi-Hao Wen, Lin Li, Christopher Kirkland, Sheng-Rong Li, Xiao-Jie Sun and Jia-Li Lei, 2022, A machine learning approach to discrimination of igneous rocks and ore deposits by zircon trace elements, and feel free to cite it.
