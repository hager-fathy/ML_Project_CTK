# -*- coding: utf-8 -*-
"""
Created on Sat May 11 21:08:44 2024

@author: hp
"""
from sklearn.neural_network import MLPClassifier , MLPRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score, f1_score, classification_report
#from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ANN_Frame(ctk.CTkFrame):
    def __init__(self, master, write_func):
        super().__init__(master)
        self.pack(expand=True, fill='both')
        self.show_detail_model = write_func
        self.rowconfigure(index=list(range(11)), weight=1, uniform="a")
        self.columnconfigure(index=(0, 1), weight=1, uniform="a")
        
        
        self.label_column = ctk.CTkOptionMenu(self)
        self.label_column.grid(row=0, column=0, padx=5, pady=5, sticky="ew"  )
        self.label_column.set("Label" )
        
           
        self.sizehidden_ent = ctk.CTkEntry(self , placeholder_text = "Hidden Layer Size")
        self.sizehidden_ent.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.learningrate_option = ctk.CTkOptionMenu(self , values = ['adaptive', 'constant', 'invscaling'])
        self.learningrate_option.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.learningrate_option.set("learning rate")
        
        self.Activat_func_option = ctk.CTkOptionMenu(self , values = ['tanh', 'logistic', 'relu', 'identity'])
        self.Activat_func_option.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.Activat_func_option.set("Activation Functions")
                 
        

        self.split_data_btn = ctk.CTkButton(self , text = "Make Model", command=self.Apply_model)
        self.split_data_btn.grid(row=2, column=0, padx=5, pady=10, sticky="ew" , columnspan =2)



        self.testsize_ent = ctk.CTkEntry(self , placeholder_text = "Test Size")
        self.testsize_ent.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        
        self.random_ent = ctk.CTkEntry(self , placeholder_text = "Random State")
        self.random_ent.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        
        self.split_btn = ctk.CTkButton(self , text = "Split Data" , command =self.split_data)
        self.split_btn.grid(row=4, column=0, padx=5, pady=5, sticky="ew" , columnspan =2 )   
        
        self.k_ent = ctk.CTkEntry(self , placeholder_text = "n_splits" )
        self.k_ent.grid(row=5 , column=1, padx=5, pady=5, sticky="ew")
               
        self.kfoldmethod_btn = ctk.CTkButton(self , text = "k-fold Method" , command =self.k_fold_method)
        self.kfoldmethod_btn.grid(row=5, column=0, padx=5, pady=5, sticky="ew" )
    
        self.smote_btn = ctk.CTkButton(self , text = "Smote" , command = self.apply_smote )
        self.smote_btn.grid(row=6, column=0, padx=5, pady=5, sticky="ew", columnspan =2)

        self.train_btn = ctk.CTkButton(self , text = "Train Data" , command = self.train_model )
        self.train_btn.grid(row=7, column=0, padx=5, pady=5, sticky="ew")
        
        self.test_btn = ctk.CTkButton(self , text = "Test Data"  , command = self.test_model)
        self.test_btn.grid(row=7, column= 1, padx=5, pady=5, sticky="ew")  
        
        self.precision_btn = ctk.CTkButton(self , text = "Classification Report" , command = self.show_report  )
        self.precision_btn.grid(row=8, column= 0, padx=5, pady=5, sticky="ew" , columnspan =2 ) 
        
        self.accurcy_btn = ctk.CTkButton(self , text = "Accurcy" , command = self.show_accuracy)
        self.accurcy_btn.grid(row=9, column= 0, padx=5, pady=5, sticky="ew"  ) 
      
        self.recall_btn = ctk.CTkButton(self , text = "Recall" , command = self.show_recall )
        self.recall_btn.grid(row=9, column= 1, padx=5, pady=5, sticky="ew") 
        
        self.precision_btn = ctk.CTkButton(self , text = "precision" , command = self.show_precision)
        self.precision_btn.grid(row=10, column= 0, padx=5, pady=5, sticky="ew" )   
        
        self.f1_btn = ctk.CTkButton(self , text = "f1_score" , command =self.show_f1measure )
        self.f1_btn.grid(row=10, column= 1, padx=5, pady=5, sticky="ew") 
        
        self.plotconf_btn = ctk.CTkButton(self , text = "Plot Matrix" , command =self.plot_confusion_matrix )
        self.plotconf_btn.grid(row=11, column= 0, padx=5, pady=5, sticky="ew" , columnspan = 2)

    def get_df(self, df):
       self.dataframe = df
       self.label_column.configure(values=self.dataframe.columns)
       
    def Apply_model(self):
        
        self.ann = MLPClassifier(activation= self.Activat_func_option.get() , hidden_layer_sizes = int(self.sizehidden_ent.get()), learning_rate= self.learningrate_option.get())
        self.show_detail_model('Model Created')





    def k_fold_method(self):
    # Get the label column name and prepare feature and target data
         label = self.label_column.get()
         self.x = self.dataframe.drop(columns=[label], axis=1)
         self.y = self.dataframe[label]
    
    # Get the number of folds
         self.k = int(self.k_ent.get())
    
    # Initialize KFold with the specified number of splits
         kfold = KFold(n_splits=self.k, shuffle=True, random_state=42)
         self.acc_list = []
         self.precision_list = []
         self.recall_list = []
         self.f1_list = []

    # Perform K-Fold cross-validation
         for train_index, test_index in kfold.split(self.x):
            self.x_train, self.x_test = self.x.iloc[train_index], self.x.iloc[test_index]
            self.y_train, self.y_test = self.y.iloc[train_index], self.y.iloc[test_index]
        
        # Apply SMOTE to balance the training data
            smote = SMOTE(random_state=42)
            self.x_train_res, self.y_train_res = smote.fit_resample(self.x_train, self.y_train)
        
        # Fit the ANN model on the resampled training data
            self.ann.fit(self.x_train_res, self.y_train_res)
        
        # Predict the test data
            self.y_pred = self.ann.predict(self.x_test)
        
        # Calculate evaluation metrics
            self.accuracy = accuracy_score(self.y_test, self.y_pred)
            self.precision = precision_score(self.y_test, self.y_pred, average='weighted')
            self.recall = recall_score(self.y_test, self.y_pred, average='weighted')
            self.f1_measure = f1_score(self.y_test, self.y_pred, average='weighted')
        
        # Append metrics to respective lists
            self.acc_list.append(self.accuracy)
            self.precision_list.append(self.precision)
            self.recall_list.append(self.recall)
            self.f1_list.append(self.f1_measure)
    
    # Calculate average metrics across all folds
         self.accuracy_k_fold = sum(self.acc_list) / self.k
         self.precision_k_fold = sum(self.precision_list) / self.k
         self.recall_k_fold = sum(self.recall_list) / self.k
         self.f1_k_fold = sum(self.f1_list) / self.k

    # Display model details
         self.show_detail_model('Model done by K-Fold method.')
         self.show_detail_model(f'Accuracy: {self.accuracy_k_fold * 100:.2f}%')
         self.show_detail_model(f'Precision: {self.precision_k_fold * 100:.2f}%')
         self.show_detail_model(f'Recall: {self.recall_k_fold * 100:.2f}%')
         self.show_detail_model(f'F1: {self.f1_k_fold * 100:.2f}%')

#------------------------------------------------------------------------------------------------------------------------------
    def split_data(self):
        label = self.label_column.get()
        self.x = self.dataframe.drop(columns=[label])
        self.y = self.dataframe[label]        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                                                test_size=float(self.testsize_ent.get()),
                                                                                random_state=int(self.random_ent.get()),
                                                                                shuffle=True)
        self.show_detail_model('Split Done by Hold-out Method ')   
    def apply_smote(self):
        sm = SMOTE()
        self.x_train, self.y_train = sm.fit_resample(self.x_train, self.y_train)
        self.show_detail_model('Smote Done Data Become Balanced')

    
    def train_model(self):
        self.ann.fit(self.x_train, self.y_train)
        self.show_detail_model('Model Trained')
        
    def test_model(self):
        self.y_pred = self.ann.predict(self.x_test)
        self.show_detail_model('Model Tested')

        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.precision = precision_score(self.y_test, self.y_pred, average='weighted')
        self.recall = recall_score(self.y_test, self.y_pred, average='weighted')
        self.f1_measure = f1_score(self.y_test, self.y_pred, average='weighted')
        self.con_matrix = confusion_matrix(self.y_test, self.y_pred)
        self.class_report = classification_report(self.y_test, self.y_pred)

        
    def show_accuracy(self):
        self.show_detail_model(f'Accuracy: {self.accuracy * 100:.2f}%')

    def show_recall(self):
        self.show_detail_model(f'Recall: {self.recall * 100:.2f}%')

    def show_precision(self):
       self.show_detail_model(f'Precision: {self.precision * 100:.2f}%')

    def show_f1measure(self):
        self.show_detail_model(f'F1: {self.f1_measure * 100:.2f}%')
 
    def show_report(self):
       self.show_detail_model(f'Classification Report:\n{self.class_report}')
    

    def plot_confusion_matrix(self):
        # Create a new Tkinter window
        confusion_win = tk.Toplevel()
        confusion_win.title("Confusion Matrix")

        # Compute confusion matrix
        con_matr = confusion_matrix(self.y_test, self.y_pred)

        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=(5, 4))

        # Plot the confusion matrix using seaborn
        sns.heatmap(con_matr, annot=True, cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"], ax=ax)
        ax.set_title('Confusion Matrix for Decision Tree:', color='k', fontsize=14)
        ax.set_xlabel('Predicted Labels', color='k')
        ax.set_ylabel('Actual', color='k')

        # Embed the matplotlib figure in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=confusion_win)
        canvas.draw()
        canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=True)
        confusion_win.mainloop()

        self.show_detail_model("Plot Confusion Matrix done")