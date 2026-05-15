# -*- coding: utf-8 -*-
"""
Created on Sat May 11 21:08:43 2024

@author: hp
"""
from sklearn.svm import SVC , SVR

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score, f1_score, classification_report
#from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import customtkinter as ctk
import tkinter as tk 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.model_selection import GridSearchCV

class SVM_Frame(ctk.CTkFrame):
    def __init__(self, master, write_func):
        super().__init__(master)
        self.pack(expand=True, fill='both')
        self.show_detail_model = write_func
        self.rowconfigure(index=list(range(11)), weight=1, uniform="a")
        self.columnconfigure(index=(0, 1), weight=1, uniform="a")
        

        self.label_column = ctk.CTkOptionMenu(self)
        self.label_column.grid(row=0, column=0, padx=5, pady=5, sticky="ew" , columnspan= 2)
        self.label_column.set("Label")

                
        self.k_ent = ctk.CTkEntry(self , placeholder_text = "K_splits" )
        self.k_ent.grid(row=1 , column=1, padx=5, pady=5, sticky="ew")
        
        self.testsize_ent = ctk.CTkEntry(self , placeholder_text = "Test Size")
        self.testsize_ent.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        self.random_ent = ctk.CTkEntry(self , placeholder_text = "Random State")
        self.random_ent.grid(row=2, column=1, padx=5, pady=5, sticky="ew")   
        
        self.kernal_meth = ctk.CTkOptionMenu(self , values = ['sigmoid', 'linear', 'poly', 'precomputed', 'rbf'])
        self.kernal_meth.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.kernal_meth.set("Kernal")
        
        self.Regularization_c_ent = ctk.CTkEntry(self , placeholder_text = "(C)")
        self.Regularization_c_ent.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
      
        
        self.split_data_btn = ctk.CTkButton(self , text = "Make Model", command=self.Apply_model)
        self.split_data_btn.grid(row=4, column=0, padx=5, pady=10, sticky="ew" , columnspan =2)

        self.rfe_btn = ctk.CTkButton(self , text = "RFE"  , command= self.apply_rfe)
        self.rfe_btn.grid(row=5 , column=0, padx=5, pady=5, sticky="ew")
        
        self.rfe_ent = ctk.CTkEntry(self , placeholder_text = "Num columns" )
        self.rfe_ent.grid(row=5 , column=1, padx=5, pady=5, sticky="ew")

        
        self.split_btn = ctk.CTkButton(self , text = "Split Data" , command =self.split_data)
        self.split_btn.grid(row=6, column=0, padx=5, pady=5, sticky="ew" , columnspan =2 )
        
        self.smote_btn = ctk.CTkButton(self , text = "Smote" , command = self.apply_smote )
        self.smote_btn.grid(row=7, column=0, padx=5, pady=5, sticky="ew", columnspan =2)

        self.train_btn = ctk.CTkButton(self , text = "Train Data" , command = self.train_model )
        self.train_btn.grid(row=8, column=0, padx=5, pady=5, sticky="ew")
        
        self.test_btn = ctk.CTkButton(self , text = "Test Data"  , command = self.test_model)
        self.test_btn.grid(row=8, column= 1, padx=5, pady=5, sticky="ew")  
        
        self.precision_btn = ctk.CTkButton(self , text = "Classification Report" , command = self.show_report  )
        self.precision_btn.grid(row=9, column= 0, padx=5, pady=5, sticky="ew" , columnspan =2 ) 
        
        self.accurcy_btn = ctk.CTkButton(self , text = "Accurcy" , command = self.show_accuracy)
        self.accurcy_btn.grid(row=10, column= 0, padx=5, pady=5, sticky="ew"  ) 
      
        self.recall_btn = ctk.CTkButton(self , text = "Recall" , command = self.show_recall )
        self.recall_btn.grid(row=10, column= 1, padx=5, pady=5, sticky="ew") 
        
        self.precision_btn = ctk.CTkButton(self , text = "precision" , command = self.show_precision)
        self.precision_btn.grid(row=11, column= 0, padx=5, pady=5, sticky="ew" )   
        
        self.f1_btn = ctk.CTkButton(self , text = "f1_score" , command =self.show_f1measure )
        self.f1_btn.grid(row=11, column= 1, padx=5, pady=5, sticky="ew") 
        
        self.plotconf_btn = ctk.CTkButton(self , text = "Plot Matrix" , command =self.plot_confusion_matrix )
        self.plotconf_btn.grid(row=12, column= 0, padx=5, pady=5, sticky="ew" , columnspan = 2)
        
        self.tunning_btn = ctk.CTkButton(self , text = "Tunning" , command = self.tunning_algorithm)
        self.tunning_btn.grid(row=13, column= 0, padx=5, pady=5, sticky="ew" , columnspan=2)
        
    def get_df(self, df):
       self.dataframe = df
       self.label_column.configure(values=self.dataframe.columns)
       
    def Apply_model(self):

        self.svm  =  SVC(kernel= self.kernal_meth.get() ,C =int(self.Regularization_c_ent.get()) )
        self.show_detail_model('Model Created')


    def apply_rfe(self):
        label = self.label_column.get()
        self.x = self.dataframe.drop(columns=[label])
        self.y = self.dataframe[label]
        self.model=self.svm
        n_feature =  int(self.rfe_ent.get())
        rfe_model = RFE(self.model,n_features_to_select=n_feature)
        rfe_model.fit(self.x, self.y)

        # Get the support and ranking from the RFE model
        self.filter = rfe_model.support_
        self.ranking = rfe_model.ranking_

        # Update the dataframe to include only the selected features
        self.x = self.x.loc[:, self.filter]
        
        # Combine the selected features with the target column
        self.dataframe = pd.concat([self.x, self.y], axis=1)
        self.show_detail_model('RFE DoNe ...')
        self.show_detail_model(f'{self.filter}')
        self.show_detail_model(f'{self.ranking}')
    
    
    def split_data(self) :
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
        self.svm.fit(self.x_train, self.y_train)
        self.show_detail_model('Model Trained')
        
    def test_model(self):
        self.y_pred = self.svm.predict(self.x_test)
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
    def tunning_algorithm (self):
       self.params_svm = {"C" : np.arange(1, 20)}
       self.svm_gs = GridSearchCV(self.svm, self.params_svm, cv=5)
       self.svm_gs.fit(self.x_train, self.y_train)
       self.svm_best = self.svm_gs.best_estimator_
       self.show_detail_model(f'Best Paramter : {self.svm_gs.best_params_}')