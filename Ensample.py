# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:22:00 2024

@author: hp
"""


import numpy as np 
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import customtkinter as ctk
import tkinter as tk
from sklearn.model_selection import KFold
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.svm import SVC , SVR
from sklearn.tree import DecisionTreeClassifier ,  plot_tree , DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier , KNeighborsRegressor
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score, f1_score, classification_report
from sklearn.ensemble import VotingClassifier



class ES_Frame(ctk.CTkFrame):
    def __init__(self, master, write_func):
        super().__init__(master)
        self.pack(expand=True, fill='both')
        self.show_detail_model = write_func
        self.rowconfigure(index=list(range(11)), weight=1, uniform="a")
        self.columnconfigure(index=(0, 1), weight=1, uniform="a")
        #------------------------------------------------------------------------------------------------------------
        self.kernal_meth = ctk.CTkOptionMenu(self , values = ['sigmoid', 'linear', 'poly', 'precomputed', 'rbf'])
        self.kernal_meth.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.kernal_meth.set("Kernal")
        
        self.Regularization_c_ent = ctk.CTkEntry(self , placeholder_text = "(C)")
        self.Regularization_c_ent.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.model_btn = ctk.CTkButton(self , text = "SVM Model", command=self.Apply_svm)
        self.model_btn.grid(row=1, column=0, padx=5, pady=10, sticky="ew" , columnspan =2)        
        #------------------------------------------------------------------------------------------------------------
        self.Matrix_meth = ctk.CTkOptionMenu(self , values = ["euclidean" , "manhattan" , "minkowski"])
        self.Matrix_meth.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.Matrix_meth.set("Distance Metric")
        
        self.n_neighbors_ent = ctk.CTkEntry(self , placeholder_text = "N_Neighbors")
        self.n_neighbors_ent.grid(row=2, column=1, padx=5, pady=5, sticky="ew")   
        
        self.model_btn = ctk.CTkButton(self , text = "KNN Model", command=self.Apply_knn)
        self.model_btn.grid(row=3, column=0, padx=5, pady=10, sticky="ew" , columnspan =2)          
        #-------------------------------------------------------------------------------------------------------------

        self.max_depth_ent = ctk.CTkEntry(self , placeholder_text = "Max Depth")
        self.max_depth_ent.grid(row=4, column=0, padx=5, pady=10, sticky="ew" , columnspan=2)
        
        self.tree_criterion = ctk.CTkOptionMenu(self , values = ["gini", "entropy", "log_loss"])
        self.tree_criterion.grid(row=5, column=0, padx=5, pady=10, sticky="ew")
        self.tree_criterion.set("Criterion")
        
        self.tree_splitter = ctk.CTkOptionMenu(self , values = ["best", "random"])
        self.tree_splitter.grid(row=5, column=1, padx=5, pady=10, sticky="ew")
        self.tree_splitter.set("Spliter")        
        
        self.model_btn = ctk.CTkButton(self , text = "DT Model", command=self.Apply_dt)
        self.model_btn.grid(row=6, column=0, padx=5, pady=10, sticky="ew" , columnspan =2)   
        
        self.label_column = ctk.CTkOptionMenu(self)
        self.label_column.grid(row=7, column=0, padx=5, pady=10, sticky="ew")
        self.label_column.set("Label")

        
        self.model_btn = ctk.CTkButton(self , text = "Split Data", command=self.split_data)
        self.model_btn.grid(row=7, column=1, padx=5, pady=10, sticky="ew" )           
        #-------------------------------------------------------------------------------------------------------------
        self.model_btn = ctk.CTkButton(self , text = "Ensample Model", command=self.ensamle_algorithm)
        self.model_btn.grid(row=8, column=0, padx=5, pady=10, sticky="ew" , columnspan =2)         
        #--------------------------------------------------------------------------------------------------------------
        self.report_btn = ctk.CTkButton(self , text = "Classification Report" , command = self.show_report)
        self.report_btn.grid(row=9, column= 0, padx=5, pady=10, sticky="ew" , columnspan =2 ) 
        
        self.accurcy_btn = ctk.CTkButton(self , text = "Accurcy" , command = self.show_accuracy)
        self.accurcy_btn.grid(row=10, column= 0, padx=5, pady=10, sticky="ew"  ) 
      
        self.recall_btn = ctk.CTkButton(self , text = "Recall" , command = self.show_recall)
        self.recall_btn.grid(row=10, column= 1, padx=5, pady=10, sticky="ew") 
        
        self.precision_btn = ctk.CTkButton(self , text = "Precision" , command = self.show_precision)
        self.precision_btn.grid(row=11, column= 0, padx=5, pady=10, sticky="ew" )   
        
        self.f1_btn = ctk.CTkButton(self , text = "F1" , command =self.show_f1measure )
        self.f1_btn.grid(row=11, column= 1, padx=5, pady=10, sticky="ew")         
        #------------------------------------------------------------------------------------------------------------------------
    def get_df(self, df):
       self.dataframe = df
       self.label_column.configure(values=self.dataframe.columns)
    def Apply_svm(self):
        self.svm  = SVC(kernel= self.kernal_meth.get() ,C =int(self.Regularization_c_ent.get()) )
        self.show_detail_model('SVM Model Done....')
    def Apply_knn(self):
        self.knn  = KNeighborsClassifier(n_neighbors = int(self.n_neighbors_ent.get()) , metric = self.Matrix_meth.get() )
        self.show_detail_model('KNN Model Done....')
    def Apply_dt(self):
        self.dt = DecisionTreeClassifier(criterion=self.tree_criterion.get(),
                                           max_depth=int(self.max_depth_ent.get()),
                                           splitter=self.tree_splitter.get())
        self.show_detail_model('Desicion Tree model  Done....')
    def split_data(self):
        label = self.label_column.get()
        self.x = self.dataframe.drop(columns=[label])
        self.y = self.dataframe[label]
            
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                    test_size=0.3,random_state=42,shuffle=True)
        self.show_detail_model('Data Split ....')
        
            
    def ensamle_algorithm (self) :
        self.estimators=[('DT',self.dt), ('KNN',self.knn), ('SVM',  self.svm ) ]
        self.ensemble = VotingClassifier(self.estimators, voting='hard')
        self.ensemble.fit(self.x_train, self.y_train)
        self.show_detail_model('Ensamble Done....')
    
        self.y_pred = self.ensemble.predict(self.x_test)
        self.show_detail_model('Model Tested')

        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.precision = precision_score(self.y_test, self.y_pred, average='weighted')
        self.recall = recall_score(self.y_test, self.y_pred, average='weighted')
        self.f1_measure = f1_score(self.y_test, self.y_pred, average='weighted')
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