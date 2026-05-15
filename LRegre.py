# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:28:43 2024

@author: hp
"""

 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
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

class LR_Frame(ctk.CTkFrame):
    def __init__(self, master, write_func):
        super().__init__(master)
        self.pack(expand=True, fill='both')
        self.show_detail_model = write_func
        self.rowconfigure(index=list(range(11)), weight=1, uniform="a")
        self.columnconfigure(index=(0, 1), weight=1, uniform="a")
        

        self.label_column = ctk.CTkOptionMenu(self)
        self.label_column.grid(row=0, column=0, padx=5, pady=5, sticky="ew" , columnspan =2   )
        self.label_column.set("Label")

        self.split_meth = ctk.CTkOptionMenu(self , values = ["Hold-out" , "k_fold"])
        self.split_meth.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.split_meth.set("Split Method")
        
        self.num_kfold= ctk.CTkEntry(self , placeholder_text = "K-Splits")
        self.num_kfold.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        self.testsize_ent = ctk.CTkEntry(self , placeholder_text = "Test Size")
        self.testsize_ent.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        self.random_ent = ctk.CTkEntry(self , placeholder_text = "Random State")
        self.random_ent.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        self.split_data_btn = ctk.CTkButton(self , text = "Make Model", command=self.Apply_model)
        self.split_data_btn.grid(row=3, column=0, padx=5, pady=10, sticky="ew" , columnspan =2)
        
        self.rfe_model_btn = ctk.CTkButton(self , text = "RFE"  , command=self.apply_rfe)
        self.rfe_model_btn.grid(row=4 , column=0, padx=5, pady=10, sticky="ew")
        
        self.rfe_model_entry = ctk.CTkEntry(self , placeholder_text = "RFE `N`" )
        self.rfe_model_entry.grid(row=4 , column=1, padx=5, pady=10, sticky="ew")
        
        self.split_data_btn = ctk.CTkButton(self , text = "Split Data", command=self.split_data )
        self.split_data_btn.grid(row=5, column=0, padx=5, pady=10, sticky="ew" , columnspan =2)
        
        self.smote_btn = ctk.CTkButton(self , text = "Apply SMOTE" , command = self.apply_smote)
        self.smote_btn.grid(row=6, column=0, padx=5, pady=10, sticky="ew", columnspan =2)        
        
        self.train_btn = ctk.CTkButton(self , text = "Train Model" , command = self.train_model)
        self.train_btn.grid(row=7, column=0, padx=5, pady=10, sticky="ew")

        self.test_btn = ctk.CTkButton(self , text = "Test Model"  , command = self.test_model)
        self.test_btn.grid(row=7, column= 1, padx=5, pady=10, sticky="ew")  
        
        self.report_btn = ctk.CTkButton(self , text = "MSE" , command = self.show_mean)
        self.report_btn.grid(row=8, column= 0, padx=5, pady=10, sticky="ew"  ) 
        
        self.accurcy_btn = ctk.CTkButton(self , text = "R-MSE" , command = self.show_rootmean)
        self.accurcy_btn.grid(row=8, column= 1, padx=5, pady=10, sticky="ew"  ) 
      
        self.recall_btn = ctk.CTkButton(self , text = "intercept" , command = self.show_intercept)
        self.recall_btn.grid(row=9, column= 0, padx=5, pady=10, sticky="ew") 
        
        self.precision_btn = ctk.CTkButton(self , text = "Coefficients" , command = self.show_Coefficients)
        self.precision_btn.grid(row=9, column= 1, padx=5, pady=10, sticky="ew" )   
        
        
        self.plot_conf_btn = ctk.CTkButton(self , text = "Plot" , command =self.ploting)
        self.plot_conf_btn.grid(row=10, column= 0, padx=5, pady=10, sticky="ew" , columnspan = 2)
        


    def get_df(self, df):
       self.dataframe = df
       self.label_column.configure(values=self.dataframe.columns)
       
    def Apply_model(self):

        self.lr  =   LinearRegression()
        self.show_detail_model('Model Created...')
               
    def apply_rfe(self):
        rfe_model = RFE(self.lf, int(self.rfe_model_entry.get()))
        rfe_model.fit(self.x, self.y)
        self.filter = rfe_model.support_
        self.ranking = rfe_model.ranking_
        self.x = self.x[self.x.columns[self.filter]]
        self.dataframe[self.label_col] = self.y
        self.dataframe = pd.concat([self.x, self.dataframe[self.label_col]], axis=1)
        self.show_detail_model('RFA DONE....')
    
    def split_data(self) :
        label = self.label_column.get()
        self.x = self.dataframe.drop(columns=[label])
        self.y = self.dataframe[label]
        if self.split_meth .get() == "k_fold":
            kfold = KFold(n_splits=int(self.num_kfold.get()), random_state=None, shuffle=True)
            self.x_train, self.x_test, self.y_train, self.y_test = [], [], [], []
            for train_index, test_index in kfold.split(self.x):
                self.x_train.append(self.x.iloc[train_index, :])
                self.x_test.append(self.x.iloc[test_index, :])
                self.y_train.append(self.y.iloc[train_index])
                self.y_test.append(self.y.iloc[test_index])
            self.x_train = pd.concat(self.x_train)
            self.x_test = pd.concat(self.x_test)
            self.y_train = pd.concat(self.y_train)
            self.y_test = pd.concat(self.y_test)
            self.show_detail_model('Split Done by k_fold Method')
        else :
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                                                test_size=float(self.testsize_ent.get()),
                                                                                random_state=int(self.random_ent.get()),shuffle=True)
            self.show_detail_model('Split Done by Hold-out Method ')                                                                     

      
    def apply_smote(self):
        sm = SMOTE()
        self.x_train, self.y_train = sm.fit_resample(self.x_train, self.y_train)
   
    def train_model(self):
        self.lr.fit(self.x_train, self.y_train)
        self.show_detail_model('Model Trained')
        
    def test_model(self):
        self.y_pred = self.lr.predict(self.x_test)
        self.show_detail_model('Model Tested')
        #MSE
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        #rmse 
        self.rmse = np.sqrt(self.mse)
        self.intercept  =  self.lr.intercept_
        self.coefficients = self.lr.coef_
    def show_mean(self):
        self.show_detail_model(f'Meaan Squared Error : {str(self.mse)}')
    def show_rootmean(self):
        self.show_detail_model(f'Root Meaan Squared Error : {str(self.rmse)}')

    def show_intercept(self):
        self.show_detail_model(f'Intercept: {str(self.intercept)}')

    def show_Coefficients(self):
        self.show_detail_model(f'Coefficients : {str(self.coefficients)}')


    def  ploting (self):
        # Create a new Tkinter window
        window = tk.Tk()
        window.title("Actual vs Predicted")

        # Create a figure and a subplot
        fig = plt.Figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        # Plot the data
        ax.scatter(self.y_test, self.y_pred, color='blue', label='Predicted')
        ax.plot(self.y_test, self.y_test, color='red', label='Actual')
        ax.set_title('Actual vs Predicted')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.legend()

        # Create a canvas to embed the figure
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add a toolbar for navigation
        toolbar = tk.Frame(window)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)



        # Start the Tkinter main loop
        window.mainloop()
#Plotting


