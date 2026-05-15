from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
import tkinter as tk
import numpy as np

class KMeansFrame(ctk.CTkFrame):
    def __init__(self, master, write_func):
        super().__init__(master)
        self.pack(expand=True, fill='both')
        self.show_detail_model = write_func
        self.rowconfigure(index=list(range(11)), weight=1, uniform="a")
        self.columnconfigure(index=(0, 1), weight=1, uniform="a")
        
        self.n_clusters = ctk.CTkEntry(self, placeholder_text="N-clusters")
        self.n_clusters.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.plot_button = ctk.CTkButton(self, text="Plot Clusters", command=self.plot_clusters)
        self.plot_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.first_column = ctk.CTkLabel(self, text="")
        self.first_column.grid(row=1, column=0, padx=5, pady=5, sticky="ew")        
 
        self.second_column = ctk.CTkLabel(self, text="")
        self.second_column.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        


    def get_df(self, df):
        self.dataframe = df
        self.first_column.configure(values=self.dataframe.columns)
        self.second_column.configure(values=self.dataframe.columns)        

    def plot_clusters(self):
        n_clusters = int(self.n_clusters.get())

        kmeans = KMeans(n_clusters=n_clusters)
        x = self.dataframe # Assuming these columns exist
        kmeans.fit(x)
        

        identified_clusters = kmeans.predict(x)
        data_with_clusters = self.dataframe.copy()
        data_with_clusters['Cluster'] = identified_clusters
        ax = self.figure.add_subplot(111)
        scatter = ax.scatter(data_with_clusters[self.first_column], data_with_clusters[self.second_column], c=data_with_clusters['Cluster'], cmap='viridis')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_title('K-Means Clustering')
        self.canvas.draw()
        
