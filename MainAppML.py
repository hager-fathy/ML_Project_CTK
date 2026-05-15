# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:52:00 2024

@author: hp
"""
"""import all library"""

"""import files """
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, ttk
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier

from Function_Preprossing import *
from DT_Class import *
from SVM_Class import *
from KNN_Class import *
from ANN_Class import *
from LRegre import *
from Ensample import *
from cluster_class import *


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        # """function of grid to Gui"""
        self.grid()

        """call PreProcessingFrame classs"""
        self.left_frame = PreProcessingFrame(
            self,
            func_impute=self.func_impute,
            func_encode=self.func_encode,
            func_drop=self.func_drop,
            func_scale=self.func_scale,
            func_PCA=self.func_PCA,
        )
        """call ModelsFrame class """
        self.right_frame = ModelsFrame(self, write_func=self.show_detail_model)
        """show browser button"""
        self.browse_btn = ctk.CTkButton(self, text="Browse", command=self.browse_data)
        self.browse_btn.grid(column=0, row=2, padx=5, pady=5, sticky="ew")
        """show text box which show info of model"""
        self.left_textbox = ctk.CTkTextbox(self)
        self.left_textbox.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")
        """destory button"""
        self.destroy_btn = ctk.CTkButton(self, text="close", command=self.close_program)
        self.destroy_btn.grid(column=2, row=2, padx=5, pady=5, sticky="ew")
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    """function of grid """

    def grid(self):
        self.geometry(f"{800}x{600}")
        self.minsize(400, 300)
        self.columnconfigure(index=(0, 2), weight=1, uniform="a")
        self.columnconfigure(index=1, weight=3, uniform="a")
        self.rowconfigure(index=0, weight=5, uniform="a")
        self.rowconfigure(index=1, weight=4, uniform="a")
        self.rowconfigure(index=2, weight=1, uniform="a")

    """function of browse data """

    def browse_data(self):
        path = filedialog.askopenfilename()
        try:
            self.dataframe = pd.read_csv(path)
        except:
            print("Error.....")  # show massage box.......
        else:
            """call class which create data frame"""
            self.tkinter_dataframe = DataSet_class(self, self.dataframe)
            """call class which display info of data"""
            self.show_text_box()
            """to get a data frame and send it for this classes > (left_frame >PreProcessingFrame )"""
            self.left_frame.get_df(self.dataframe)
            self.right_frame.get_df(self.dataframe)

    def close_program(self):
        self.destroy()

    def show_detail_model(self, text):
        self.left_textbox.insert(tk.END, f"{text}\n")

    # ---------------------------------------------------
    """call all function of preprossing fill"""

    def func_impute(self):
        column = self.left_frame.impute_col.get()
        strategy = self.left_frame.impute_strategy.get()
        self.dataframe = impute_data(self.dataframe, column, strategy)
        self.update_data()

    def func_encode(self):
        column = self.left_frame.encode_col.get()
        encoder = self.left_frame.encode_type.get()
        self.dataframe = encode_data(self.dataframe, column, encoder)
        self.update_data()

    def func_scale(self):
        column = self.left_frame.label_col.get()
        method = self.left_frame.scale_type.get()
        self.dataframe = scale_data(self.dataframe, column, method)
        self.update_data()

    def func_PCA(self):
        column = self.left_frame.label_col_pca.get()
        num_columns = int(self.left_frame.ent_pca.get())
        self.dataframe = PCA_data(self.dataframe, column, num_columns)
        self.update_data()

    def func_drop(self):

        column = self.left_frame.drop_col.get()

        self.dataframe = drop_na(self.dataframe, column)

        self.update_data()

    # ---------------------------------------------------
    """Updata dataframe """

    def update_data(self):
        self.tkinter_dataframe.destroy()
        self.tkinter_dataframe = DataSet_class(self, self.dataframe)

        self.dataset_info_textbox.destroy()
        self.show_text_box()

        self.left_frame.get_df(self.dataframe)
        self.right_frame.get_df(self.dataframe)

    def show_text_box(self):
        self.dataset_info_textbox = ctk.CTkTextbox(self)
        self.dataset_info_textbox.grid(column=1, row=1, padx=10, pady=10, sticky="nsew")
        """information,  descritbion  and amount of null value """
        string_buffer = StringIO()
        self.dataframe.info(buf=string_buffer)
        self.dataset_info_textbox.insert(
            index=tk.END, text=f"{string_buffer.getvalue()}\n"
        )

        self.dataset_info_textbox.insert(
            tk.END, text=f"{str(self.dataframe.describe())}\n"
        )
        self.dataset_info_textbox.insert(
            tk.END, text=f"{str(self.dataframe.isna().sum())}\n"
        )

        self.configure(state="disabled")


"""frist class of preprossing > call all function """


class PreProcessingFrame(ctk.CTkScrollableFrame):
    def __init__(
        self, master, func_impute, func_encode, func_scale, func_PCA, func_drop
    ):
        self.dataframe = None
        self.func_impute = func_impute
        self.func_encode = func_encode
        self.func_scale = func_scale
        self.func_PCA = func_PCA
        self.func_drop = func_drop

        super().__init__(master, label_text="Pre Processing")

        self.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        self.rowconfigure(index=(0, 1, 2, 3, 4, 5), weight=1, uniform="a")
        self.columnconfigure(index=(0, 1), weight=1, uniform="a")

        # ---------------------------------------------------

        self.drop_col = ctk.CTkOptionMenu(self)
        self.drop_col.grid(column=0, row=0, padx=5, pady=5, sticky="ew")
        self.drop_col.set("Column")

        self.drop_btn = ctk.CTkButton(self, text="Drop Column", command=self.func_drop)
        self.drop_btn.grid(column=1, row=0, padx=5, pady=5, sticky="ew", columnspan=2)

        # ---------------------------------------------------

        self.impute_strategy = ctk.CTkOptionMenu(
            self, values=["median", "mean", "most_frequent"]
        )  #
        self.impute_strategy.grid(column=1, row=1, padx=5, pady=5, sticky="ew")
        self.impute_strategy.set("Stratgy")

        self.impute_col = ctk.CTkOptionMenu(self)
        self.impute_col.grid(column=0, row=1, padx=5, pady=5, sticky="ew")
        self.impute_col.set("Impute Column")

        self.impute_btn = ctk.CTkButton(self, text="Impute", command=self.func_impute)
        self.impute_btn.grid(column=0, row=2, padx=5, pady=5, sticky="ew", columnspan=2)

        # ---------------------------------------------------

        self.encode_type = ctk.CTkOptionMenu(
            self, values=["Label Encode", "OneHot Encode"]
        )
        self.encode_type.grid(column=0, row=3, padx=5, pady=5, sticky="ew")
        self.encode_type.set("Method")

        self.encode_col = ctk.CTkOptionMenu(self)
        self.encode_col.grid(column=1, row=3, padx=5, pady=5, sticky="ew")
        self.encode_col.set("Encode Column")

        self.encode_btn = ctk.CTkButton(self, text="Encode", command=self.func_encode)
        self.encode_btn.grid(column=0, row=4, padx=5, pady=5, sticky="ew", columnspan=2)

        # ---------------------------------------------------

        self.scale_type = ctk.CTkOptionMenu(self, values=["minmax", "standard"])
        self.scale_type.grid(column=0, row=5, padx=10, pady=10, sticky="ew")
        self.scale_type.set("Scale Method")

        self.label_col = ctk.CTkOptionMenu(self)
        self.label_col.grid(column=1, row=5, padx=10, pady=10, sticky="ew")
        self.label_col.set("Label column")

        self.scale_btn = ctk.CTkButton(self, text="Scale", command=self.func_scale)
        self.scale_btn.grid(column=0, row=6, padx=5, pady=5, sticky="ew", columnspan=2)

        # ---------------------------------------------------

        self.ent_pca = ctk.CTkEntry(self, placeholder_text="N For PCs")
        self.ent_pca.grid(column=0, row=7, padx=10, pady=10, sticky="ew")

        self.label_col_pca = ctk.CTkOptionMenu(self)
        self.label_col_pca.grid(column=1, row=7, padx=10, pady=10, sticky="ew")
        self.label_col_pca.set("Label Column")

        self.pca_btn = ctk.CTkButton(self, text="PCA", command=self.func_PCA)
        self.pca_btn.grid(column=0, row=8, padx=10, pady=10, sticky="ew")

        self.pca_btn = ctk.CTkButton(self, text="Plot PCA", command=self.plot_pca)
        self.pca_btn.grid(column=1, row=8, padx=10, pady=10, sticky="ew")

    def plot_pca(self):
        label_col = self.label_col_pca.get()
        components = self.dataframe.drop(columns=[label_col], axis=1)
        model = PCA(n_components=int(self.ent_pca.get()))
        model.fit_transform(components)

        var = model.explained_variance_ratio_
        cs_var = np.cumsum(var)

        plt.figure(figsize=(5, 5))
        plt.bar(range(0, len(var)), var, width=0.75, color="black")
        plt.step(range(0, len(var)), cs_var, color="black")

        plt.show()

    # ---------------------------------------------------

    def get_df(self, df):
        self.dataframe = df
        self.impute_col.configure(values=self.dataframe.columns)
        self.label_col_pca.configure(values=self.dataframe.columns)
        self.label_col.configure(values=self.dataframe.columns)
        self.encode_col.configure(values=self.dataframe.columns)
        self.drop_col.configure(values=self.dataframe.columns)


# -----------------------------------------------------------------------"-----"--

# ____________________________________________________________________________________________________


class ModelsFrame(ctk.CTkScrollableFrame):
    def __init__(self, master, write_func):
        self.dataframe = None
        super().__init__(master, label_text="Models")

        self.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        self.rowconfigure(index=(0, 1, 2, 3, 4, 5), weight=1, uniform="a")
        self.columnconfigure(index=(0, 1), weight=1, uniform="a")
        self.tap_model = ctk.CTkTabview(self)
        self.tap_model.grid(
            row=0, column=0, padx=5, pady=5, sticky="nsew", columnspan=2
        )
        """ensample"""

        self.tap_model.add("Supervised")
        self.tap_model.add("UnSupervised")

        # ------------------------------------------------------------------------------------------------------

        self.super_tabs = ctk.CTkTabview(
            self.tap_model.tab("Supervised"), fg_color="gray"
        )
        self.super_tabs.pack(expand=True, fill="both")
        self.super_tabs.add("DT")
        self.super_tabs.add("SVM")
        self.super_tabs.add("KNN")
        self.super_tabs.add("ANN")
        self.super_tabs.add("LR")
        self.super_tabs.add("ES")

        # ------------------------------------------------------------------------------------------------------

        self.DT_frame = DTFrame(self.super_tabs.tab("DT"), write_func=write_func)
        self.svm_frame = SVM_Frame(self.super_tabs.tab("SVM"), write_func=write_func)
        self.knn_frame = KNN_Frame(self.super_tabs.tab("KNN"), write_func=write_func)
        self.ann3e_frame = ANN_Frame(self.super_tabs.tab("ANN"), write_func=write_func)
        self.lr_frame = LR_Frame(self.super_tabs.tab("LR"), write_func=write_func)
        self.es_frame = ES_Frame(self.super_tabs.tab("ES"), write_func=write_func)

        # ------------------------------------------------------------------------------------------------------
        self.unsuper_tabs = ctk.CTkTabview(
            self.tap_model.tab("UnSupervised"), fg_color="gray"
        )
        self.unsuper_tabs.pack(expand=True, fill="both")
        self.unsuper_tabs.add("K-Means")
        self.clusrer_frame = KMeansFrame(
            self.unsuper_tabs.tab("K-Means"), write_func=write_func
        )

    # -------------------------------------------------------------------------------------------------------------------

    def get_df(self, df):
        self.dataframe = df

        self.DT_frame.get_df(df=self.dataframe)
        self.svm_frame.get_df(df=self.dataframe)
        self.knn_frame.get_df(df=self.dataframe)
        self.ann_frame.get_df(df=self.dataframe)
        self.lr_frame.get_df(df=self.dataframe)
        self.es_frame.get_df(df=self.dataframe)
        self.KMeansFrame.get_df(df=self.dataframe)


# ----------------------------------------------------------------------------------------------------


class DataSet_class(ttk.Treeview):
    def __init__(self, master, dataframe):
        super().__init__(master)

        self.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.df = dataframe
        style = ttk.Style()
        style.configure(
            "Treeview.head",
            background="silver",
            foreground="black",
            rowheight=25,
            fieldbackground="silver",
        )
        style.configure(
            "Treeview",
            background="silver",
            foreground="black",
            rowheight=25,
            fieldbackground="silver",
        )

        self["columns"] = list(self.df.columns)
        self["show"] = "headings"
        for col in self["columns"]:
            self.heading(col, text=col)
            self.column(col, width=150)
        df_rows = self.df.to_numpy().tolist()
        for row in df_rows:
            self.insert("", "end", values=row)
        # Scrollbars for treeview
        treescrolly = tk.Scrollbar(
            self, orient="vertical", background="silver", command=self.yview
        )
        treescrollx = tk.Scrollbar(
            self, orient="horizontal", background="silver", command=self.xview
        )
        self.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
        treescrollx.pack(side="bottom", fill="x")
        treescrolly.pack(side="right", fill="y")


if __name__ == "__main__":
    app = App()
    app.mainloop()
