# Phyllis-Take-Home-Project-

My scripts for creating cleaned csv file for feature engineering and modeling are in the preprocessing notebook. All data used is in the \data folder. 

The [Report](./Report.md) file goes over details on why I chose the specific procedure for data preprocessing, feature analysis, and algorithms used.

I used Anaconda to run my data preprocessing, feature engineering, and modeling in jupyter notebooks.

To run this, navigate to Anaconda Prompt. 
- 'cd' to this repository in terminal
  - cd path\to\this\repository
- Run the command 'jupyter notebook'
For Data preprocessing and accessing cleaned data file:

- In the Jupyter Notebook interface, open the file Data_Preprocessing_Notebook.ipynb.
- Run the first cell in the notebook to install the necessary packages and imports.

For Completing feature engineering & building Random Forest Classifier & Kmeans Clustering using cleaned dataset:

- In the Jupyter Notebook interface, open the file Feature_Engineering&_Modeling.ipynb.
- Run the first 2 cells in the notebook to install the necessary packages and imports.
- Follow the instructions in the notebook to proceed with data preprocessing, feature engineering, and modeling.


NOTE:
Without anaconda installed prior, it may be necessary to make a new conda environment and install these packages:

conda create -n census_income python=3.10 -y

conda activate census_income

conda install -y numpy pandas scikit-learn matplotlib seaborn jupyter scipy

jupyter notebook

