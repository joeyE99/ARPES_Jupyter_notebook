# ARPES_Jupyter_notebook

Jupyter notebook for ARPES data Analysis & Visualization, organized from personal labwork.

Include Functions:

* Data import from HDF file.
* Fermi edge fitting & alignment
* Conversion from angle to momentum

## 📁 Project Structure

- **`ARPES_hv_scan.ipynb`**: For photon energy  scans
- **`ARPES_map.ipynb`**: For map scans at fixed photon energy (emission & tilt angle )
- `Import_pakages.py`: Import packages for the notebook.
- `Import_functions.py`: Contains custom functions used to import data from HDF files.
- `Conversion_functions.py`: Contains custom functions used for data analysis.
- `Vis_functions.py`: Contains custom functions used for data visualization.

------

## 📊 How to Use

1. Put all the files in the same folder
2. Open the Jupyter Notebooks to see the step-by-step analysis and visualizations.

------

## 🛠️ Technologies Used

- **Python 3.x**
- **Jupyter**: For interactive documentation.
- **Matplotlib**: For data visualization.
- **Ipywidgets**: For interactive plotting
- **NumPy**: For data manipulation.
- **SciPy**: For data manipulation.
- **Lmfit**: For data fitting
- **Joblib**: For parallel computing
- **h5py**: For data importation.
- **cmcrameri**: Scientific color maps
