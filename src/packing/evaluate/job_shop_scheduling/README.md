# **Job Shop Scheduling Instance Generator & Solver (JSSP)**

This directory contains a script (`generate_instances_and_solve_gurobi.py`) for generating **Job Shop Scheduling Problem (JSSP) instances** and optionally computing **optimal makespans** using Gurobi.

## **Requirements**
- Python 3.12
- [JAX](https://jax.readthedocs.io/en/latest/)
- [Jumanji](https://instadeepai.github.io/jumanji/)
- [NumPy](https://numpy.org/install/)
- **(Optional but recommended)**: [Gurobi](https://www.gurobi.com/) for solving JSSP instances optimally.

## **1. Using Gurobi (Recommended)**
To compute the **optimal makespans** for each generated JSSP instance, Gurobi is required.

### **1.1. Obtain a Gurobi License**
Gurobi requires a valid license. You can get a free academic license or a trial license from:
- **Academic License** (for students & researchers):  
  [https://www.gurobi.com/academia/academic-program-and-licenses/](https://www.gurobi.com/academia/academic-program-and-licenses/)
- **Free Trial** (for commercial users):  
  [https://www.gurobi.com/free-trial/](https://www.gurobi.com/free-trial/)

> **Note:** The `gurobipy` package comes with a **limited license** that allows solving **small problems** without requiring a full Gurobi license. However, for larger problems, a proper license is needed.

### **1.2. Install the Gurobi Python Package**
After obtaining a license, install the Python API:
``` bash
pip install gurobipy
```

### **1.3. Set Up a Gurobi License (Without Installing the Full Package)**
If you **don't want to install the full Gurobi package**, you can still use it by setting up the license manually.

#### **Step 1: Download the Required Files**
Gurobi allows you to set up a license without installing the full software. Download the files from:  
👉 [How to set up a license without installing the full Gurobi package](https://support.gurobi.com/hc/en-us/articles/360059842732-How-do-I-set-up-a-license-without-installing-the-full-Gurobi-package)

#### **Step 2: Activate the License**
Run the following command:
``` bash
./grbprobe <LICENSE KEY>
```

#### **Step 3: Store the License File**
For **Linux**, store the license file in:
``` bash
/home/<username>/gurobi.lic
```
or

``` bash
/opt/gurobi/gurobi.lic
```

For **Windows**, store it in:
``` txt
C:\gurobi\gurobi.lic
```

For **Mac**, store it in:
``` bash
/Library/gurobi.lic
```

#### **Step 4: Verify Installation**
Run:
``` python
python -c "import gurobipy; print(gurobipy.gurobi.version())"
```
If installed correctly, it should print the Gurobi version.

## **2. Running `generate_instances_and_solve_gurobi.py`**
This script **generates JSSP instances** and optionally solves them using Gurobi.

### **Option 1: Run with Gurobi (Compute Optimal Makespans)**
If Gurobi is installed and a valid license is set up, run:
``` bash
python generate_instances_and_solve_gurobi.py
```
This will generate instances and compute their optimal makespans. However, if you don't need the optimal makespans, you can run the script by setting `USE_GUROBI = False`. You can also change the instance configurations in the file, depending on the types of instances you'd like to generate.

⚠️ **Warning:** Without Gurobi, the script **will not compute optimal makespans**, but instances will still be saved for later use.

## **3. References**
- **Gurobi Installation Guide**: [https://www.gurobi.com/documentation/](https://www.gurobi.com/documentation/)
- **License Setup (Without Full Installation)**: [https://support.gurobi.com/hc/en-us/articles/360059842732](https://support.gurobi.com/hc/en-us/articles/360059842732)
