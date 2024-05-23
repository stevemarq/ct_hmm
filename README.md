# ct_hmm
  
Notebook Descriptions:
- CT-HMM.ipynb runs the model using individual biomarkers.
- ND_CT_HMM.ipynb runs the model using all biomarkers at once as well as pars of biomarkers.
- to investigate how a CT-HMM works, run the Jupyter Notebook of choice
  
Technical Requirements:
- numpy, pandas, collections, sklearn, and matplotlib must be downloaded
- any version of Python is acceptable as long as it can handle the above libraires
- must be able to access Jupyter Notebooks (so have Anaconda installed)

Dataset:
- 20 patients with 4-10 observations each
- 3 classes (Healthy, Symptomatic, Critical)
- Each biomarker is modeled to mimic T cell behavior (not necessarly accurate, numerically-wise)
  
    Biomarker_1 ~ Terminal Effector T cells
  
    Biomarker_2 ~ Effector memory T cells
  
    Biomarker_3 ~ cell damage after infection
  
- The dataset is modeled in a way that follows these rules:
  
    General rules:
  
      It is rare to reach critical status.
  
      It is highly probable that one recovers.
  
      Once critical, you stay critical.
  
      It is highly probable that you will start as healthy

NOTE: If you are umable to run using the Jupyter Notebook, you can download the .py files and run them on the terminal using python main.py
If you are running it this way, ensure that all the files, including the dataset, are in the same location.
This method is the same way as running the CT-HMM.ipynb

NOTE: The class CT-HMM in both the notebook and .py file was written using the following paper as a guide:
Liu, Y. Y., Li, S., Li, F., Song, L., & Rehg, J. M. (2015). Efficient Learning of Continuous-Time Hidden Markov Models for Disease Progression. Advances in neural information processing systems, 28, 3599–3607.
