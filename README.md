# ct_hmm

Notebook Descriptions:
- CT-HMM.ipynb runs the model using individual biomarkers.
- ND_CT_HMM.ipynb runs the model using all biomarkers at once as well as pars of biomarkers.
  
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

