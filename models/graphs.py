# -*- coding: utf-8 -*-

# Importing
import pandas as pd
import seaborn as sns
import numpy as np
import os
from dataset.cleanDs import cleanDs
import matplotlib.pyplot as plt

#assign cleaned database here and use it


################## get data from cleanDs    ############################
cleanDataset = cleanDs()
cleanedDS = cleanDataset.clean_db()



