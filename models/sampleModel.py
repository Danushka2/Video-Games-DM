# -*- coding: utf-8 -*-

# Importing
import pandas as pd
import seaborn as sns
import numpy as np
import os
import dataset.cleanDs as dataset

#assign cleaned database here and use it

cleanedDS = dataset.clean_db()
print(cleanedDS)
print("Testing Purpose")
