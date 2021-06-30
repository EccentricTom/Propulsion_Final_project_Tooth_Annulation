# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:31:25 2020

@author: tom
"""

import os
import glob

def rename(path, old_ex, new_ex):
    for filename in glob.iglob(os.path.join(path, old_ex)):
        os.rename(filename, filename[:-4] + new_ex)