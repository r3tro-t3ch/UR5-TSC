import matplotlib.pyplot as plt
import numpy as np


class Logger:  
    def __init__(self):
        self.data = {}
        

    def log_data(self,**kwargs):
        for parameter, value in kwargs.items():
            if parameter not in self.data:
                self.data[parameter] = []
            self.data[parameter].append(value)
    
    def log_dump(self,filename):
        pass