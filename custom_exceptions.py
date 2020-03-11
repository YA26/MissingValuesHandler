class VariableNameError(Exception):
   """Raised when the name of a given variable is not found in the dataset"""
   
   def __init__(self, message=None):
       if message:
           self.message=message
       else:
           self.message=None
           
   def __str__(self):
       if self.message:
           return "{}".format(self.message)
       else:
           return 'invalid variable name'
    
    
class TargetVariableNameError(Exception):
   """Raised when the name of the target variable is wrong"""
   
   def __init__(self, message=None):
       if message:
           self.message=message
       else:
           self.message=None
           
   def __str__(self):
       if self.message:
           return "{}".format(self.message)
       else:
           return 'invalid target variable name'
    
    
class NoMissingValuesError(Exception):
   """Raised when there are no missing values in the dataset"""
   
   def __init__(self, message=None):
       if message:
           self.message=message
       else:
           self.message=None
           
   def __str__(self):
       if self.message:
           return "{}".format(self.message)
       else:
           return 'No nan values detected'
    
              
