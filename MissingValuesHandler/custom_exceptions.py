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
    
    
class TrainingResilienceValueError(Exception):
   """Raised when training_resilience<2"""
   
   def __init__(self, message=None):
       if message:
           self.message=message
       else:
           self.message=None
           
   def __str__(self):
       if self.message:
           return "{}".format(self.message)
       else:
           return 'training_resilience must be greater or equal to 2'
    

    
class TrainingSetError(Exception):
   """Raised when samples have a target value but no missing values in one or multiple features"""
   
   def __init__(self, message=None):
       if message:
           self.message=message
       else:
           self.message=None
           
   def __str__(self):
       if self.message:
           return "{}".format(self.message)
       else:
           return 'training_error'
    
                         
              
