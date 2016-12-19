from abc import abstractmethod
from contracts import contract, with_metaclass, ContractsMeta

class AttributeContractMeta(ContractsMeta):
  """Used to enforce initialization attributes in the common interface."""

  def __call__(cls, *args, **kwargs):
    instance = ContractsMeta.__call__(cls, *args, **kwargs)
    instance._check_common_attributes()
    return instance

class AttributeContractNotRespected(Exception):
  """Raise this when an interface fails to define the prescribed attributes."""

  def __init__(self, message):
    Exception.__init__(self, message)

def check_common_attributes(instance, attribute_dictionary):                     
  """Check whether an instance has a given set of attributes.                    
                                                                                 
  Args:                                                                          
    instance (object): An instance of a class.                                   
    attribute_dictionary (dict): A list of attribute names (keys, type `str`)    
      and along with their desired type (values, type `type`).                   
                                                                                 
  Raises:                                                                        
    ClassAttributeNotRespected: Raised when `instance` is either missing an      
      attribute or has initialized it with the wrong type.                       
  """                                                                            
  for attr, attr_type in attribute_dictionary.items():
    if not (hasattr(instance, attr) and
            isinstance(getattr(instance, attr), attr_type)):
      raise AttributeContractNotRespected(
              "Attribute '{:s}' must be initialized with type '{:s}'."
              .format(attr, attr_type.__name__))

