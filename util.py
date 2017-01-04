from abc import abstractmethod
from contracts import contract, with_metaclass, ContractsMeta

class AttributeContractMeta(ContractsMeta):
  """Used to enforce initialization attributes in the common interface."""

  def __call__(cls, *args, **kwargs):
    instance = ContractsMeta.__call__(cls, *args, **kwargs)
    instance._check_attribute_contract()
    return instance

class AttributeContractNotRespected(Exception):
  """Raise this when an interface fails to define the prescribed attributes."""

  def __init__(self, message):
    Exception.__init__(self, message)

def check_attributes(instance, attribute_dictionary):                     
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

def process_options(changes, defaults):
  """Fill in default option values and complain about invalid option keys.

  Args:
    changes (dict): A dictionary containing a subset of the keys in defaults,
      with changes from the default values.
    defaults (dict): A dictionary containing the full list of option keys, with
      default values.

  Raises:
    Exception: If a key in `changes` is not in the list of valid keys.
    ValueError: If a value in `changes` has incorrect type.

  Returns:
    A copy of the `defaults` dictionary, updated by `changes`.
  """
  for key, val in changes.items():
    if not key in defaults:
      raise Exception("'{:s}' is not a valid option key for this class."
                      .format(key))
    elif not isinstance(val, type(defaults[key])):
      raise ValueError("The value for option '{:s}' has incorrect type '{:s}'."
                       .format(key, type(val).__name__))
  options = defaults.copy()
  options.update(changes)
  return options

def with_doc(docstring):
  """Decorator for adding to the docstring from another source.

  I got this from sunqm's pyscf.lib.misc.with_doc.

  Args:
    docstring (str): String to be added to function documentation.
  """
  def replace_docstring(function):
    function.__doc__ = docstring
    return function

  return replace_docstring


