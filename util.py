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


def check_attributes(instance, attribute_types):
    """Check whether an instance has a given set of attributes.                    
                                                                                   
    Args:                                                                          
      instance (object): An instance of a class.                                   
      attribute_types (dict): A list of attribute names (keys, type `str`), along
        with their desired type (values, type `type`).
  
    Raises:
      AttributeContractNotRespected: Raised when `instance` is either missing an      
        attribute or has initialized it with the wrong type.                       
    """
    for attr, attr_type in attribute_types.items():
        if not (hasattr(instance, attr) and
                    isinstance(getattr(instance, attr), attr_type)):
            raise AttributeContractNotRespected(
                "Attribute '{:s}' must be initialized with type '{:s}'."
                    .format(attr, attr_type.__name__))


def process_options(specified_options, default_options):
    """Fill in default option values and complain about invalid option keys.
  
    Args:
      specified_options (dict): A dictionary containing a subset of the keys in
        default_options, specifying changes to the default values.
      default_options (dict): A dictionary containing the full list of option
        keys, along with their default values.
  
    Raises:
      Exception: If a key in `specified_options` is not in the list of valid keys.
      ValueError: If a value in `specified_options` has incorrect type.
  
    Returns:
      A copy of the `default_options` dictionary, updated by `specified_options`.
    """
    for key, val in specified_options.items():
        if not key in default_options:
            raise Exception("'{:s}' is not a valid option key for this class."
                            .format(key))
        elif not isinstance(val, type(default_options[key])):
            raise ValueError(
                "The value for option '{:s}' has incorrect type '{:s}'."
                .format(key, type(val).__name__))
    options = default_options.copy()
    options.update(specified_options)
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


def compute_if_unknown(instance, attr_name, compute_attr, set_attr=False):
    """Compute a value if the instance doesn't know it already.
  
    If the instance has this value stored as an attribute, return that.  Otherwise
    invoke the (presumably expensive) computation to get the value and save it
    as an attribute for later use.
  
    Args:
      instance (instance): An instance of a class.
      attr_name (str): The name of the attribute to return.
      compute_attr (function): A function which can be called to compute the
        value of the attribute, if unknown.
      set_attr (bool): Store the computed value as an attribute for later use?
  
    Returns:
      The attribute value.
    """
    if hasattr(instance, attr_name):
        return getattr(instance, attr_name)
    else:
        attr = compute_attr()
        if set_attr:
            setattr(instance, attr_name, attr)
        return attr
