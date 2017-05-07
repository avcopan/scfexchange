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
