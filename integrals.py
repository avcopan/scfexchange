from .molecule import Molecule
from abc import abstractmethod
from contracts import contract, ContractsMeta, with_metaclass

class AttributeContractNotRespected(Exception):                                               
                                                                                 
  def __init__(self, message):                                                   
    Exception.__init__(self, message)                                            


class IntegralsCommonInterface(with_metaclass(ContractsMeta, object)):
  """Abstract base class defining a consistent interface for integrals.

  Not sure if this is good OO design, but it made sense to me.

  Attributes:
    basis_label (str): The basis set label (e.g. 'sto-3g').
    molecule: Together with `self.basis_label`, this specifies the atomic
      orbitals entereing the integral computation.
    nbf (int): The number of basis functions.
  """
  _common_attributes = {
    'basis_label': str,
    'molecule': Molecule,
    'nbf': int
  }

  def __init__(self):
    """
    Make sure the common attributes of IntegralsBase have been defined.

    Assuming subclasses always call IntegralsCommonInterface.__init__() at the
    very end of their own __init__() methods, checks to make sure all of the
    common attributes have been initialized 
    """
    for attr, attr_type in IntegralsCommonInterface._common_attributes.items():
      if not (hasattr(self, attr) and
              isinstance(getattr(self, attr), attr_type)):
        raise AttributeContractNotRespected(
                "Attribute '{:s}' must be initialized with type '{:s}'."
                .format(attr, attr_type.__name__))
    

  @abstractmethod
  @contract(returns='array[NxN](float64)')
  def get_ao_1e_overlap_integrals(self):
    """
    Compute overlap integrals for this molecule and basis set.

    < mu(1) | nu(1) >
    """
    return

  @abstractmethod
  @contract(returns='array[NxN](float64)')
  def get_ao_1e_potential_integrals(self):
    """
    Compute nuclear potential operator for this molecule and basis set.

    < mu(1) | - 1 / 2 * nabla_1^2 | nu(1) >
    """
    return

  @abstractmethod
  @contract(returns='array[NxN](float64)')
  def get_ao_1e_kinetic_integrals(self):
    """
    Compute kinetic energy operator for this molecule and basis set.

    < mu(1) | sum_A Z_A / r_1A | nu(1) >
    """
    return

  @abstractmethod
  @contract(returns='array[NxNxNxN](float64)')
  def get_ao_2e_repulsion_integrals(self):
    """
    Compute electron-repulsion operator for this molecule and basis set.

    Returns AO basis integrals as 4d array in physicist's notation, i.e.
    < mu(1) nu(2) | 1 / r_12 | rh(1) si(2) >
    """
    return


