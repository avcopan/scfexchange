import abc
import numpy as np
from six import with_metaclass # required for python 2/3 compatibility

class IntegralsBase(with_metaclass(abc.ABCMeta, object)):
  """Abstract base class defining a consistent interface for integrals.

  Not sure if this is good OO design, but it made sense to me.

  Attributes:
    basis_label: A string (e.g. 'sto-3g') identifying the basis set.
    molecule: An scfexchange.Molecule object.  Together with `self.basis_label`,
      this specifies the atomic orbitals entering the integral computation.
  """

  def __init__(self, molecule, basis_label):
    self.molecule = molecule
    self.basis_label = basis_label

  @abc.abstractmethod
  def get_ao_1e_overlap_integrals(self):
    """
    Compute overlap integrals for this molecule and basis set.

    < mu(1) | nu(1) >
    """
    return

  @abc.abstractmethod
  def get_ao_1e_potential_integrals(self):
    """
    Compute nuclear potential operator for this molecule and basis set.

    < mu(1) | - 1 / 2 * nabla_1^2 | nu(1) >
    """
    return

  @abc.abstractmethod
  def get_ao_1e_kinetic_integrals(self):
    """
    Compute kinetic energy operator for this molecule and basis set.

    < mu(1) | sum_A Z_A / r_1A | nu(1) >
    """
    return

  @abc.abstractmethod
  def get_ao_2e_repulsion_integrals(self):
    """
    Compute electron-repulsion operator for this molecule and basis set.

    Returns AO basis integrals as 4d array in physicist's notation, i.e.
    < mu(1) nu(2) | 1 / r_12 | rh(1) si(2) >
    """
    return

