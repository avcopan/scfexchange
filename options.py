class Options(object):
  """Common options for controlling the behavior of electronic structure codes.

  Attributes:
    e_convergence (float): Energy convergence threshold.  When the change in
      energy is less then this, the wavefunction is considered converged.
    niter (int): The number of iterations.  When this many iterations have gone
      by without reaching convergence, quit and notify the user that the
      wavefunction is unconverged.
    nfrozen (int): The number of low-energy orbitals to freeze in a post-
      Hartree-Fock computation.  If set to None, default to freezing all core
      orbitals.
  """

  def __init__(self,
               e_convergence = 1e-7,
               niter         = 20,
               nfrozen       = None):
    self.e_convergence = e_convergence
    self.niter         = niter
    self.nfrozen       = nfrozen
