# ngl_tools
Collection of tools developed by the Next Generation Liquefaction modeling teams.

## smt
The supported modeling team has developed a Python code that contains the following functions:

### cpt_inverse_filter(**kwargs)
 This function implements the thin-layer correction algorithm by Boulanger and DeJong (2018) to inverse-filter cone penetration test data to
  account for the effects of layerin on measured CPT resistance.
  
  Boulanger, R.W. and DeJong, J.T. (2018). 'Inverse filtering procedure to correct cone penetration data for thin-layer and transition effects.' 
  Cone Penetration Testing 2018 - Hicks, Pisano & Peuchen (Eds). 2018 Delft University of Technology, The Netherlands, ISBN 978-1-138-58449-5."
  
  Description of cpt_inverse_filter command. Variables enclosed in <> are optional.
  Method 'layer_correct' returns depth and qt_inv, where qt_inv is the corrected cone tip resistance.
  layer_correct(qt=$qt, z=$z, <fs=$fs, sigmav=$sigmav, sigmavp=$sigmavp>, <z50ref=$z50ref>, <m50=$m50>, <mq=$mq>, <mz=$mz>, <dc=$dc>,
  <N=$N>, <remove_interface=$remove_interface>, <rate_lim=$rate_lim>, <smooth=True/False>, <tol=$tol>,
  <low_pass=True/False>)
  
  qt = Numpy array of cone tip resistance values.
  fs = Nump array of cone sleeve friction values.
  z = Numpy array of depth values.
  sigmav = Numpy array of initial vertical total stress values.
  sigmavp = Numpy array of initial vertical effective stress values.
  z50ref = Scalar dimensionless depth filter parameter. Default = 4.2.
  m50 = Scalar filter exponent parameter. Default = 0.5.
  mq = Scalar filter parameter for computing z50. Default = 2.0.
  mz = Scalar filter parameter for computing w2. Default = 3.0
  dc = Scalar cone diameter in same units as z. Default = 0.03568 m.
  n_iter = Integer maximum number of iterations. Default = 500
  remove_interface = Boolean indicating whether to apply the layer interface correction. Default = True.
  rate_lim = Scalar limiting dimensionless rate change in qt for defining an interface. Default = 0.1. Only applies when remove_interface = True.
  smooth = Boolean indicating whether to apply a smoothing window to inverted data. Default = True.
  tol = Scalar convergence tolerance. Default = 1.0e-6.
  low_pass = Boolean indicating whether to apply low pass filter after convergence. Default = True.
  
  method 'remove_interface(qtrial,z,rateLim,dc)' applies the layer interface correction, and returns a vector of corrected qt values. All variables must be provided (i.e., no optional arguments).
  method 'def convolve(qt, zprime, C1, C2, z50ref, m50, mq, mz)' convolves qt with the filter, and returns a vector of filtered qt values. zprime, C1, and C2 are pre-computed in "layer_correct" to save time.
  method 'get_Ic_Q_Fr(qt, fs, sigmav, sigmavp, pa=101.325, maxiter=30)' returns soil behavior type index, dimensionless cone tip resistance, and dimensionless sleeve friction.

### convolve(qt, zprime, C1, C2, z50ref, m50, mq, mz)
  This function performs the convolution operation between the CPT tip resistance and the filter.
  It is called by cpt_inverse_filter().
  
  inputs:
  qt = Numpy array of cone tip resistance values.
  zprime = Dimensionless depth between a given depth and the cone tip. zprime = (z - z_tip) / dc
  dc = cone diameter
  C1 = A filter parameter used to define the w1 filter.
  C2 = A filter parameter that is 0.8 for points above the cone tip, and 1.0 below the cone tip.
  z50ref = Scalar dimensionless depth filter parameter.
  m50 = Scalar filter exponent parameter.
  mq = Scalar filter parameter for computing z50.
  mz = Scalar filter parameter for computing w2.
  
  outputs:
  qt_convolved = Numpy array of convolved cone tip resistance values.

### remove_interface_function(qtrial, z, rate_lim, dc)
  This function identifies layer interfaces and transition zones and sharpens the cone profile following
  the procedure described by Boulanger and DeJong (2018).  This function is called by cpt_inverse_filter().
  
  inputs:
  qtrial = Numpy array of trial values of cone tip resistance obtained by convolution.
  z = Numpy array of depth values.
  rate_lim = Scalar valued limiting rate of change of the log of cone tip resistance for identifying a transition zone.
  dc = Scalar valued cone diameter.
  
  outputs:
  q_corrected = Numpy array of cone penetration test values after sharpening.

### smooth_function(y, span)
  A function to smooth the cone data. This function is called by cpt_inverse_filter().
  
  inputs:
  y = Numpy array of values to smooth.
  span = Integer number of data points to include in smoothing window
  
  outputs:
  smooth_y = Numpy array of smoothed values.
    
### cpt_layering(qc1Ncs, Ic, depth, dGWT=0, num_layers=None, tknob=0.5)
  A function that uses the agglomerative clustering algorithm by Hudson et al. (2023) to automatically
  identify spatially contiguous soil layers with similar qc1Ncs and Ic values.
  
  Hudson, K. S., Ulmer, K., Zimmaro, P., Kramer, S. L., Stewart, J. P., and Brandenberg, S. J. (2023). 
  "Unsupervised Machine Learning for Detecting Soil Layer Boundaries from Cone Penetration Test Data." 
  Earthquake Engineering and Structural Dynamics, 52(11)
  
  Inputs:
  qc1Ncs = Numpy array of overburden and fines-corrected cone tip resistance values.
  Ic = Numpy array of soil behavior type index values.
  depth = Numpy array of depth values.
  dGWT = Scalar valued depth to groundwater.
  num_layers = Integer number of layers. Optional. Default = None, in which case the optimal number of layers is selected automatically.
  tknob = Scalar valued constant used to define layer thickness parameter in cost function. Optional. Default = 0.5.
  
  Outputs:
  ztop = Numpy array of depths to the tops of the layers.
  zbot = Numpy array of depths to the bottoms of the layers.
  qc1Ncs_lay = Numpy array of qc1Ncs values for the layers.
  Ic_lay = Numpy array of soil behavior type index values for the layers.

### get_FC_from_Ic()
  This function computes fines content from soil behavior type index following the method by Hudson et al. (2024).
  
  Hudson, K. S., Ulmer, K., Zimmaro, P., Kramer, S. L., Stewart, J. P., and Brandenberg, S. J. (2024). 
  "Relationship Between Fines Content and Soil Behavior Type Index at Liquefaction Sites." 
  Journal of Geotechnical and Geoenvironmental Engineering, 150(5)
  
  Ic = Soil behavior type index. Either scalar valued or a Numpy array.
  epsilon = Number of standard deviations from median. Scalar valued.
  FC = Fines content in percent. Same dimensions as Ic (either scalar valued or Numpy array).
  
### get_Ic_Qtn_Fr(qt, fs, sigmav, sigmavp, pa = 101.25, maxiter = 30)
  This function computes Ic, Qtn, Fr, qc1N, and qc1Ncs. Iterations are required to solve for
  these parameters because relationships among Ic, Qtn, Fr, qc1N, and qc1Ncs are implicit.
  
  Inputs:
  
  depth = Numpy array of depth values
  qt = Numpy array of cone tip resistance values
  fs = Numpy array of cone sleeve friction values
  sigmav = Numpy array of vertical total stress values
  sigmavp = Numpy array of vertical effective stress values
  FC = Numpy array of fines content values in percent
  pa = Scalar value of atmospheric pressure. Default = 101.325 kPa
  maxiter = Scaler value of the maximum number of iterations. Default = 30.
  
  Outputs: 
  
  Ic = Numpy array of soil behavior type index values. Ic = np.sqrt((3.47 - np.log10(Qtn)) ** 2 + (np.log10(Fr) + 1.22) ** 2)
  Qtn = Numpy array of dimensionless cone tip resistance values. Qtn = (qt - sigmav) / pa * (pa / sigmavp) ** n >= 0.0001
  Fr = Numpy array of dimensionless cone sleeve friction values. Fr = fs / (qt - sigmav) * 100 >= 0.001
