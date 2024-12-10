import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

def cpt_inverse_filter(qt, depth, **kwargs):
    """
    This function implements the thin-layer correction algorithm by Boulanger and DeJong (2018) to inverse-filter cone penetration test data to
    account for the effects of layerin on measured CPT resistance.

    Boulanger, R.W. and DeJong, J.T. (2018). 'Inverse filtering procedure to correct cone penetration data for thin-layer and transition effects.' 
    Cone Penetration Testing 2018 - Hicks, Pisano & Peuchen (Eds). 2018 Delft University of Technology, The Netherlands, ISBN 978-1-138-58449-5."

    Description of cpt_inverse_filter command. Variables enclosed in <> are optional.
    Method 'cpt_inverse_filter' returns depth and qt_inv, where qt_inv is the corrected cone tip resistance.
    cpt_inverse_filter(qt=$qt, z=$z, <fs=$fs, sigmav=$sigmav, sigmavp=$sigmavp>, <z50ref=$z50ref>, <m50=$m50>, <mq=$mq>, <mz=$mz>, <dc=$dc>,
    <N=$N>, <remove_interface=$remove_interface>, <rate_lim=$rate_lim>, <smooth=True/False>, <tol=$tol>,
    <low_pass=True/False>)

    qt = Numpy array of cone tip resistance values.
    depth = Numpy array of depth values.
    fs = Nump array of cone sleeve friction values.
    sigmav = Numpy array of initial vertical total stress values.
    sigmavp = Numpy array of initial vertical effective stress values.
    z50ref = Scalar dimensionless depth filter parameter. Default = 4.2.
    m50 = Scalar filter exponent parameter. Default = 0.5.
    mq = Scalar filter parameter for computing z50. Default = 2.0.
    mz = Scalar filter parameter for computing w2. Default = 3.0
    dc = Scalar cone diameter in same units as z. Default = 0.03568 m.
    n_iter = Integer maximum number of iterations. Default = 500.
    remove_interface = Boolean indicating whether to apply the layer interface correction. Default = True.
    rate_lim = Scalar limiting dimensionless rate change in qt for defining an interface. Default = 0.1. Only applies when remove_interface = True.
    smooth = Boolean indicating whether to apply a smoothing window to inverted data. Default = True.
    tol = Scalar convergence tolerance. Default = 1.0e-6.
    low_pass = Boolean indicating whether to apply low pass filter after convergence. Default = True.
    
    method 'remove_interface(qtrial,z,rateLim,dc)' applies the layer interface correction, and returns a vector of corrected qt values. All variables must be provided (i.e., no optional arguments).
    method 'def convolve(qt, zprime, C1, C2, z50ref, m50, mq, mz)' convolves qt with the filter, and returns a vector of filtered qt values. zprime, C1, and C2 are pre-computed in "layer_correct" to save time.
    method 'get_Ic_Q_Fr(qt, fs, sigmav, sigmavp, pa=101.325, maxiter=30)' returns soil behavior type index, dimensionless cone tip resistance, and dimensionless sleeve friction.
    """
    if('fs' in kwargs and ('sigmav' not in kwargs or 'sigmavp' not in kwargs)):
        print('If you include fs, you must also include sigmav and sigmavp')
        return
    valid_kwargs = ['fs', 'sigmav', 'sigmavp', 'pa', 'z50ref', 'm50', 'mq', 'mz', 'dc', 'niter', 'remove_interface',
                    'rate_lim', 'smooth', 'tol', 'low_pass']
    for kwarg in kwargs:
        if(kwarg not in valid_kwargs):
            print('WARNING: You sepcified ' + kwarg + ', which is not a valid option. It has no influence on the calculation')

    pa = kwargs.get('pa', 101.325)
    z50ref = kwargs.get('z50ref', 4.2)
    m50 = kwargs.get('m50', 0.5)
    mq = kwargs.get('mq', 2.0)
    mz = kwargs.get('mz', 3.0)
    dc = kwargs.get('dc', 0.03568)
    niter = kwargs.get('niter', 500)
    remove_interface = kwargs.get('remove_interface',True)
    rate_lim = kwargs.get('rate_lim', 0.1)
    smooth = kwargs.get('smooth', True)
    tol = kwargs.get('tol', 1.e-6)
    low_pass = kwargs.get('low_pass', True)
    qt[qt<0.01] = 0.01
    dz = (np.max(depth)-np.min(depth))/len(depth)
    zprime = (depth[:, np.newaxis] - depth) / dc
    C1 = 1 + zprime / 8.0
    C1[zprime > 0] = 1.0
    C1[zprime < -4] = 0.5
    C2 = np.ones((len(depth), len(depth)))
    C2[zprime < 0] = 0.8
    qtrial_con = convolve(qt, zprime, C1, C2, z50ref, m50, mq, mz)
    qtrial = np.maximum(0.5*qt, 2*qt - qtrial_con)
    qlast = np.copy(qt)
    err_last = 0.0
    
    for j in range(niter):
        err = np.sum(np.abs(qtrial - qlast))/ np.sum(np.abs(qt))
        if np.abs(err) < tol or np.abs(err-err_last) < tol:
            break
        err_last = err
        
        qlast = np.copy(qtrial)
        qtrial_con = convolve(qtrial, zprime, C1, C2, z50ref, m50, mq, mz)
        qtrial = qt + qtrial - qtrial_con
        if smooth == True:
            qtrial = smooth_function(qtrial,np.max([3.0,np.ceil(0.866*dc/dz)]))
    
    if low_pass==True:
        qtrial = convolve(qtrial,zprime,C1, C2, 0.866,m50,mq,mz)
            
    if remove_interface==True:
        qtrial = remove_interface_function(qtrial,depth,rate_lim,dc)
    
    #apply fs correction
    if 'fs' in kwargs:
        fs = kwargs.get('fs')
        sigmav = kwargs.get('sigmav')
        sigmavp = kwargs.get('sigmavp')
        fs[fs < 0.001] = 0.001
        sigmav[sigmav < 0.001] = 0.1
        sigmavp[sigmavp<=0] = 0.1
        Rf = fs/qtrial*100
        Ic_meas, Qtn_meas, Fr_meas = get_Ic_Qtn_Fr(qt, fs, sigmav, sigmavp, pa, 30)
        Ic_inv, Qtn_inv, Fr_inv = get_Ic_Qtn_Fr(qtrial, fs, sigmav, sigmavp, pa, 30)
        F_inv = 10.0**((3.47-np.log10(Qtn_inv))/(3.47-np.log10(Qtn_meas))*(1.22+np.log10(Fr_meas))-1.22)
        F_inv[F_inv < 0.001] = 0.001
        fs_inv = F_inv/100*(qtrial-sigmav)
        fs_inv[fs_inv < 0.01] = 0.01
        return qtrial, fs_inv, Ic_inv
            
    return qtrial

def convolve(qt, zprime, C1, C2, z50ref, m50, mq, mz):
    """
    This function performs the convolution operation between the CPT tip resistance and the filter.
    It is called by cpt_inverse_filter().

    inputs:
    qt = Numpy array of cone tip resistance values.
    zprime = Dimensionless depth between a given depth and the cone tip. zprime = (z - z_tip) / dc.
    dc = cone diameter
    C1 = A filter parameter used to define the w1 filter.
    C2 = A filter parameter that is 0.8 for points above the cone tip, and 1.0 below the cone tip.
    z50ref = Scalar dimensionless depth filter parameter.
    m50 = Scalar filter exponent parameter.
    mq = Scalar filter parameter for computing z50.
    mz = Scalar filter parameter for computing w2.

    outputs:
    qt_convolved = Numpy array of convolved cone tip resistance values.
    """
    qt[qt<0.01] = 0.01
    qt_ratio = qt / qt[:, np.newaxis]
    w2 = np.zeros(zprime.shape, dtype=float)
    w2 = np.sqrt(2.0 / (1.0 + (1.0/qt_ratio)**mq))
    zprime_50 = np.zeros(zprime.shape, dtype=float)
    zprime_50 = 1.0 + 2.0 * (C2 * z50ref - 1.0)*(1.0 - 1.0 / (1.0 + qt_ratio**m50))
    w1 = np.zeros(zprime.shape, dtype=float)
    w1 = C1 / (1 + np.abs(zprime / zprime_50)**mz)
    wc = w1 * w2 / np.sum(w1 * w2, axis=0)
    qt_convolved = np.sum(qt*wc.T, axis=1)
    return qt_convolved
    # qt[qt<0.01] = 0.01
    # qt_ratio = qt / qt[:, np.newaxis]
    # w2 = np.zeros(zprime.shape, dtype=float)
    # filt = np.abs(zprime) < 100000
    # w2[filt] = np.sqrt(2.0 / (1.0 + (1.0/qt_ratio[filt])**mq))
    # zprime_50 = np.zeros(zprime.shape, dtype=float)
    # zprime_50[filt] = 1.0 + 2.0 * (C2[filt] * z50ref - 1.0)*(1.0 - 1.0 / (1.0 + qt_ratio[filt]**m50))
    # w1 = np.zeros(zprime.shape, dtype=float)
    # w1[filt] = C1[filt] / (1 + np.abs(zprime[filt] / zprime_50[filt])**mz)
    # wc = w1 * w2 / np.sum(w1 * w2, axis=0)
    # qt_convolved = np.sum(qt*wc.T, axis=1)
    # return qt_convolved

def remove_interface_function(qtrial, depth, rate_lim, dc):
    """
    This function identifies layer interfaces and transition zones and sharpens the cone profile following
    the procedure described by Boulanger and DeJong (2018).  This function is called by cpt_inverse_filter().
    
    inputs:
    qtrial = Numpy array of trial values of cone tip resistance obtained by convolution.
    depth = Numpy array of depth values.
    rate_lim = Scalar valued limiting rate of change of the log of cone tip resistance for identifying a transition zone.
    dc = Scalar valued cone diameter.

    outputs:
    q_corrected = Numpy array of cone penetration test values after sharpening.
    """
    m = []
    q_corrected = np.copy(qtrial)
    dz = (np.max(depth)-np.min(depth))/(len(depth)-1)
    InTZ = False
    load_dir = 0
    for i in range(len(qtrial)-1):
        m.append((np.log(qtrial[i+1]/qtrial[i]))/(dz/dc))
        if(np.abs(m[i])>rate_lim/5.0 and InTZ == False):
            ind1 = i
            InTZ = True
            if m[i] > 0:
                load_dir = 1
            else:
                load_dir = -1
            continue
        
        #increasing stiffness (positive m)
        if((m[i]<=rate_lim/5.0 or np.sign(m[i]) != np.sign(m[i-1])) and InTZ == True and load_dir == 1):
            InTZ = False
            ind2 = i
            if np.max(m[ind1:ind2]) < rate_lim:
                continue
            if (depth[ind2]-depth[ind1])/dc > 3:
                ind3 = int(0.5*(ind1 + ind2))
                if (depth[ind2]-depth[ind1])/dc > 12:
                    ind1 = np.max([ind3 - int(6*dc/dz),0])
                    ind2 = np.min([ind3 + int(6*dc/dz),len(qtrial)-1])
                for j in np.arange(ind1,ind2):
                    if(j<=ind1+0.4*(ind2-ind1)):
                        q_corrected[j] = qtrial[ind1]
                    else:
                        q_corrected[j] = qtrial[ind2]
                
        #decreasing stiffness (negative m)
        if((-m[i]<=rate_lim/5.0 or np.sign(m[i]) != np.sign(m[i-1])) and InTZ == True and load_dir == -1):
            InTZ = False
            ind2 = i
            if -1.0*np.min(m[ind1:ind2]) < rate_lim:
                continue
            if (depth[ind2]-depth[ind1])/dc > 3:
                ind3 = int(0.5*(ind1 + ind2))
                if (depth[ind2]-depth[ind1])/dc > 18:
                    ind1 = np.max([ind3 - int(9*dc/dz),0])
                    ind2 = np.min([ind3 + int(9*dc/dz),len(qtrial)-1])
                for j in np.arange(ind1,ind2):
                    if(j<=ind1+0.6*(ind2-ind1)):
                        q_corrected[j] = qtrial[ind1]
                    else:
                        q_corrected[j] = qtrial[ind2]
    return q_corrected
                
def smooth_function(y, span):
    """
    A function to smooth the cone data. This function is called by cpt_inverse_filter().

    inputs:
    y = Numpy array of values to smooth.
    span = Integer number of data points to include in smoothing window

    outputs:
    smooth_y = Numpy array of smoothed values.
    """
    #round down to next odd integer if span is even
    if span % 2 == 0:
        span = span - 1
    smooth_y = np.empty(len(y))
    for i in range(len(y)):
        sub_span = int(np.min([i,(span-1)/2,len(y)-i]))
        smooth_y[i] = np.mean(y[i-sub_span:i+sub_span+1])
    return smooth_y 

def cpt_layering(qc1Ncs, Ic, depth, **kwargs):
    """
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
    tref = Scalar valued constant used to define layer thickness parameter in cost function. Optional. Default = 0.5.
    wD = Scalar valued weight factor for distortion score portion of cost function. Optional. Default = 1.0.
    wT = Scalar valued weight factor for layer thickness portion of cost function. Optional. Default = 1.0.
    
    Outputs:
    ztop = Numpy array of depths to the tops of the layers.
    zbot = Numpy array of depths to the bottoms of the layers.
    qc1Ncs_lay = Numpy array of qc1Ncs values for the layers.
    Ic_lay = Numpy array of soil behavior type index values for the layers.
    """
    ##Read keyword arguments
    Nmin = kwargs.get('Nmin', 1)
    Nmax = kwargs.get('Nmax', None)
    tref = kwargs.get('tref', 0.5)
    wD = kwargs.get('wD', 1)
    wT = kwargs.get('wT', 1)

    ##Standardize (normalize, "norm") the qc1Ncs and Ic values
    qc1Ncs_norm = (qc1Ncs - np.mean(qc1Ncs)) / np.std(qc1Ncs)
    Ic_norm = (Ic - np.mean(Ic)) / np.std(Ic)

    ##Create nearest-neighbor matrix (tri-diagonal ones, zeros elsewhere)
    X = np.concatenate((qc1Ncs_norm.reshape(-1, 1).T, Ic_norm.reshape(-1, 1).T)).T
    knn_graph = np.zeros((len(depth), len(depth)))
    for i in range(len(depth)):
        knn_graph[i][i] = 1.0
        if i > 0:
            knn_graph[i - 1][i] = 1.0
            knn_graph[i][i - 1] = 1.0
        if i < len(depth) - 1:
            knn_graph[i + 1][i] = 1.0
            knn_graph[i][i + 1] = 1.0
    if(Nmax == None):
        Nmax = len(depth)

    # Compute distortion score for a single cluster to use as baseline for normalization
    JD1 = np.sum(qc1Ncs_norm**2 + Ic_norm**2)
    tavg = np.max(depth) - np.min(depth)
    JT1 = 0.2 * (tref / tavg) ** 3
    cost = wD * JD1 / JD1 + wT * JT1

    for k in np.arange(Nmin, Nmax + 1):
        model = AgglomerativeClustering(
            linkage="ward", connectivity=knn_graph, n_clusters=k
        )

        model.fit(X)
        labels = model.labels_
        JD = 0
        n_clusters = Nmin
        for label in np.unique(labels):
            cluster = X[labels == label]
            center = cluster.mean(axis=0)
            center = np.array([center])
            distances = pairwise_distances(cluster, center, metric="euclidean")
            distances = distances**2
            JD += distances.sum()
        tavg = (np.max(depth) - np.min(depth)) / k
        JT = 0.2 * (tref / tavg) ** 3
        if(k==Nmin):
            cost = (wD * JD / JD1 + wT * JT)
        elif((wD * JD / JD1 + wT * JT) < cost):
            cost = (wD * JD / JD1 + wT * JT)
        else:
            ## Model has converged. Previous number of layers was the best.
            n_clusters = k-1
            break

    model = AgglomerativeClustering(
        linkage="ward", connectivity=knn_graph, n_clusters=n_clusters
    )
    model.fit(X)
    labels = model.labels_
    unique_labels = np.unique(labels)
    ztop = np.zeros(n_clusters)
    zbot = np.zeros(n_clusters)
    qc1Ncs_lay = np.zeros(n_clusters)
    Ic_lay = np.zeros(n_clusters)
    for i in range(n_clusters):
        ztop[i] = depth[labels == unique_labels[i]][0]
        zbot[i] = depth[labels == unique_labels[i]][-1]
        qc1Ncs_lay[i] = np.median(qc1Ncs[labels == unique_labels[i]])
        Ic_lay[i] = np.median(Ic[labels == unique_labels[i]])
    sorted_indices = np.argsort(ztop)
    ztop = ztop[sorted_indices]
    zbot = zbot[sorted_indices]
    qc1Ncs_lay = qc1Ncs_lay[sorted_indices]
    Ic_lay = Ic_lay[sorted_indices]
    zmid = 0.5*(ztop[1:] + zbot[0:-1])
    for i in range(len(zmid)):
        ztop[i+1] = zmid[i]
        zbot[i] = zmid[i]
    return (ztop, zbot, qc1Ncs_lay, Ic_lay)


def get_FC_from_Ic(Ic, epsilon = 0.0):
    """
    This function computes fines content from soil behavior type index following the method by Hudson et al. (2024).

    Hudson, K. S., Ulmer, K., Zimmaro, P., Kramer, S. L., Stewart, J. P., and Brandenberg, S. J. (2024). 
    "Relationship Between Fines Content and Soil Behavior Type Index at Liquefaction Sites." 
    Journal of Geotechnical and Geoenvironmental Engineering, 150(5)

    Ic = Soil behavior type index. Either scalar valued or a Numpy array.
    epsilon = Number of standard deviations from median. Scalar valued.
    FC = Fines content in percent. Same dimensions as Ic (either scalar valued or Numpy array).
    """
    FC = (
            np.exp(2.084 * Ic - 5.066 + 1.869 * epsilon)
            / (1 + np.exp(2.084 * Ic - 5.066 + 1.869 * epsilon))
            * 100
        )
    return FC

##Function for computing qc1Ncs and Ic
def get_Ic_Qtn_Fr(qt, fs, sigmav, sigmavp, pa = 101.325, maxiter = 30):
    """
    This function computes Ic, Qtn, Fr. Iterations are required to solve for
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
    """
    
    # Set zero values to small numbers for numerical stability
    qt[qt<=0] = 0.001 * pa
    fs[fs<=0] = 0.001 * pa
    sigmav[sigmav<=0] = 0.001 * pa
    sigmavp[sigmavp<=0] = 0.001*pa

    # Develop initial estimates
    Fr = fs / (qt - sigmav) * 100
    n = np.full(len(qt), 0.5)
    Qtn = (qt - sigmav) / pa * (pa / sigmavp) ** n
    Ic = ((3.47 - np.log10(Qtn)) ** 2 + (1.22 + np.log10(Fr)) ** 2) ** 0.5
    R = 0.381 * Ic + 0.05 * (sigmavp / pa) - 0.15 - n

    # Iterate until convergence
    I = 0
    unconverged = np.full(len(qt), True)
    dRdn = np.zeros(len(qt), dtype=float)
    nlast = np.zeros(len(qt), dtype=float)
    R1 = np.zeros(len(qt), dtype=float)
    R2 = np.zeros(len(qt), dtype=float)
    while(np.any(unconverged) & (I < maxiter)):
        Qtn[unconverged] = (qt[unconverged] - sigmav[unconverged]) / pa * (pa / sigmavp[unconverged]) ** n[unconverged]
        Ic[unconverged] = ((3.47 - np.log10(Qtn[unconverged])) ** 2 + (1.22 + np.log10(Fr[unconverged])) ** 2) ** 0.5
        R1[unconverged] = 0.381 * Ic[unconverged] + 0.05*(sigmavp[unconverged] / pa) - 0.15 - n[unconverged]
        dRdn[unconverged] = (
            0.381 * (-0.28818 * ((pa / sigmavp[unconverged]) ** n[unconverged] * (qt[unconverged] - sigmav[unconverged]) / pa) / np.log(10) + 1.0) * np.log(pa / sigmavp[unconverged])
            / (0.12361 * (0.81967 * np.log(Fr[unconverged]) / np.log(10) + 1) ** 2.0 + ((-0.28818 * ((pa / sigmavp[unconverged]) ** n[unconverged] * (qt[unconverged] - sigmav[unconverged]) / pa) / np.log(10) + 1.0) ** 2.0)) ** 0.5 * np.log(10)
            - 1.0
        )
        nlast[unconverged] = np.copy(n[unconverged])
        n[unconverged] = n[unconverged] - R1[unconverged] / dRdn[unconverged]
        n[n>1.0] = 1.0
        n[n<0.0] = 0.0
        R2[unconverged] = n[unconverged] - nlast[unconverged]
        R[unconverged] = np.minimum(np.abs(R1[unconverged]), np.abs(R2[unconverged]))
        unconverged = np.abs(R) > 0.01
        I += 1
    return (Ic, Qtn, Fr)

def get_qc1N_qc1Ncs(qt, fs, sigmav, sigmavp, FC, pa = 101.325, maxiter = 30):
    """
    This function computes qc1N, and qc1Ncs. Iterations are required to solve for these parameters.
    The equations are implicit because the stress normalization exponent depends on the fines correction.

    Inputs:

    qt = Numpy array of cone tip resistance values.
    fs = Numpy array of cone sleeve friction values.
    sigmav = Numpy array of vertical total stress values.
    sigmavp = Numpy array of vertical effective stress values.
    FC = Numpy array of fines content values in percent.
    pa = Scalar value of atmospheric pressure. Optional. Default = 101.325 kPa
    maxiter = Scaler value of the maximum number of iterations. Optional. Default = 30.

    Outputs: 

    qc1N = Overburden corrected tip resistance, where exponent for overburden correction is a function of qc1Ncs.
    qc1Ncs = Overburden and fines-corrected tip resistance.
    """

    # Set zero values to small numbers for numerical stability
    qt[qt<=0] = 0.001 * pa
    fs[fs<=0] = 0.001 * pa
    sigmav[sigmav<=0] = 0.001 * pa
    sigmavp[sigmavp<=0] = 0.001 * pa

    # Iterate on overburden correction term
    m = np.full(len(qt), 0.5)
    CN = (pa / sigmavp) ** m
    qc1N = CN * qt / pa
    dqc1N = (11.9 + qc1N / 14.6) * np.exp(
        1.63 - 9.7 / (FC + 2) - (15.7 / (FC + 2)) ** 2
    )
    qc1Ncs = qc1N + dqc1N
    R = np.abs(m - (1.338 - 0.249 * qc1Ncs ** 0.264))
    I = 0
    while(np.any(R > 0.01) & (I < maxiter)):
        CN = (pa / sigmavp) ** m
        qc1N = CN * qt / pa
        dqc1N = (11.9 + qc1N / 14.6) * np.exp(
            1.63 - 9.7 / (FC + 2) - (15.7 / (FC + 2)) ** 2
        )
        qc1Ncs = qc1N + dqc1N
        R1 = m - (1.338 - 0.249 * qc1Ncs ** 0.264)
        mlast = np.copy(m)
        dRdm = (
            0.382899 * (0.018082 * (pa / sigmavp) ** m) * np.exp(-9.7 / (FC + 2.0) - 246.49 / (FC + 2.0) ** 2.0) * np.log(pa / sigmavp) / pa + 0.051725 * (pa / sigmavp) ** m * np.log(pa / sigmavp) / pa
            / ((0.06849315 * (pa / sigmavp) ** m / pa + 11.9) * np.exp(-9.7 / (FC + 2.0) - 246.49 / (FC + 2.0) ** 2.0) + 0.19592957 * (pa / sigmavp) ** m / pa) ** 0.736
            + 1.0
        )
        m = m - R1 / dRdm
        R2 = m - mlast
        R = np.minimum(np.abs(R1), np.abs(R2))
        I += 1
    return qc1N, qc1Ncs

def get_pfs(Ic):
    '''
    Inputs:
    Ic = Soil behavior type index. Numpy array.

    Output:
    pfs = Probability factor for susceptibility. Numpy array.
    '''
    pfs = 1.0 - 1.0 / (1.0 + np.exp(-14.8 * (Ic / 2.635 - 1.0)))
    return pfs

def box_cox(x, lam):
    x_hat = (x**lam - 1.0) / lam
    return x_hat

def inv_box_cox(x_hat, lam):
    x = (x_hat * lam + 1.0) ** (1 / lam)
    return x

def get_crr_hat(qc1Ncs):
    '''
    Inputs:
    qc1Ncs = Overburden and fines corrected cone tip resistance. Numpy array.

    Outputs: 
    crr_hat = Box-Cox transformed cyclic resistance ratio
    '''
    dr = 47.8 * qc1Ncs ** 0.264 - 106.3
    dr[dr < 0.0] = 0.0
    dr[dr > 132.0] = 132.0
    lambda_dr = 1.226
    dr_hat = (dr ** lambda_dr - 1.0) / lambda_dr
    crr_hat = -5.420 + 0.0193 * dr_hat
    return crr_hat

def get_csrm(amax, m, sigmav, sigmavp, depth, qc1Ncs, pa=101.325):
    '''
    Inputs:
    amax = Peak horizontal acceleration in g. Numpy array.
    m = Moment magnitude. Numpy array.
    sigmav = vertical total stress in same units as pa. Default kPa. Numpy array.
    sigmavp = vertical effective stress in same units as pa. Default kPa. Numpy array.
    z = depth in meters. Numpy array.
    pa = Atmospheric pressure. Scalar valued float. Default = 101.325 kPa.

    Output:
    csrm = Cyclic stress ratio, corrected for magnitude and overburden. Numpy array.
    '''
    alpha = np.exp(-4.373 + 0.4491 * m)
    beta = -20.11 + 6.247 * m
    rd = (1.0 - alpha) * np.exp(-depth / beta) + alpha
    neq = np.exp(0.4605 - 0.4082 * np.log(amax) + 0.2332 * m)
    msf = (14 / neq) ** 0.2
    dr = 47.8 * qc1Ncs ** 0.264 - 106.3
    dr[dr < 0.0] = 0.0
    dr[dr > 100.0] = 100.0
    ksigma = (sigmavp / pa) ** (-0.0015 * dr)
    ksigma[ksigma > 1.2] = 1.2
    csrm = 0.65 * amax * sigmav / sigmavp * rd / msf / ksigma
    return csrm

def get_pfts(csrm_hat, crr_hat, Ksat):
    '''
    Inputs:
    csrm_hat = Box-Cox transformed cyclic stress ratio, corrected for magnitude and overburden. Numpy array.
    crr_hat = Box-Cox transformed cyclic resistance ratio.

    Outputs:
    pfts = Probability factor for triggering conditional on susceptibility.
    '''
    pfts = Ksat / (1.0 + np.exp(-2.020 * (csrm_hat - crr_hat)))
    return pfts
    
def get_pfmt(ztop, Ic):
    '''
    Inputs:
    ztop = Depth to top of layer in meters. Numpy array.
    Ic = Soil behavior type index. Numpy array.

    Output:
    pfmt = Probability factor for manifestation condtional on triggering. Numpy array.
    '''
    pfmt = 1.0 / (1.0 + np.exp(-(7.613 - 0.338 * ztop - 3.042 * Ic)))
    return pfmt

def get_pmp(pfmt, pfts, pfs, t):
    '''
    Inputs:
    pfmt = Probability factor for manifestation condtional on triggering. Numpy array.
    pfts = Probability factor for triggering condtional on susceptibility. Numpy array.
    pfs = Probability factor for susceptibility. Numpy array.
    Ksat = Saturation correction factor. Numpy array.
    t = Layer thickness in meters

    Output:
    pmp = Probability of profile manifestation.
    '''
    tc = 2.0
    pmp = 1.0 - np.prod((1.0 - pfmt * pfts * pfs) ** (t / tc))
    return pmp

def smt_model(depth, qt, fs, amax, m, pa=101.325, **kwargs):
    '''
    Inputs:
    depth = Depth values. Numpy array.
    qt = Cone tip resistance values in same units as pa. Numpy array [dtype=float].
    fs = Cone sleeve friction values in same units as pa. Numpy array [dtype=float].
    amax = Maximum horizontal surface acceleration in g. Float.
    m = Earthquake magnitude. Float.
    pa = Atmospheric pressure. Float. Optional. Default = 101.325 kPa.

    Keyword Arguments (kwargs):
    gamma = Total unit weight of soil. Float.
    dGWT = Depth to groundwater table. Float.
    gammaw = Unit weight of water. Float. Optional. Default = 9.81 kN/m^3.
    sigmav = Vertical total stress in same units as pa. Numpy array [dtype=float].
    sigmavp = Vertical effective stress in same units as pa. Numpy array [dtype=float].
    Ksat = Saturation factor to multiply probability of triggering. Numpy array[dtype=float].

    Output:
    qt_inv = Inverse-filtered cone tip resistance. Numpy array [dtype=float].
    fs_inv = Inverse-filtered sleeve friction. Numpy array [dtype=float].
    Ic_inv = Inverse-filtered soil behavior type index. Numpy array [dtype=float].
    FC = Fines content computed from Ic_inv. Numpy array [dtype=float].
    qc1Ncs = Overburden- and fines-corrected cone tip resistance. Numpy array [dtype=float].
    ztop = Depth to top of layers. Numpy array [dtype=float].
    zbot = Depth to bottom of layers. Numpy array [dtype=float].
    qc1Ncs_lay = Overburden- and fines-corrected cone tip resistance for layers. Numpy array [dtype=float].
    Ic_lay = Soil behavior type index for layers. Numpy array [dtype=float].
    Ksat_lay = Ksat value for layers. Numpy array [dtype=float].
    pfs = Probability factor for susceptibility of layers. Numpy array [dtype=float].
    pfts = Probability factor for triggering conditioned on susceptibility. Numpy array [dtype=float].
    pft = Probability factor for triggering of layers. Numpy array [dtype=float].
    pfmt = Probability factor for manifestation conditioned on triggering. Numpy array [dtype=float].
    pfm = Probability factor for manifestation of layers. Numpy array [dtype=float].
    pmp = Probability of profile manifestation. Float

    Notes: 
    
    1. Either dGWT and gamma must be specified, or sigmav and sigmavp must be specified.
    2. If sigmav, sigmavp, and gamma are specified, gamma will be ignored.
    3. If sigmav and sigmavp are specified, and dGWT is not specified, dGWT will be
       inferred as the deepest point where sigmav and sigmavp are equal.
    4. If Ksat is not specified, it will be assumed equal to 0.0 above dGWT and 1.0 below dGWT.
    5. All input Numpy arrays must have the same length.
    6. The following output Numpy arrays have the same length as the input arrays: 
        qt_inv, fs_inv, Ic_inv, FC, qc1Ncs.
    7. The following output Numpy arrays have a length equal to the number of layers: 
        ztop, zbot, qc1Ncs_lay, Ic_lay, Ksat_lay, pfs, pfts, pft, pfmt, pfm
    '''
    # define constants
    pa = kwargs.get('pa', 101.325)
    lambda_csr = -0.361
    lambda_dr = 1.226

    # Read kwargs
    if(not (all(elem in kwargs for elem in ['dGWT', 'gamma']) or all(elem in kwargs for elem in ['sigmav', 'sigmavp']))):
        return "You must specify either dGWT and gamma, or sigmav and sigmavp."
    gammaw = kwargs.get('gammaw', 9.81)
    if('dGWT' in kwargs):
        dGWT = kwargs['dGWT']
        u = gammaw * (depth - dGWT)
        u[u<0.0] = 0.0
        gamma = kwargs['gamma']
        sigmav = gamma * depth
        sigmavp = sigmav - u
    if('sigmav' in kwargs):
        sigmav = kwargs.get('sigmav')
        sigmavp = kwargs.get('sigmavp')
        if('dGWT' in kwargs):
            dGWT = kwargs['dGWT']
        else:
            dGWT = depth[sigmav == sigmavp][-1]
    if('Ksat' in kwargs):
        Ksat = kwargs['Ksat']
    else:
        Ksat = np.zeros(len(depth))
        Ksat[depth > dGWT] = 1.0
    
    # apply inverse-filter algorithm
    qt_inv, fs_inv, Ic_inv = cpt_inverse_filter(qt, depth, fs=fs, sigmav=sigmav, sigmavp=sigmavp, low_pass=True, smooth=True, remove_interface=True)
    Ic, Qtn, Fr = get_Ic_Qtn_Fr(qt, fs, sigmav, sigmavp)
    Ic_inv, Qtn_inv, Fr_inv = get_Ic_Qtn_Fr(qt_inv, fs_inv, sigmav, sigmavp)

    # Compute fines content and overburden- and fines-corrected tip resistance
    FC = get_FC_from_Ic(Ic_inv, 0.0)
    qc1N, qc1Ncs = get_qc1N_qc1Ncs(qt, fs, sigmav, sigmavp, FC)
    qc1N_inv, qc1Ncs_inv = get_qc1N_qc1Ncs(qt_inv, fs_inv, sigmav, sigmavp, FC)
    
    # apply layering algorithm
    ztop, zbot, qc1Ncs_lay, Ic_lay = cpt_layering(qc1Ncs_inv, Ic_inv, depth, dGWT=dGWT, Nmin=1, Nmax=None)
    
    # insert layer break at groundwater table depth if needed
    if((dGWT not in ztop) and (dGWT not in zbot) and (dGWT > np.min(ztop)) and (dGWT < np.max(zbot))):
        Ic_dGWT = Ic_lay[ztop<dGWT][-1]
        qc1Ncs_dGWT = qc1Ncs_lay[ztop<dGWT][-1]
        Ic_lay = np.hstack((Ic_lay[zbot<dGWT], Ic_dGWT, Ic_lay[zbot>dGWT]))
        qc1Ncs_lay = np.hstack((qc1Ncs_lay[zbot<dGWT], qc1Ncs_dGWT, qc1Ncs_lay[zbot>dGWT]))
        ztop = np.hstack((ztop[ztop<dGWT], dGWT, ztop[ztop>dGWT]))
        zbot = np.hstack((zbot[zbot<dGWT], dGWT, zbot[zbot>dGWT]))
    sigmav_lay = np.interp(0.5 * (ztop + zbot), depth, sigmav)
    sigmavp_lay = np.interp(0.5 * (ztop + zbot), depth, sigmavp)
    Ksat_lay = np.interp(0.5 * (ztop + zbot), depth, Ksat)

    # compute probabilities
    crr_hat_lay = get_crr_hat(qc1Ncs_lay)
    crr_lay = inv_box_cox(crr_hat_lay, lambda_csr)
    csrm_lay = get_csrm(amax, m, sigmav_lay, sigmavp_lay, 0.5 * (ztop + zbot), qc1Ncs_lay)
    csrm_hat_lay = box_cox(csrm_lay, lambda_csr)
    t = zbot - ztop
    pfs = get_pfs(Ic_lay)
    pfts = get_pfts(csrm_hat_lay, crr_hat_lay, Ksat_lay)
    pft = pfs * pfts
    pfmt = get_pfmt(ztop, Ic_lay)
    pfm = pfs * pfts * pfmt
    pmp = get_pmp(pfmt, pfts, pfs, t)
    return (Ic, qt_inv, fs_inv, qc1Ncs_inv, Ic_inv, FC, qc1Ncs, ztop, zbot, qc1Ncs_lay, Ic_lay, Ksat_lay, pfs, pfts, pft, pfmt, pfm, pmp)
    
def smt_model_fragility(depth, qt, fs, amax, m, pa=101.325, **kwargs):
    '''
    Inputs:
    depth = Depth values. Numpy array.
    qt = Cone tip resistance values in same units as pa. Numpy array [dtype=float].
    fs = Cone sleeve friction values in same units as pa. Numpy array [dtype=float].
    amax = Maximum horizontal surface acceleration in g. Numpy array [dtype=float]
    m = Earthquake magnitude. Numpy array [dtype=float]
    pa = Atmospheric pressure. Float. Optional. Default = 101.325 kPa.

    Keyword Arguments (kwargs):
    gamma = Total unit weight of soil. Float.
    dGWT = Depth to groundwater table. Float.
    gammaw = Unit weight of water. Float. Optional. Default = 9.81 kN/m^3.
    sigmav = Vertical total stress in same units as pa. Numpy array [dtype=float].
    sigmavp = Vertical effective stress in same units as pa. Numpy array [dtype=float].
    Ksat = Saturation factor to multiply probability of triggering. Numpy array[dtype=float].

    Output:
    pmp = Probability of profile manifestation. Numpy ndarray [dtype=float]

    Notes: 
    
    1. Dimensions of pmp are len(m), len(amax) 
    '''
    # define constants
    pa = kwargs.get('pa', 101.325)
    lambda_csr = -0.361
    lambda_dr = 1.226

    # Read kwargs
    if(not (all(elem in kwargs for elem in ['dGWT', 'gamma']) or all(elem in kwargs for elem in ['sigmav', 'sigmavp']))):
        return "You must specify either dGWT and gamma, or sigmav and sigmavp."
    gammaw = kwargs.get('gammaw', 9.81)
    if('dGWT' in kwargs):
        dGWT = kwargs['dGWT']
        u = gammaw * (depth - dGWT)
        u[u<0.0] = 0.0
        gamma = kwargs['gamma']
        sigmav = gamma * depth
        sigmavp = sigmav - u
    if('sigmav' in kwargs):
        sigmav = kwargs.get('sigmav')
        sigmavp = kwargs.get('sigmavp')
        if('dGWT' in kwargs):
            dGWT = kwargs['dGWT']
        else:
            dGWT = depth[sigmav == sigmavp][-1]
    if('Ksat' in kwargs):
        Ksat = kwargs['Ksat']
    else:
        Ksat = np.zeros(len(depth))
        Ksat[depth > dGWT] = 1.0
    
    # apply inverse-filter algorithm
    qt_inv, fs_inv, Ic_inv = cpt_inverse_filter(qt, depth, fs=fs, sigmav=sigmav, sigmavp=sigmavp, low_pass=True, smooth=True, remove_interface=True)
    Ic, Qtn, Fr = get_Ic_Qtn_Fr(qt, fs, sigmav, sigmavp)
    Ic_inv, Qtn_inv, Fr_inv = get_Ic_Qtn_Fr(qt_inv, fs_inv, sigmav, sigmavp)

    # Compute fines content and overburden- and fines-corrected tip resistance
    FC = get_FC_from_Ic(Ic_inv, 0.0)
    qc1N, qc1Ncs = get_qc1N_qc1Ncs(qt, fs, sigmav, sigmavp, FC)
    qc1N_inv, qc1Ncs_inv = get_qc1N_qc1Ncs(qt_inv, fs_inv, sigmav, sigmavp, FC)
    
    # apply layering algorithm
    ztop, zbot, qc1Ncs_lay, Ic_lay = cpt_layering(qc1Ncs_inv, Ic_inv, depth, dGWT=dGWT, Nmin=1, Nmax=None)
    
    # insert layer break at groundwater table depth if needed
    if((dGWT not in ztop) and (dGWT not in zbot) and (dGWT > np.min(ztop)) and (dGWT < np.max(zbot))):
        Ic_dGWT = Ic_lay[ztop<dGWT][-1]
        qc1Ncs_dGWT = qc1Ncs_lay[ztop<dGWT][-1]
        Ic_lay = np.hstack((Ic_lay[zbot<dGWT], Ic_dGWT, Ic_lay[zbot>dGWT]))
        qc1Ncs_lay = np.hstack((qc1Ncs_lay[zbot<dGWT], qc1Ncs_dGWT, qc1Ncs_lay[zbot>dGWT]))
        ztop = np.hstack((ztop[ztop<dGWT], dGWT, ztop[ztop>dGWT]))
        zbot = np.hstack((zbot[zbot<dGWT], dGWT, zbot[zbot>dGWT]))
    sigmav_lay = np.interp(0.5 * (ztop + zbot), depth, sigmav)
    sigmavp_lay = np.interp(0.5 * (ztop + zbot), depth, sigmavp)
    Ksat_lay = np.interp(0.5 * (ztop + zbot), depth, Ksat)

    # compute probabilities
    crr_hat_lay = get_crr_hat(qc1Ncs_lay)
    crr_lay = inv_box_cox(crr_hat_lay, lambda_csr)
    csrm_lay = get_csrm(amax[np.newaxis, np.newaxis, :], m[np.newaxis, :, np.newaxis], sigmav_lay[:, np.newaxis, np.newaxis], sigmavp_lay[:, np.newaxis, np.newaxis], 0.5 * (ztop[:, np.newaxis, np.newaxis] + zbot[:, np.newaxis, np.newaxis]), qc1Ncs_lay[:, np.newaxis, np.newaxis])
    csrm_hat_lay = box_cox(csrm_lay, lambda_csr)
    t = zbot - ztop
    pfs = get_pfs(Ic_lay)
    pfts = get_pfts(csrm_hat_lay, crr_hat_lay[:,np.newaxis,np.newaxis], Ksat_lay[:, np.newaxis, np.newaxis])
    pft = pfs[:, np.newaxis, np.newaxis] * pfts
    pfmt = get_pfmt(ztop, Ic_lay)
    pfm = pft * pfmt[:, np.newaxis, np.newaxis]
    tc = 2.0
    pmp = 1.0 - np.prod((1.0 - pfmt[:, np.newaxis, np.newaxis] * pfts * pfs[:, np.newaxis, np.newaxis]) ** (t[:, np.newaxis, np.newaxis] / tc), axis=0)
    return pmp
    
