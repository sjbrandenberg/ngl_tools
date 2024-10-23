import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

def cpt_inverse_filter(**kwargs):
    """
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
    """
    if('qt' not in kwargs or 'z' not in kwargs):
        print('You must specify qt and z')
        return
    if('fs' in kwargs and ('sigmav' not in kwargs or 'sigmavp' not in kwargs)):
        print('If you include fs, you must also include sigmav and sigmavp')
        return
    valid_kwargs = ['qt', 'z', 'fs', 'sigmav', 'sigmavp', 'pa', 'z50ref', 'm50', 'mq', 'mz', 'dc', 'niter', 'remove_interface',
                    'rate_lim', 'smooth', 'tol', 'low_pass']
    for kwarg in kwargs:
        if(kwarg not in valid_kwargs):
            print('WARNING: You sepcified ' + kwarg + ', which is not a valid option. It has no influence on the calculation')

    qt = kwargs.get('qt')
    z = kwargs.get('z')
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
    dz = (np.max(z)-np.min(z))/len(z)
    zprime = (z[:, np.newaxis] - z) / dc
    C1 = 1 + zprime / 8.0
    C1[zprime > 0] = 1.0
    C1[zprime < -4] = 0.5
    C2 = np.ones((len(z), len(z)))
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
        qtrial = remove_interface_function(qtrial,z,rate_lim,dc)
    
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
        return z, qtrial, fs_inv
            
    return z,qtrial

def convolve(qt, zprime, C1, C2, z50ref, m50, mq, mz):
    """
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
    """

    qt[qt<0.01] = 0.01
    qt_ratio = qt / qt[:, np.newaxis]
    w2 = np.zeros(zprime.shape, dtype=float)
    filt = np.abs(zprime) < 30
    w2[filt] = np.sqrt(2.0 / (1.0 + qt_ratio[filt]**-mq))
    zprime_50 = np.zeros(zprime.shape, dtype=float)
    zprime_50[filt] = 1.0 + 2.0 * (C2[filt] * z50ref - 1.0)*(1.0 - 1.0 / (1.0 + qt_ratio[filt]**m50))
    w1 = np.zeros(zprime.shape, dtype=float)
    w1[filt] = C1[filt] / (1 + np.abs(zprime[filt] / zprime_50[filt])**mz)
    wc = w1 * w2 / np.sum(w1 * w2, axis=0)
    qt_convolved = np.sum(qt*wc.T, axis=1)
    return qt_convolved

def remove_interface_function(qtrial, z, rate_lim, dc):
    """
    This function identifies layer interfaces and transition zones and sharpens the cone profile following
    the procedure described by Boulanger and DeJong (2018).  This function is called by cpt_inverse_filter().
    
    inputs:
    qtrial = Numpy array of trial values of cone tip resistance obtained by convolution.
    z = Numpy array of depth values.
    rate_lim = Scalar valued limiting rate of change of the log of cone tip resistance for identifying a transition zone.
    dc = Scalar valued cone diameter.

    outputs:
    q_corrected = Numpy array of cone penetration test values after sharpening.
    """
    m = []
    q_corrected = np.copy(qtrial)
    dz = (np.max(z)-np.min(z))/(len(z)-1)
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
            if (z[ind2]-z[ind1])/dc > 3:
                ind3 = int(0.5*(ind1 + ind2))
                if (z[ind2]-z[ind1])/dc > 12:
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
            if (z[ind2]-z[ind1])/dc > 3:
                ind3 = int(0.5*(ind1 + ind2))
                if (z[ind2]-z[ind1])/dc > 18:
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
    for i, yval in enumerate(y):
        sub_span = int(np.min([i,(span-1)/2,len(y)-i]))
        smooth_y[i] = np.mean(y[i-sub_span:i+sub_span+1])
    return smooth_y 

def cpt_layering(qc1Ncs, Ic, depth, dGWT=0, num_layers=None, tknob=0.5):
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
    tknob = Scalar valued constant used to define layer thickness parameter in cost function. Optional. Default = 0.5.

    Outputs:
    ztop = Numpy array of depths to the tops of the layers.
    zbot = Numpy array of depths to the bottoms of the layers.
    qc1Ncs_lay = Numpy array of qc1Ncs values for the layers.
    Ic_lay = Numpy array of soil behavior type index values for the layers.
    """
    ##Standardize (normalize, "norm") the qc1Ncs and Ic values
    qc1Ncs_norm = (qc1Ncs - np.mean(qc1Ncs)) / np.std(qc1Ncs)
    Ic_norm = (Ic - np.mean(Ic)) / np.std(Ic)

    ##Create nearest-neighbor matrix (tri-diagonal ones, zeros elsewhere)
    X = np.concatenate((qc1Ncs_norm.reshape(-1, 1).T, Ic_norm.reshape(-1, 1).T)).T
    ind = np.argmin(
        np.abs(dGWT - depth)
    )  ##force a layer break at the groundwater table depth
    knn_graph = np.zeros((len(depth), len(depth)))
    for i in range(len(depth)):
        knn_graph[i][i] = 1.0
        if i > 0:
            knn_graph[i - 1][i] = 1.0
            knn_graph[i][i - 1] = 1.0
        if i < len(depth) - 1:
            knn_graph[i + 1][i] = 1.0
            knn_graph[i][i + 1] = 1.00

    ##If there is not a specified number of layers input to the function, find the optimal number of layers, otherwise use the input number of layers
    if num_layers == None:
        distk = []
        kmin = 4
        kmax = np.min([50, len(depth)])
        for k in range(kmin, kmax):
            model = AgglomerativeClustering(
                linkage="ward", connectivity=knn_graph, n_clusters=k
            )

            model.fit(X)
            labels = model.labels_
            distortion = 0
            for label in np.unique(labels):
                cluster = X[labels == label]
                center = cluster.mean(axis=0)
                center = np.array([center])
                distances = pairwise_distances(cluster, center, metric="euclidean")
                distances = distances**2
                distortion += distances.sum()
            distk.append(distortion)

        model = AgglomerativeClustering(
            linkage="ward", connectivity=knn_graph, n_clusters=1
        )
        model.fit(X)
        labels = model.labels_
        distortion = 0
        for label in np.unique(labels):
            cluster = X[labels == label]
            center = cluster.mean(axis=0)
            center = np.array([center])
            distances = pairwise_distances(cluster, center, metric="euclidean")
            distances = distances**2
            distortion += distances.sum()
        distk1 = distortion
        dist_norm = distk / distk1
        tavg = (np.max(depth)) / np.arange(kmin, kmax)
        cost2 = 0.2 * (tknob / tavg) ** 3

        model = AgglomerativeClustering(
            linkage="ward",
            connectivity=knn_graph,
            n_clusters=np.arange(kmin, kmax)[np.argmin(dist_norm + cost2)],
        )

        model.fit(X)
    else:
        model = AgglomerativeClustering(
            linkage="ward", connectivity=knn_graph, n_clusters=num_layers
        )
        model.fit(X)

    labels = model.labels_
    ztop = np.zeros(len(np.where(np.diff(labels) != 0)[0]) + 1)
    zbot = np.zeros(len(np.where(np.diff(labels) != 0)[0]) + 1)
    Ic_lay = np.zeros(len(np.where(np.diff(labels) != 0)[0]) + 1)
    qc1Ncs_lay = np.zeros(len(np.where(np.diff(labels) != 0)[0]) + 1)

    ztop[0] = depth[0]
    zbot[0] = depth[np.where(np.diff(labels) != 0)[0][0]]
    Ic_lay[0] = np.mean(Ic[0 : np.where(np.diff(labels) != 0)[0][0]])
    qc1Ncs_lay[0] = np.mean(qc1Ncs[0 : np.where(np.diff(labels) != 0)[0][0]])
    ztop[-1] = depth[np.where(np.diff(labels) != 0)[0][-1]]
    zbot[-1] = depth[-1]
    Ic_lay[-1] = np.mean(Ic[np.where(np.diff(labels) != 0)[0][-1] : -1])
    qc1Ncs_lay[-1] = np.mean(qc1Ncs[np.where(np.diff(labels) != 0)[0][-1] : -1])
    for lay in range(1, len(zbot) - 1):
        lay_indx1 = np.where(np.diff(labels) != 0)[0][lay - 1]
        lay_indx2 = np.where(np.diff(labels) != 0)[0][lay]
        ztop[lay] = depth[lay_indx1]
        zbot[lay] = depth[lay_indx2]
        Ic_lay[lay] = np.percentile(Ic[lay_indx1:lay_indx2], 50)
        qc1Ncs_lay[lay] = np.percentile(qc1Ncs[lay_indx1:lay_indx2], 50)
    if ztop[0] == ztop[1]:
        ztop = np.delete(ztop, 0)
        zbot = np.delete(zbot, 0)
        Ic_lay = np.delete(Ic_lay, 0)
        qc1Ncs_lay = np.delete(qc1Ncs_lay, 0)

    return (ztop, zbot, qc1Ncs_lay, Ic_lay)


def get_FC_from_Ic(Ic, epsilon):
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
def get_Ic_Qtn_Fr(qt, fs, sigmav, sigmavp, pa = 101.25, maxiter = 30):
    """
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
    qc1N = 
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
