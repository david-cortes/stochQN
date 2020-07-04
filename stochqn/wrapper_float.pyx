import numpy as np
cimport numpy as np

ctypedef float real_t
ctypedef np.float32_t np_real_t
np_real_t_obj = np.float32
include "pywrapper.pxi"
