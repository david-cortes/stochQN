#define USE_DOUBLE 
#include "stochqn.h"
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <string.h>
#include <stdio.h>

bfgs_mem get_c_BFGS_mem(SEXP s_mem, SEXP y_mem, SEXP buffer_rho, SEXP buffer_alpha, SEXP s_bak, SEXP y_bak,
                        SEXP mem_size, SEXP mem_used, SEXP mem_st_ix, SEXP upd_freq,
                        SEXP y_reg, SEXP min_curvature)
{
    bfgs_mem BFGS_mem;
    BFGS_mem.s_mem = REAL(s_mem);
    BFGS_mem.y_mem = REAL(y_mem);
    BFGS_mem.buffer_rho = REAL(buffer_rho);
    BFGS_mem.buffer_alpha = REAL(buffer_alpha);
    BFGS_mem.s_bak = REAL(s_bak);
    BFGS_mem.y_bak = REAL(y_bak);
    BFGS_mem.mem_size = (size_t) INTEGER(mem_size)[0];
    BFGS_mem.mem_used = (size_t) INTEGER(mem_used)[0];
    BFGS_mem.mem_st_ix = (size_t) INTEGER(mem_st_ix)[0];
    BFGS_mem.upd_freq = (size_t) INTEGER(upd_freq)[0];
    BFGS_mem.y_reg = REAL(y_reg)[0];
    BFGS_mem.min_curvature = REAL(min_curvature)[0];
    return BFGS_mem;
}

fisher_mem get_c_Fisher_mem(SEXP F, SEXP buffer_y, SEXP mem_size, SEXP mem_used, SEXP mem_st_ix)
{
    fisher_mem Fisher_mem;
    Fisher_mem.F = REAL(F);
    Fisher_mem.buffer_y = REAL(buffer_y);
    Fisher_mem.mem_size = (size_t) INTEGER(mem_size)[0];
    Fisher_mem.mem_used = (size_t) INTEGER(mem_used)[0];
    Fisher_mem.mem_st_ix = (size_t) INTEGER(mem_st_ix)[0];
    return Fisher_mem;
}


workspace_oLBFGS get_c_oLBFGS(bfgs_mem *BFGS_mem, SEXP grad_prev, SEXP hess_init, SEXP niter,
                              SEXP section, SEXP nthreads, SEXP check_nan, SEXP n)
{
    workspace_oLBFGS oLBFGS;
    oLBFGS.bfgs_memory = BFGS_mem;
    oLBFGS.grad_prev = REAL(grad_prev);
    oLBFGS.hess_init = REAL(hess_init)[0];
    oLBFGS.niter = (size_t) INTEGER(niter)[0];
    oLBFGS.section = INTEGER(section)[0];
    oLBFGS.nthreads = INTEGER(nthreads)[0];
    oLBFGS.check_nan = INTEGER(check_nan)[0];
    oLBFGS.n = INTEGER(n)[0];
    return oLBFGS;
}

workspace_SQN get_c_SQN(bfgs_mem *BFGS_mem, SEXP grad_prev, SEXP x_sum, SEXP x_avg_prev, SEXP use_grad_diff,
                        SEXP niter, SEXP section, SEXP nthreads, SEXP check_nan, SEXP n)
{
    workspace_SQN SQN;
    SQN.bfgs_memory = BFGS_mem;
    SQN.grad_prev = REAL(grad_prev);
    SQN.x_sum = REAL(x_sum);
    SQN.x_avg_prev = REAL(x_avg_prev);
    SQN.use_grad_diff = INTEGER(use_grad_diff)[0];
    SQN.niter = (size_t) INTEGER(niter)[0];
    SQN.section = INTEGER(section)[0];
    SQN.nthreads = INTEGER(nthreads)[0];
    SQN.check_nan = INTEGER(check_nan)[0];
    SQN.n = INTEGER(n)[0];
    return SQN;
}

workspace_adaQN get_c_adaQN(bfgs_mem *BFGS_mem, fisher_mem *Fisher_mem, SEXP H0, SEXP grad_prev, SEXP x_sum, SEXP x_avg_prev,
                            SEXP grad_sum_sq, SEXP f_prev, SEXP max_incr, SEXP scal_reg, SEXP rmsprop_weight, SEXP use_grad_diff,
                            SEXP niter, SEXP section, SEXP nthreads, SEXP check_nan, SEXP n)
{
    workspace_adaQN adaQN;
    adaQN.bfgs_memory = BFGS_mem;
    adaQN.fisher_memory = Fisher_mem;
    adaQN.H0 = REAL(H0);
    adaQN.grad_prev = REAL(grad_prev);
    adaQN.x_sum = REAL(x_sum);
    adaQN.x_avg_prev = REAL(x_avg_prev);
    adaQN.grad_sum_sq = REAL(grad_sum_sq);
    adaQN.f_prev = REAL(f_prev)[0];
    adaQN.max_incr = REAL(max_incr)[0];
    adaQN.scal_reg = REAL(scal_reg)[0];
    adaQN.rmsprop_weight = REAL(rmsprop_weight)[0];
    adaQN.use_grad_diff = INTEGER(use_grad_diff)[0];
    adaQN.niter = (size_t) INTEGER(niter)[0];
    adaQN.section = INTEGER(section)[0];
    adaQN.nthreads = INTEGER(nthreads)[0];
    adaQN.check_nan = INTEGER(check_nan)[0];
    adaQN.n = INTEGER(n)[0];
    return adaQN;
}

SEXP r_run_oLBFGS(SEXP s_mem, SEXP y_mem, SEXP buffer_rho, SEXP buffer_alpha, SEXP s_bak, SEXP y_bak,
                  SEXP mem_size, SEXP mem_used, SEXP mem_st_ix, SEXP upd_freq,
                  SEXP y_reg, SEXP min_curvature,
                  SEXP grad_prev, SEXP hess_init, SEXP niter,
                  SEXP section, SEXP nthreads, SEXP check_nan, SEXP n,
                  SEXP x, SEXP grad, SEXP step_size,
                  SEXP x_changed, SEXP req_R, SEXP task_R, SEXP iter_info_R)
{
    bfgs_mem BFGS_mem = get_c_BFGS_mem(s_mem, y_mem, buffer_rho, buffer_alpha, s_bak, y_bak,
                                       mem_size, mem_used, mem_st_ix, upd_freq,
                                       y_reg, min_curvature);
    workspace_oLBFGS oLBFGS = get_c_oLBFGS(&BFGS_mem, grad_prev, hess_init, niter,
                                           section, nthreads, check_nan, n);
    info_enum iter_info;
    task_enum task;
    double* req;

    INTEGER(x_changed)[0] = run_oLBFGS(REAL(step_size)[0], REAL(x), REAL(grad), &req, &task, &oLBFGS, &iter_info);

    INTEGER(niter)[0]       = (int) oLBFGS.niter;
    INTEGER(section)[0]     = (int) oLBFGS.section;
    INTEGER(mem_used)[0]    = (int) BFGS_mem.mem_used;
    INTEGER(mem_st_ix)[0]   = (int) BFGS_mem.mem_st_ix;
    INTEGER(task_R)[0]      = (int) task;
    INTEGER(iter_info_R)[0] = (int) iter_info;
    memcpy(REAL(req_R), req, oLBFGS.n * sizeof(double));
    return R_NilValue;
}

SEXP r_run_SQN(SEXP s_mem, SEXP y_mem, SEXP buffer_rho, SEXP buffer_alpha, SEXP s_bak, SEXP y_bak,
               SEXP mem_size, SEXP mem_used, SEXP mem_st_ix, SEXP upd_freq,
               SEXP y_reg, SEXP min_curvature,
               SEXP grad_prev, SEXP x_sum, SEXP x_avg_prev, SEXP use_grad_diff,
               SEXP niter, SEXP section, SEXP nthreads, SEXP check_nan, SEXP n,
               SEXP x, SEXP grad, SEXP hess_vec, SEXP step_size,
               SEXP x_changed, SEXP req_R, SEXP req_vec_R, SEXP task_R, SEXP iter_info_R)
{
    bfgs_mem BFGS_mem = get_c_BFGS_mem(s_mem, y_mem, buffer_rho, buffer_alpha, s_bak, y_bak,
                                       mem_size, mem_used, mem_st_ix, upd_freq,
                                       y_reg, min_curvature);
    workspace_SQN SQN = get_c_SQN(&BFGS_mem, grad_prev, x_sum, x_avg_prev, use_grad_diff,
                                  niter, section, nthreads, check_nan, n);
    info_enum iter_info;
    task_enum task;
    double* req;
    double* req_vec;

    INTEGER(x_changed)[0] = run_SQN(REAL(step_size)[0], REAL(x), REAL(grad), REAL(hess_vec),
                                    &req, &req_vec, &task, &SQN, &iter_info);


    INTEGER(niter)[0]       = (int) SQN.niter;
    INTEGER(section)[0]     = (int) SQN.section;
    INTEGER(mem_used)[0]    = (int) BFGS_mem.mem_used;
    INTEGER(mem_st_ix)[0]   = (int) BFGS_mem.mem_st_ix;
    INTEGER(task_R)[0]      = (int) task;
    INTEGER(iter_info_R)[0] = (int) iter_info;
    memcpy(REAL(req_R), req, SQN.n * sizeof(double));
    if (task == calc_hess_vec) memcpy(REAL(req_vec_R), req_vec, SQN.n * sizeof(double));
    return R_NilValue;
}

SEXP r_run_adaQN(SEXP s_mem, SEXP y_mem, SEXP buffer_rho, SEXP buffer_alpha, SEXP s_bak, SEXP y_bak,
                 SEXP mem_size, SEXP mem_used, SEXP mem_st_ix, SEXP upd_freq,
                 SEXP y_reg, SEXP min_curvature,
                 SEXP F, SEXP buffer_y, SEXP F_mem_size, SEXP F_mem_used, SEXP F_mem_st_ix,
                 SEXP H0, SEXP grad_prev, SEXP x_sum, SEXP x_avg_prev,
                 SEXP grad_sum_sq, SEXP f_prev, SEXP max_incr, SEXP scal_reg, SEXP rmsprop_weight, SEXP use_grad_diff,
                 SEXP niter, SEXP section, SEXP nthreads, SEXP check_nan, SEXP n,
                 SEXP x, SEXP f, SEXP grad, SEXP step_size,
                 SEXP x_changed, SEXP req_R, SEXP task_R, SEXP iter_info_R)
{
    bfgs_mem BFGS_mem = get_c_BFGS_mem(s_mem, y_mem, buffer_rho, buffer_alpha, s_bak, y_bak,
                                       mem_size, mem_used, mem_st_ix, upd_freq,
                                       y_reg, min_curvature);
    fisher_mem Fisher_mem = get_c_Fisher_mem(F, buffer_y, F_mem_size, F_mem_used, F_mem_st_ix);
    workspace_adaQN adaQN = get_c_adaQN(&BFGS_mem, &Fisher_mem, H0, grad_prev, x_sum, x_avg_prev,
                                        grad_sum_sq, f_prev, max_incr, scal_reg, rmsprop_weight, use_grad_diff,
                                        niter, section, nthreads, check_nan, n);

    info_enum iter_info;
    task_enum task;
    double* req;

    INTEGER(x_changed)[0] = run_adaQN(REAL(step_size)[0], REAL(x), REAL(f)[0], REAL(grad), &req, &task, &adaQN, &iter_info);


    INTEGER(niter)[0]       = (int) adaQN.niter;
    INTEGER(section)[0]     = (int) adaQN.section;
    INTEGER(mem_used)[0]    = (int) BFGS_mem.mem_used;
    INTEGER(mem_st_ix)[0]   = (int) BFGS_mem.mem_st_ix;
    INTEGER(task_R)[0]      = (int) task;
    INTEGER(iter_info_R)[0] = (int) iter_info;
    INTEGER(F_mem_used)[0]  = (int) Fisher_mem.mem_used;
    INTEGER(F_mem_st_ix)[0] = (int) Fisher_mem.mem_st_ix;
    REAL(f_prev)[0]         = adaQN.f_prev;
    memcpy(REAL(req_R), req, adaQN.n * sizeof(double));
    return R_NilValue;
}

SEXP copy_vec(SEXP src, SEXP dest, SEXP n)
{
    memcpy(REAL(dest), REAL(src), INTEGER(n)[0] * sizeof(double));
    return R_NilValue;
}


static const R_CallMethodDef callMethods [] = {
    { "r_run_oLBFGS", (DL_FUNC) &r_run_oLBFGS, 26 },
    { "r_run_SQN",    (DL_FUNC) &r_run_SQN,    30 },
    { "r_run_adaQN",  (DL_FUNC) &r_run_adaQN,  40 },
    { "copy_vec",     (DL_FUNC) &copy_vec,     3  },
    { NULL, NULL, 0 }
};


void R_init_stochQN(DllInfo *info)
{
    R_registerRoutines(info, NULL, callMethods, NULL, NULL);
    R_useDynamicSymbols(info, TRUE);


    R_RegisterCCallable("stochQN", "initialize_oLBFGS", (DL_FUNC) &initialize_oLBFGS);
    R_RegisterCCallable("stochQN", "dealloc_oLBFGS", (DL_FUNC) &dealloc_oLBFGS);
    R_RegisterCCallable("stochQN", "initialize_SQN", (DL_FUNC) &initialize_SQN);
    R_RegisterCCallable("stochQN", "dealloc_SQN", (DL_FUNC) &dealloc_SQN);
    R_RegisterCCallable("stochQN", "initialize_adaQN", (DL_FUNC) &initialize_adaQN);
    R_RegisterCCallable("stochQN", "dealloc_adaQN", (DL_FUNC) &dealloc_adaQN);
    R_RegisterCCallable("stochQN", "run_oLBFGS", (DL_FUNC) &run_oLBFGS);
    R_RegisterCCallable("stochQN", "run_SQN", (DL_FUNC) &run_SQN);
    R_RegisterCCallable("stochQN", "run_adaQN", (DL_FUNC) &run_adaQN);
}
