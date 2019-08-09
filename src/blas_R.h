#ifdef __cplusplus
extern "C" {
#endif

#include <R_ext/BLAS.h>

typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;

double cblas_ddot(int n, double *x, int incx, double *y, int incy) {return ddot_(&n, x, &incx, y, &incy);}
void cblas_daxpy(int n, double a, double *x, int incx, double *y, int incy) {daxpy_(&n, &a, x, &incx, y, &incy);}
void cblas_dscal(int n, double alpha, double *x, int incx) {dscal_(&n, &alpha, x, &incx);}
double cblas_dnrm2(int n, double *x, int incx) {return dnrm2_(&n, x, &incx);}

/*	A word of warning:
	This 'cblas_dgemv' made from FORTRAN-dgemv will not work for all combinations of inputs and is not designed
	for general-purpose usage, only as a quick equivalence for the way in which it is used in this package.
*/
void cblas_dgemv(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans, int m, int n, double alpha, double *a, int lda,
	double *x, int incx, double beta, double *y, int incy)
{
	char T;
	int M, N, LDA;
	if (order == CblasRowMajor) {
		if (trans == CblasNoTrans) { T = 'T'; M = n; N = m; LDA = n; } else { T = 'N'; M = n; N = m; LDA = n; }
	} else {
		M = m; N = n; LDA = lda;
		if (trans == CblasNoTrans) { T = 'N'; } else { T = 'T'; }
	}
	dgemv_(&T, &M, &N, &alpha, a, &LDA, x, &incx, &beta, y, &incy);
}

#ifdef __cplusplus
}
#endif
