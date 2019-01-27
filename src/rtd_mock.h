double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy) {return 0;}
double cblas_dnrm2(const int N, const double *X, const int incX) {return 0;}
void cblas_dscal(const int N, const double alpha, double *X, const int incX) {return;}
void cblas_dgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const int m, const int n,
	const double alpha, const double  *a, const int lda,  const double  *x, const int incx,  const double beta,
	double  *y, const int incy) {return;}
void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy) {return;} 
