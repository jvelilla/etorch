note
	description: "C bindings for OpenBLAS operations (Double Precision REAL_64)."

class
	ET_BLAS

feature -- BLAS Constants

	CblasRowMajor: INTEGER = 101
	CblasColMajor: INTEGER = 102

	CblasNoTrans: INTEGER = 111
	CblasTrans: INTEGER = 112
	CblasConjTrans: INTEGER = 113

feature -- BLAS Operations

	cblas_dgemm (order, transa, transb: INTEGER; m, n, k: INTEGER; alpha: REAL_64; a: POINTER; lda: INTEGER; b: POINTER; ldb: INTEGER; beta: REAL_64; c: POINTER; ldc: INTEGER)
			-- Perform D-GEMM: C = alpha * A * B + beta * C
			-- where A and B can be transposed or non-transposed depending on `transa` and `transb`.
		require
			valid_order: order = CblasRowMajor or order = CblasColMajor
			valid_transa: transa = CblasNoTrans or transa = CblasTrans or transa = CblasConjTrans
			valid_transb: transb = CblasNoTrans or transb = CblasTrans or transb = CblasConjTrans
			valid_m: m > 0
			valid_n: n > 0
			valid_k: k > 0
		do
			cblas_dgemm_external (order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
		end

feature {NONE} -- External bindings

	cblas_dgemm_external (order, transa, transb: INTEGER; m, n, k: INTEGER; alpha: REAL_64; a: POINTER; lda: INTEGER; b: POINTER; ldb: INTEGER; beta: REAL_64; c: POINTER; ldc: INTEGER)
		external
			"C inline use <et_blas.h>"
		alias
			"[
			cblas_dgemm((enum CBLAS_ORDER)$order, (enum CBLAS_TRANSPOSE)$transa, (enum CBLAS_TRANSPOSE)$transb, $m, $n, $k, $alpha, (const double *)$a, $lda, (const double *)$b, $ldb, $beta, (double *)$c, $ldc);
			]"
		end

end
