#pragma OPENCL EXTENSION cl_khr_fp16 : enable
float sum8(float8 a){
    return
    a.s0 + a.s1 + a.s2 +
    a.s3 + a.s4 + a.s5 +
    a.s6 + a.s7 ;
}
float sum16(float16 a){
    return
    a.s0 + a.s1 + a.s2 + a.s3 +
    a.s4 + a.s5 + a.s6 + a.s7 +
    a.s8 + a.s9 + a.sa + a.sb +
    a.sc + a.sd + a.se + a.sf ;
}
#define WRD (9)
kernel void gemm_nn9W (const int M, const int N, const int K, const float ALPHA,
		 global half *restrict A, const int lda,
		 global half *restrict B, const int ldb,
		 global half *restrict C, const int ldc
		)
{
  int i, j, k;
  half A_PART;
  int wK = K/9;   // 1/9
  int wlda = lda/9;   // 1/9
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      half Cn = C[ i*ldc + j ];
	  for (k = 0; k < wK; ++k) {
        float3  Ax1= vload_half3 (( i*wlda + k + 0), A);
        float3  Ax2= vload_half3 (( i*wlda + k + 3), A);
        float3  Ax3= vload_half3 (( i*wlda + k + 6), A);
        float3  Bx1= vload_half3 (( j*wlda + k + 0), B);
        float3  Bx2= vload_half3 (( j*wlda + k + 3), B);
        float3  Bx3= vload_half3 (( j*wlda + k + 6), B);
        Cn+= dot(Bx1, Bx2) + dot(Bx2,Ax2) + dot(Bx3,Ax3);
	  }
      C[ i*ldc + j ] = Cn;
	}
  }
}

kernel void gemm_nn20W (const int M, const int N, const int K, const float ALPHA,
		 global half *restrict A, const int lda,
		 global half *restrict B, const int ldb,
		 global half *restrict C, const int ldc
		)
{
  int i, j, k;
  half A_PART;
  int wK = K/32;    // 1/32
  int wlda = lda/32;    // 1/32
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      half Cn;
      for (k = 0, Cn = C[ i*ldc + j ];k < wK; ++k) {
        float16 Ax1= vload_half16(( i*wlda + k + 0), A);
        float16 Ax2= vload_half16(( i*wlda + k +16), A);
        float16 Bx1= vload_half16(( j*wlda + k + 0), B);
        float16 Bx2= vload_half16(( j*wlda + k +16), B);
        float16 Cx1= Bx1 * Ax1;
        float16 Cx2= Bx2 * Ax2;
        Cn+= sum16(Cx1) + sum16(Cx2);
      }
      C[ i*ldc + j ] = Cn;
    }
  }
}

