#include <stdio.h>
#include <stdlib.h>
#include "my_timer.h"
#include <x86intrin.h>
#include <omp.h>

#define min(a,b) ((a)<(b) ? (a) : (b))

#define NI 4096
#define NJ 4096
#define NK 4096

// #define NI 1024
// #define NJ 1024
// #define NK 1024
// #define NI 256
// #define NJ 256
// #define NK 256

/* Array initialization. */
static
void init_array(float C[NI*NJ], float A[NI*NK], float B[NK*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i*NJ+j] = (float)((i*j+1) % NI) / NI;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i*NK+j] = (float)(i*(j+1) % NK) / NK;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i*NJ+j] = (float)(i*(j+2) % NJ) / NJ;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(float C[NI*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      printf("C[%d][%d] = %f\n", i, j, C[i*NJ+j]);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_and_valid_array_sum(float C[NI*NJ])
{
  int i, j;

  float sum = 0.0;
  float golden_sum = 27789682688.000000;
  
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
    {
      sum += C[i*NJ+j];
      // printf("sum %f", sum);
    }

  if ( abs(sum-golden_sum)/golden_sum > 0.00001 ) // more than 0.001% error rate
    printf("Incorrect sum of C array. Expected sum: %f, your sum: %f\n", golden_sum, sum);
  else
    printf("Correct result. Sum of C array = %f\n", sum);
}


/* Main computational kernel: baseline. The whole function will be timed,
   including the call and return. DO NOT change the baseline.*/
static
void gemm_base(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i*NJ+j] *= beta;
    }
  }
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      for (k = 0; k < NK; ++k) {
	C[i*NJ+j] += alpha * A[i*NK+k] * B[k*NJ+j];
      }
    }
  }
}

/* Main computational kernel: with tiling optimization. */
static
void gemm_tile(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
// int TILE_SIZE_k = 128;
// int TILE_SIZE_j = 256;
// int TILE_SIZE_i = 512;

int TILE_SIZE_k = 8;
int TILE_SIZE_j = 16;
int TILE_SIZE_i = 32;

  // for (i = 0; i < NI; i++) {
  //   for (j = 0; j < NJ; j++) {
  //     C[i*NJ+j] *= beta;
  //   }
  
  //   for (j = 0; j < NJ; j++) {
  //     for (k = 0; k < NK; ++k) {
	// C[i*NJ+j] += alpha * A[i*NK+k] * B[k*NJ+j];
  //     }
  //   }
  // }

  for (i = 0; i < NI; i+=TILE_SIZE_i) 
  {
    for (j = 0; j < NJ; j+= TILE_SIZE_j) 
    // for (j = 0; j < NJ; j++) {
      {
      for(int i_n = i; i_n < i + TILE_SIZE_i && i_n < NI; i_n++)
        {
           

          for(int j_n = j; j_n < j + TILE_SIZE_j && j_n < NJ; j_n++)
            {
              
              C[i_n*NJ+j_n] *= beta;
            }
        }
      }
    
    for (j = 0; j < NJ; j+= TILE_SIZE_j) 
    {
      for (k = 0; k < NK; k+=TILE_SIZE_k) 
      {
        
        for(int I_new = i; I_new < i + TILE_SIZE_i && I_new < NI; I_new++)
        {

          for(int K_new = k; K_new < k + TILE_SIZE_k && K_new < NK; K_new++)
          {
            // float sum = C[I_new*NJ+J_new];

            for(int J_new = j; J_new < j + TILE_SIZE_j && J_new < NJ; J_new++)
            {
              C[I_new*NJ+J_new] += alpha * A[I_new*NK+K_new] * B[K_new*NJ+J_new];
              //  C[I_new*NJ+J_new] += alpha * A[I_new*NK+K_new] * B[K_new*NJ+J_new];
            }
            // C[I_new*NJ+J_new] = sum;

          }

        }

	// C[i*NJ+j] += alpha * A[i*NK+k] * B[k*NJ+j];
      }
    }
  }
}

/* Main computational kernel: with tiling and simd optimizations. */
static
void gemm_tile_simd(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k;
  int TILE_SIZE_k = 16;
  int TILE_SIZE_j = 32;
  int TILE_SIZE_i = 64;
  for (i = 0; i < NI; i+=TILE_SIZE_i) 
  {
    for (j = 0; j < NJ; j+= TILE_SIZE_j) 
      {
        for(int i_n = i; i_n < i + TILE_SIZE_i && i_n < NI; i_n++)
      // for(int i_n = i; i_n < i + TILE_SIZE_i && i_n < NI; i_n+=8)
        {
          // for(int j_n = j; j_n < j + TILE_SIZE_j && j_n < NJ; j_n+=8)
          for(int j_n = j; j_n < j + TILE_SIZE_j && j_n < NJ; j_n++)
            {
              // __m256 betamatrixC = _mm256_set1_ps(C[i_n*NJ+j_n] * beta);
              // _mm256_storeu_ps(&C[i_n*NJ+j_n], betamatrixC);
              C[i_n*NJ+j_n] *= beta;

             
            }
        }
      }
    
    __m256 SUM = _mm256_setzero_ps();
    __m256 matrixA = _mm256_setzero_ps();
    __m256 matrixB = _mm256_setzero_ps();
    __m256 prod = _mm256_setzero_ps();
    for (j = 0; j < NJ; j+= TILE_SIZE_j) 
    {
      for (k = 0; k < NK; k+=TILE_SIZE_k) 
      {
        
        for(int I_new = i; I_new < i + TILE_SIZE_i && I_new < NI; I_new+=8)
        {

          for(int J_new = j; J_new < j + TILE_SIZE_j && J_new < NJ; J_new+=8)
          {
            SUM = _mm256_setzero_ps();

            for(int K_new = k; K_new < k + TILE_SIZE_k && K_new < NK; K_new+=8)
            {
              matrixA = _mm256_set1_ps(alpha * A[I_new*NK+K_new]);
              matrixB = _mm256_loadu_ps(&B[K_new*NJ+J_new]);
              prod = _mm256_mul_ps(matrixA, matrixB);
              SUM = _mm256_add_ps(prod, SUM);
            }
            __m256 matrixC = _mm256_loadu_ps(&C[I_new*NJ+J_new]);
            matrixC = _mm256_add_ps(matrixC,SUM);
            _mm256_storeu_ps(&C[I_new*NJ+J_new],matrixC);
          }
        }
      }
    }
  }
}

/* Main computational kernel: with tiling, simd, and parallelization optimizations. */
static
void gemm_tile_simd_par(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k;
  int TILE_SIZE_k = 8;
  int TILE_SIZE_j = 16;
  int TILE_SIZE_i = 32;

  // int TILE_SIZE_k = 16;
  // int TILE_SIZE_j = 32;
  // int TILE_SIZE_i = 64;
  // __m256 SUM = _mm256_setzero_ps();
  // __m256 matrixA = _mm256_setzero_ps();
  // __m256 matrixB = _mm256_setzero_ps();
  // __m256 prod = _mm256_setzero_ps();
  __m256 scalar_vector = _mm256_set1_ps(beta);
  
  

  #pragma omp parallel \
  num_threads(20)
  {
      #pragma omp for schedule(dynamic)
      for (i = 0; i < NI; i+=TILE_SIZE_i) 
      {
          for (j = 0; j < NJ; j+= TILE_SIZE_j) 
          {
            for(int i_n = i; i_n < i + TILE_SIZE_i /*&& i_n < NI*/; i_n++)
          // for(int i_n = i; i_n < i + TILE_SIZE_i && i_n < NI; i_n+=8)
            {
              //for(int j_n = j; j_n < j + TILE_SIZE_j && j_n < NJ; j_n+=8)
              for(int j_n = j; j_n < j + TILE_SIZE_j /*&& j_n < NJ*/; j_n++)
                {
                  C[i_n*NJ+j_n] *= beta;
                  // __m256 matrixC = _mm256_loadu_ps(&C[i_n*NJ+j_n]);
                  // __m256 result = _mm256_mul_ps(matrixC,scalar_vector);
                  // _mm256_storeu_ps(&C[i_n*NJ+j_n], result);

                
                }
            }
          }
        
        
        for (k = 0; k < NK; k+=TILE_SIZE_k) 
        {
          for (j = 0; j < NJ; j+= TILE_SIZE_j) 
          {
            
            for(int I_new = i; I_new < i + TILE_SIZE_i /*&& I_new < NI*/; I_new++)
            {

              for(int J_new = j; J_new < j + TILE_SIZE_j /*&& J_new < NJ*/; J_new+=8)
              {
                __m256 SUM = _mm256_setzero_ps();

                // for(int K_new = k; K_new < k + TILE_SIZE_k /*&& K_new < NK*/; K_new++)
                // {
                //   __m256 matrixA = _mm256_set1_ps(alpha * A[I_new*NK+K_new]);
                //   __m256 matrixB = _mm256_loadu_ps(&B[K_new*NJ+J_new]);
                //   __m256 prod = _mm256_mul_ps(matrixA, matrixB);
                //   SUM = _mm256_add_ps(prod, SUM);
                // }
                for(int K_new = k; K_new < k + TILE_SIZE_k /*&& K_new < NK*/; K_new+=8)
                {
                  __m256 matrixA = _mm256_set1_ps(alpha * A[I_new*NK+K_new]);
                  __m256 matrixB = _mm256_loadu_ps(&B[K_new*NJ+J_new]);
                  __m256 prod = _mm256_mul_ps(matrixA, matrixB);
                  SUM = _mm256_add_ps(prod, SUM);
                  ///////////////////////////////////////////////////
                   matrixA = _mm256_set1_ps(alpha * A[I_new*NK+K_new+1]);
                   matrixB = _mm256_loadu_ps(&B[(K_new+1)*NJ+J_new]);
                   prod = _mm256_mul_ps(matrixA, matrixB);
                  SUM = _mm256_add_ps(prod, SUM);
                  ///////////////////////////////////////////////////
                   matrixA = _mm256_set1_ps(alpha * A[I_new*NK+K_new+2]);
                   matrixB = _mm256_loadu_ps(&B[(K_new+2)*NJ+J_new]);
                   prod = _mm256_mul_ps(matrixA, matrixB);
                  SUM = _mm256_add_ps(prod, SUM);
                  ///////////////////////////////////////////////////
                   matrixA = _mm256_set1_ps(alpha * A[I_new*NK+K_new+3]);
                   matrixB = _mm256_loadu_ps(&B[(K_new+3)*NJ+J_new]);
                   prod = _mm256_mul_ps(matrixA, matrixB);
                  SUM = _mm256_add_ps(prod, SUM);

                   matrixA = _mm256_set1_ps(alpha * A[I_new*NK+K_new+4]);
                   matrixB = _mm256_loadu_ps(&B[(K_new+4)*NJ+J_new]);
                   prod = _mm256_mul_ps(matrixA, matrixB);
                  SUM = _mm256_add_ps(prod, SUM);
                  ///////////////////////////////////////////////////
                   matrixA = _mm256_set1_ps(alpha * A[I_new*NK+K_new+5]);
                   matrixB = _mm256_loadu_ps(&B[(K_new+5)*NJ+J_new]);
                   prod = _mm256_mul_ps(matrixA, matrixB);
                  SUM = _mm256_add_ps(prod, SUM);
                  ///////////////////////////////////////////////////
                   matrixA = _mm256_set1_ps(alpha * A[I_new*NK+K_new+6]);
                   matrixB = _mm256_loadu_ps(&B[(K_new+6)*NJ+J_new]);
                   prod = _mm256_mul_ps(matrixA, matrixB);
                  SUM = _mm256_add_ps(prod, SUM);
                  ///////////////////////////////////////////////////
                   matrixA = _mm256_set1_ps(alpha * A[I_new*NK+K_new+7]);
                   matrixB = _mm256_loadu_ps(&B[(K_new+7)*NJ+J_new]);
                   prod = _mm256_mul_ps(matrixA, matrixB);
                  SUM = _mm256_add_ps(prod, SUM);

                }


                __m256 matrixC = _mm256_loadu_ps(&C[I_new*NJ+J_new]);
                matrixC = _mm256_add_ps(matrixC,SUM);
                
                _mm256_storeu_ps(&C[I_new*NJ+J_new],matrixC);
              }
            }
          }
        }
      }
  }
}

int main(int argc, char** argv)
{
  /* Variable declaration/allocation. */
  float *A = (float *)malloc(NI*NK*sizeof(float));
  float *B = (float *)malloc(NK*NJ*sizeof(float));
  float *C = (float *)malloc(NI*NJ*sizeof(float));

  /* opt selects which gemm version to run */
  int opt = 0;
  if(argc == 2) {
    opt = atoi(argv[1]);
  }
  //printf("option: %d\n", opt);
  
  /* Initialize array(s). */
  init_array (C, A, B);

  /* Start timer. */
  timespec timer = tic();

  switch(opt) {
  case 0: // baseline
    /* Run kernel. */
    gemm_base (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "baseline time");
    break;
  case 1: // tiling
    /* Run kernel. */
    gemm_tile (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling time");
    break;
  case 2: // tiling and simd
    /* Run kernel. */
    gemm_tile_simd (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling-simd time");
    break;
  case 3: // tiling, simd, and parallelization
    /* Run kernel. */
    gemm_tile_simd_par (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling-simd-par time");
    break;
  default: // baseline
    /* Run kernel. */
    gemm_base (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "baseline time");
  }
  /* Print results. */
  print_and_valid_array_sum(C);

  /* free memory for A, B, C */
  free(A);
  free(B);
  free(C);
  
  return 0;
}
