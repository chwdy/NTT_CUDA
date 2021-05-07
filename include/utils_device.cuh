#ifndef UTILS_CU_H
#define UTILS_CU_H

#include <cstdint> 	/* int64_t, uint64_t */
#include <cstdlib>	/* RAND_MAX */

/**
 * Return vector with each element of the input at its bit-reversed position
 *
 * @param vec The vector to bit reverse
 * @param n   The length of the vector, must be a power of two
 * @return    The bit reversed vector
 */
__device__ uint64_t *bit_reverse_D(uint64_t *vec, uint64_t n);

/**
 * Perform the operation 'base^exp (mod m)' using the memory-efficient method
 *
 * @param base	The base of the expression
 * @param exp	The exponent of the expression
 * @param m	The modulus of the expression
 * @return 	The result of the expression
 */
 __device__ uint64_t modExp_D(uint64_t base, uint64_t exp, uint64_t m);

/**
 * Perform the operation 'base (mod m)'
 *
 * @param base	The base of the expression
 * @param m	The modulus of the expression
 * @return 	The result of the expression
 */
 __device__ uint64_t modulo_D(int64_t base, int64_t m);

/**
 * Generate an array of arbitrary length containing random positive integers 
 *
 * @param n	The length of the array
 * @param max	The maximum value for an array element [Default: RAND_MAX]
 */
 //__device__ uint64_t *randVec_D(uint64_t n, uint64_t max=RAND_MAX);


 __device__ uint64_t log2_D( uint64_t n);
 __device__ uint64_t pow_D( uint64_t base,uint64_t power);
#endif
