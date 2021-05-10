#include <cmath>		/* log2() */
#include <cstdint> 		/* int64_t, uint64_t */
#include <cstdlib>		/* srand(), rand() */
#include <ctime>		/* time() */
#include <iostream> 		/* std::cout, std::endl */

#include "../include/utils.h" 	//INCLUDE HEADER FILE

/**
 * Return vector with each element of the input at its bit-reversed position
 *
 * @param vec The vector to bit reverse
 * @param n   The length of the vector, must be a power of two
 * @return    The bit reversed vector
 */
uint64_t *bit_reverse(uint64_t *vec, uint64_t n){

	uint64_t num_bits = log2(n);

	uint64_t *result;
	result = (uint64_t *) malloc(n*sizeof(uint64_t));

	uint64_t reverse_num;
	for(uint64_t i = 0; i < n; i++){

		reverse_num = 0;
		for(uint64_t j = 0; j < num_bits; j++){

			reverse_num = reverse_num << 1;
			if(i & (1 << j)){
				reverse_num = reverse_num | 1;
			}
		}

		result[reverse_num] = vec[i];

	}

	return result;
}



/**
 * Perform the operation 'base^exp (mod m)' using the memory-efficient method
 *
 * @param base	The base of the expression
 * @param exp	The exponent of the expression
 * @param m	The modulus of the expression
 * @return 	The result of the expression
 */
uint64_t modExp(uint64_t base, uint64_t exp, uint64_t m){

	uint64_t result = 1;
	
	while(exp > 0){

		if(exp % 2){

			result = modulo(result*base, m);

		}

		exp = exp >> 1;
		base = modulo(base*base,m);
	}

	return result;
}

/**
 * Perform the operation 'base (mod m)'
 *
 * @param base	The base of the expression
 * @param m	The modulus of the expression
 * @return 	The result of the expression
 */
uint64_t modulo(int64_t base, int64_t m){
	int64_t result = base % m;

	return (result >= 0) ? result : result + m;
}

/**
 * Print an array of arbitrary length in a readable format
 *
 * @param vec	The array to be displayed
 * @param n	The length of the array
 */
void printVec(uint64_t *vec, uint64_t n){

	std::cout << "[";
	for(uint64_t i = 0; i < n; i++){

		std::cout << vec[i] << ",";

	}
	std::cout << "]" << std::endl;
}

