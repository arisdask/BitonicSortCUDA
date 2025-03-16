#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cstdint>

/**
 * Struct to represent an integer array and its size.
 */
struct IntArray {
    int*      data;    // Pointer to the array data.
    uint64_t  length;  // Number of elements in the array.

    // Constructor: Allocates memory for the array and initializes its size.
    explicit IntArray(uint64_t length);

    // Destructor: Frees allocated memory.
    ~IntArray();

    // Copy constructor and assignment operator disabled to avoid shallow copies.
    IntArray(const IntArray&) = delete;
    IntArray& operator=(const IntArray&) = delete;
};


/**
 * Namespace for array-related utility functions.
 */
namespace ArrayUtils {
    /**
     * Fills an `IntArray` with random integer values.
     *
     * @param array      The `IntArray` to fill with random numbers.
     * @param max_value  The maximum value (exclusive) for the random numbers.
     *                   Each number will be in the range [0, max_value-1].
     *
     * @return void
     *
     * @note   This function assumes that the array's memory has already been allocated.
     */
    void fill_array_random(IntArray& array, uint64_t max_value);


    /**
     * Prints the elements of an `IntArray` to the standard output.
     *
     * @param array The `IntArray` to be printed.
     *
     * @return void
     *
     * @note   Elements are printed in order, separated by spaces.
     *         The function does not modify the array.
     */
    void print_arr(const IntArray& array);
}

#endif // UTILS_HPP
