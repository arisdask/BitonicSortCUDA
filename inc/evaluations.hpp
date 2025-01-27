#ifndef EVALUATIONS_HPP
#define EVALUATIONS_HPP

#include <iostream>
#include <chrono>
#include <string>
#include "utils.hpp"
#include "bitonic_sort.cuh"

/**
 * Selects the appropriate bitonic sort function based on the version number.
 *
 * @param version The version of bitonic sort to use. Supported values:
 *                - 0: `bitonic_sort_v0`
 *                - 1: `bitonic_sort_v1`
 *                - 2: `bitonic_sort_v2`
 *
 * @return A pointer to the selected bitonic sort function.
 *
 * @throw  If the version is invalid, the program throws an exception with an error message.
 */
void (*choose_version(int version))(IntArray&);


/**
 * Namespace for algorithm evaluation tools.
 */
namespace EvalTools {
    /**
     * Measures and prints the execution time of a sorting function applied to an `IntArray`.
     *
     * @param sort_func  A pointer to the sorting function to be timed.
     * @param array      The `IntArray` to be sorted.
     * @param tag        A string tag to be displayed at the beginning of the result message.
     *
     * @return void
     *
     * @note   The function times the sorting operation and prints the elapsed time
     *         in seconds. **The sorting function modifies the array in place.**
     */
    void eval_time(void (*sort_func)(IntArray&), IntArray& array, const std::string& tag);


    /**
     * Validates that two `IntArray` instances are identical in content and order.
     *
     * @param array1     The first `IntArray` for comparison.
     * @param array2     The second `IntArray` for comparison.
     * @param eval_flag  A reference to a boolean that will be updated to indicate
     *                   whether the arrays match:
     *                   - `true`:  Arrays are identical.
     *                   - `false`: Arrays differ.
     *
     * @return void
     *
     * @note   The function iterates through both arrays and compares each element.
     *         If a mismatch is found, `eval_flag` is set to `false`.
     */
    void eval_sort(const IntArray& array1, const IntArray& array2, bool& eval_flag);
}

#endif // EVALUATIONS_HPP
