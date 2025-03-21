#include <iostream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include "../inc/utils.hpp"
#include "../inc/evaluations.hpp"
#include "../inc/bitonic_sort.cuh"

// Enable (1) or disable (0) array printing before and after sorting
#define PRINT_ARRAY 0

// Comparison function for `std::qsort`
int compare_ints(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

// Wrapper for compatibility with `IntArray`
void qsort_wrapper(IntArray& array) {
    std::qsort(array.data, array.length, sizeof(int), compare_ints);
}

// Print the header with arr_length and version
void print_header(int power, int version) {
    std::cout << "+==============================================  arr_length -> 2^"
            << power
            << "  |  version -> "
            << version
            << "  ==============================================+" << std::endl
            << std::endl;
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <q: 2^q numbers to sort> <v: version>" << std::endl;
        return EXIT_FAILURE;
    }

    uint64_t  arr_length = 1 << std::atoi(argv[1]);
    int       version    = std::atoi(argv[2]);

    if (version < 0 || version > 2) {
        std::cerr << "Invalid version. Supported versions: 0, 1, 2." << std::endl;
        return EXIT_FAILURE;
    }

    // Seed the random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Create and initialize arrays of the same length
    IntArray arr(arr_length);
    IntArray qsort_arr(arr_length);

    ArrayUtils::fill_array_random(arr, RAND_MAX);

    print_header(std::atoi(argv[1]), version);

    #if PRINT_ARRAY
        std::cout << "Original Array:" << std::endl;
        ArrayUtils::print_arr(arr);
    #endif

    // Copy the data from `arr` to `qsort_arr`
    std::memcpy(qsort_arr.data, arr.data, arr.length * sizeof(int));

    // Sort using `std::qsort` for validation and performance comparison
    EvalTools::eval_time(qsort_wrapper, qsort_arr, "qsort");

    // Select the bitonic sort version
    void (*bitonic_sort)(IntArray&) = choose_version(version);

    // Sort using the selected bitonic sort version
    EvalTools::eval_time(bitonic_sort, arr, "bitonic sort V" + std::to_string(version));

    #if PRINT_ARRAY
        std::cout << "Sorted Array (Bitonic Sort V" << version << "):" << std::endl;
        ArrayUtils::print_arr(arr);
    #endif

    // Validate the result
    bool eval_flag = false;
    EvalTools::eval_sort(arr, qsort_arr, eval_flag);

    if (eval_flag) {
        std::cout << "Validation successful: The arrays of qsort and bitonic are sorted!!" << std::endl;
    } else {
        std::cout << "Validation failed: The arrays of qsort or bitonic aren't sorted :(" << std::endl;
    }

    return EXIT_SUCCESS;
}
