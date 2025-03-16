#include "../inc/evaluations.hpp"


void (*choose_version(int version))(IntArray&) {
    switch (version) {
        case 0: return bitonic_sort_v0;
        case 1: return bitonic_sort_v1;
        case 2: return bitonic_sort_v2;
        default:
            throw std::invalid_argument("choose_version: Invalid version: " + std::to_string(version) +
                                        ". Supported versions: 0, 1, 2.");
    }
}

namespace EvalTools {

    void eval_time(void (*sort_func)(IntArray&), IntArray& array, const std::string& tag) {
        using Clock = std::chrono::high_resolution_clock;

        // Measure start time
        auto start = Clock::now();

        // Call the sorting function
        sort_func(array);

        // Measure end time
        auto end = Clock::now();
        std::chrono::duration<double> elapsed_time = end - start;

        // Print execution time
        double time_taken = elapsed_time.count();
        std::cout << "[" << tag << "] "
                << "Execution Time: " << time_taken << " seconds, "
                << "Normalized Execution Time: " << (time_taken / static_cast<double>(array.length))
                << " seconds per element" << std::endl << std::endl;
    }

    void eval_sort(const IntArray& array1, const IntArray& array2, bool& eval_flag) {
        eval_flag = true;

        // Check if lengths match
        if (array1.length != array2.length) {
            eval_flag = false;
            return;
        }

        // Compare elements and check sorted order in one loop
        for (uint64_t i = 0; i < array1.length; i++) {
            if ( array1.data[i] != array2.data[i] || 
                (i > 0 && array1.data[i] < array1.data[i - 1]) ) {
                eval_flag = false;
                return;
            }
        }
    }
}
