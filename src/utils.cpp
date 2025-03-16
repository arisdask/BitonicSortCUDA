#include "../inc/utils.hpp"


IntArray::IntArray(uint64_t length) : length(length) {
    data = new int[length];
    if (!data) {
        throw std::runtime_error("IntArray: Memory allocation failed for array of length " + std::to_string(length));
    }
}

IntArray::~IntArray() {
    delete[] data;
}


namespace ArrayUtils {
    
    void fill_array_random(IntArray& array, uint64_t max_value) {
        for (uint64_t i = 0; i < array.length; i++) {
            array.data[i] = std::rand() % max_value;
        }
    }

    void print_arr(const IntArray& array) {
        for (uint64_t i = 0; i < array.length; i++) {
            std::cout << array.data[i] << " ";
        }
        std::cout << std::endl << std::endl;
    }
}
