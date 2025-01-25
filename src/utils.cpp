#include "../inc/utils.hpp"


IntArray::IntArray(int length) : length(length) {
    data = new int[length];
    if (!data) {
        throw std::runtime_error("IntArray: Memory allocation failed for array of length " + std::to_string(length));
    }
}

IntArray::~IntArray() {
    delete[] data;
}


namespace ArrayUtils {
    
    void fill_array_random(IntArray& array, int max_value) {
        for (int i = 0; i < array.length; i++) {
            array.data[i] = std::rand() % max_value;
        }
    }

    void print_arr(const IntArray& array) {
        for (int i = 0; i < array.length; i++) {
            std::cout << array.data[i] << " ";
        }
        std::cout << std::endl;
    }
}
