#include <iostream>
#include "huffman.hpp"

int main() {

    std::cout << "Enter text (binary-safe input, but reads until newline):\n";
    std::string input;
    getline(std::cin, input);

    std::vector<uint8_t> data(input.begin(), input.end());

    std::string encoded;
    auto codes = encodeHuffman(data, encoded);

    std::cout << "Encoded: " << encoded << "\n";

    return 0;
}
