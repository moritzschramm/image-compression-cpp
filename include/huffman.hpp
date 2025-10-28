#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct Node {
    uint8_t value;
    int freq;
    Node *left, *right;
    Node(uint8_t v, int f, Node* l = nullptr, Node* r = nullptr)
        : value(v), freq(f), left(l), right(r) {}
};

void buildCodes(Node* root, const std::string& str, std::vector<std::string>& huffmanCode);
std::vector<std::string> encodeHuffman(const std::vector<uint8_t>& data, std::string& encodedOutput);
