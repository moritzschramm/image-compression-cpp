#include "huffman.hpp"
#include <queue>
#include <vector>


struct Compare {
    bool operator()(Node* a, Node* b) {
        return a->freq > b->freq;
    }
};

void buildCodes(Node* root, const std::string& str, std::vector<std::string>& huffmanCode) {
    if (!root) return;
    if (!root->left && !root->right) {
        huffmanCode[root->value] = str;
        return;
    }
    buildCodes(root->left,  str + "0", huffmanCode);
    buildCodes(root->right, str + "1", huffmanCode);
}

std::vector<std::string> encodeHuffman(const std::vector<uint8_t>& data, std::string& encodedOutput) {
    int freq[256] = {0};
    for (uint8_t b : data) freq[b]++;

    std::priority_queue<Node*, std::vector<Node*>, Compare> pq;
    for (int i = 0; i < 256; i++)
        if (freq[i] > 0) pq.push(new Node(i, freq[i]));

    if (pq.empty()) return std::vector<std::string>(256);

    while (pq.size() > 1) {
        Node* l = pq.top(); pq.pop();
        Node* r = pq.top(); pq.pop();
        pq.push(new Node(0, l->freq + r->freq, l, r));
    }

    Node* root = pq.top();

    std::vector<std::string> huffmanCode(256);
    buildCodes(root, "", huffmanCode);

    encodedOutput.clear();
    for (uint8_t b : data)
        encodedOutput += huffmanCode[b];

    return huffmanCode;
}
