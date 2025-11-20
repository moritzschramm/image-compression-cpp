#include "metadata.h"
#include <fstream>

void write_metadata_binary(const std::vector<SliceMetadata>& metadata, const std::string& path,
    uint32_t image_width, uint32_t image_height)
{
    std::ofstream f(path, std::ios::binary);

    // write header
    SliceRecordHeader header;
    header.magic = 0x534C4943; // "SLIC"
    header.count = metadata.size();
    header.original_width = image_width;
    header.original_height = image_height;

    f.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // write each record
    for (const auto& m : metadata) {
        SliceRecordFixed fixed;
        fixed.label  = m.label;
        fixed.x      = m.x;
        fixed.y      = m.y;
        fixed.width  = m.width;
        fixed.height = m.height;
        fixed.filename_len = m.filename.size();

        // write fixed part
        f.write(reinterpret_cast<const char*>(&fixed), sizeof(fixed));

        // write filename bytes
        f.write(m.filename.data(), m.filename.size());
    }
}

std::vector<SliceMetadata> read_metadata_binary(const std::string& path, uint32_t& image_width, uint32_t& image_height)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open metadata file");

    // read header
    SliceRecordHeader header;
    f.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (header.magic != 0x534C4943)
        throw std::runtime_error("Invalid metadata file (magic mismatch)");

    image_width = header.original_width;
    image_height = header.original_height;

    std::vector<SliceMetadata> meta;
    meta.reserve(header.count);

    // read records
    for (uint32_t i = 0; i < header.count; ++i) {

        SliceRecordFixed fixed;
        f.read(reinterpret_cast<char*>(&fixed), sizeof(fixed));

        SliceMetadata m;
        m.label = fixed.label;
        m.x = fixed.x;
        m.y = fixed.y;
        m.width = fixed.width;
        m.height = fixed.height;

        // read filename bytes
        std::string filename(fixed.filename_len, '\0');
        f.read(&filename[0], fixed.filename_len);

        m.filename = std::move(filename);

        meta.push_back(m);
    }

    return meta;
}
