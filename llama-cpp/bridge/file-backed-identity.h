#pragma once

// The caller must provide immutable regular files on a filesystem with stable,
// unique (device, inode) identities. This is not a general VFS identity probe.
#include <cerrno>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

inline int32_t lcb_file_backed_source_index(int retained_fd, uint64_t file_bytes,
                                            const std::vector<std::string> & paths, int single_source_fd = -1) {
    const auto valid = [file_bytes](const struct stat & status) {
        return S_ISREG(status.st_mode) && status.st_ino != 0 && status.st_size >= 0 &&
               uint64_t(status.st_size) == file_bytes;
    };
    const auto same = [](const struct stat & a, const struct stat & b) {
        return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size;
    };
    struct stat retained;
    if (fstat(retained_fd, &retained) != 0 || !valid(retained))
        throw std::runtime_error("file-backed rows: invalid retained file identity");
    if (paths.empty()) {
        // Only the caller can establish that this is the sole input source.
        struct stat source;
        if (single_source_fd < 0 || fstat(single_source_fd, &source) != 0 || !valid(source) || !same(retained, source))
            throw std::runtime_error("file-backed rows: no unique supplied source");
        return 0;
    }
    if (single_source_fd >= 0 || paths.size() > uint64_t(std::numeric_limits<int32_t>::max()) + 1)
        throw std::runtime_error("file-backed rows: invalid supplied sources");
    int32_t result = -1;
    for (size_t i = 0; i < paths.size(); ++i) {
        struct stat source;
        if (stat(paths[i].c_str(), &source) != 0) {
            // Upstream can skip a missing original PLE shard when an extra
            // supplies it. Other errors cannot exclude an inaccessible alias.
            if (errno == ENOENT || errno == ENOTDIR)
                continue;
            throw std::runtime_error("file-backed rows: supplied source identity unavailable");
        }
        if (source.st_dev == retained.st_dev && source.st_ino == retained.st_ino) {
            if (!valid(source) || !same(retained, source) || result >= 0)
                throw std::runtime_error("file-backed rows: ambiguous supplied source");
            result = int32_t(i);
        }
    }
    struct stat after;
    struct stat selected;
    if (result < 0 || stat(paths[size_t(result)].c_str(), &selected) != 0 || !valid(selected) ||
        !same(retained, selected) || fstat(retained_fd, &after) != 0 || !valid(after) || !same(retained, after))
        throw std::runtime_error("file-backed rows: retained source not uniquely supplied");
    return result;
}
