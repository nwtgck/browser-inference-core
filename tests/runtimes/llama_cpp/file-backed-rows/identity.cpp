// Small actual-syscall proof for the production identity helper; no model build.
#include "file-backed-identity.h"
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>

namespace {
int retained = -1;
int checks = 0;
void check(bool ok) {
    if (!ok)
        throw std::runtime_error("file identity fixture failed");
    ++checks;
}
template <class F> void rejects(F && operation) {
    bool failed = false;
    try {
        operation();
    } catch (const std::runtime_error &) {
        failed = true;
    }
    check(failed);
}
struct descriptor {
    int value;
    explicit descriptor(int fd) : value(fd) {
        check(fd >= 0);
    }
    ~descriptor() {
        close(value);
    }
};
} // namespace

extern "C" int lcb_identity_errno(const char * name) {
    if (std::strcmp(name, "ENOENT") == 0)
        return ENOENT;
    if (std::strcmp(name, "EROFS") == 0)
        return EROFS;
    if (std::strcmp(name, "EIO") == 0)
        return EIO;
    if (std::strcmp(name, "EINVAL") == 0)
        return EINVAL;
    return -1;
}

extern "C" int lcb_identity_release() {
    if (retained >= 0) {
        const int fd = retained;
        retained = -1;
        return close(fd);
    }
    return 0;
}

extern "C" int lcb_identity_again() {
    unsigned char byte = 0;
    return retained >= 0 && pread(retained, &byte, 1, 7) == 1 && byte == 7 && lseek(retained, 0, SEEK_CUR) == 13;
}

extern "C" int lcb_identity_check(const char * selected, const char * other, const char * missing) {
    try {
        check(retained == -1);
        descriptor input(open(selected, O_RDONLY));
        descriptor different(open(other, O_RDONLY));
        retained = dup(input.value);
        check(retained >= 0);
        struct stat original, duplicate, unrelated;
        check(stat(selected, &original) == 0 && fstat(retained, &duplicate) == 0 &&
              fstat(different.value, &unrelated) == 0);
        check(S_ISREG(duplicate.st_mode) && duplicate.st_ino > 0);
        check(original.st_dev == duplicate.st_dev && original.st_ino == duplicate.st_ino &&
              original.st_size == duplicate.st_size);
        check(unrelated.st_dev != duplicate.st_dev || unrelated.st_ino != duplicate.st_ino);
        check(original.st_size == unrelated.st_size && original.st_size == 256);
        check(lseek(input.value, 13, SEEK_SET) == 13);
        check(lcb_file_backed_source_index(retained, 256, {other, missing, selected}) == 2);
        check(lcb_file_backed_source_index(retained, 256, {selected}) == 0);
        check(lcb_file_backed_source_index(retained, 256, {}, input.value) == 0);
        rejects([&] { lcb_file_backed_source_index(retained, 256, {selected, selected}); });
        rejects([&] { lcb_file_backed_source_index(retained, 256, {other, missing}); });
        rejects([&] { lcb_file_backed_source_index(retained, 256, {}); });
        rejects([&] { lcb_file_backed_source_index(retained, 256, {}, different.value); });
        rejects([&] { lcb_file_backed_source_index(retained, 255, {selected}); });
        rejects([&] { lcb_file_backed_source_index(retained, 256, {selected}, input.value); });
        check(lseek(input.value, 0, SEEK_CUR) == 13);
        unsigned char byte = 0;
        check(pread(retained, &byte, 1, 7) == 1 && byte == 7);
        check(lseek(input.value, 0, SEEK_CUR) == 13);
        std::printf("{\"checks\":%d,\"device\":%llu,\"inode\":%llu,\"otherInode\":%llu,\"size\":%lld}\n", checks,
                    (unsigned long long)duplicate.st_dev, (unsigned long long)duplicate.st_ino,
                    (unsigned long long)unrelated.st_ino, (long long)duplicate.st_size);
        return checks;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        lcb_identity_release();
        return -1;
    }
}

#ifndef __EMSCRIPTEN__
#include <filesystem>
int main(int argc, char ** argv) {
    if (argc != 2)
        return 2;
    try {
        const std::string prefix = argv[1];
        const auto selected = prefix + "/selected", other = prefix + "/other", missing = prefix + "/missing";
        std::filesystem::create_directories(prefix);
        for (const auto & path : {selected, other}) {
            descriptor fd(open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600));
            unsigned char bytes[256];
            for (size_t i = 0; i < sizeof(bytes); ++i)
                bytes[i] = (unsigned char)i;
            check(write(fd.value, bytes, sizeof(bytes)) == sizeof(bytes));
        }
        check(lcb_identity_check(selected.c_str(), other.c_str(), missing.c_str()) > 0);
        // The original open/loader handles have closed; the duplicate survives.
        check(lcb_file_backed_source_index(retained, 256, {other, selected}) == 1);
        const auto hard = prefix + "/hard", soft = prefix + "/soft", loop = prefix + "/loop";
        std::filesystem::create_hard_link(selected, hard);
        std::filesystem::create_symlink(selected, soft);
        std::filesystem::create_symlink(loop, loop);
        rejects([&] { lcb_file_backed_source_index(retained, 256, {selected, hard}); });
        rejects([&] { lcb_file_backed_source_index(retained, 256, {selected, soft}); });
        rejects([&] { lcb_file_backed_source_index(retained, 256, {selected, loop}); });
        check(lcb_file_backed_source_index(retained, 256, {soft}) == 0);
        const auto moved = prefix + "/moved";
        std::filesystem::rename(selected, moved);
        rejects([&] { lcb_file_backed_source_index(retained, 256, {selected}); });
        {
            descriptor replacement(open(selected.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600));
            check(ftruncate(replacement.value, 256) == 0);
        }
        rejects([&] { lcb_file_backed_source_index(retained, 256, {selected}); });
        check(lcb_file_backed_source_index(retained, 256, {selected, moved}) == 1);
        descriptor writable(open(moved.c_str(), O_RDWR));
        const uint64_t huge = (uint64_t(1) << 32) + 19;
        check(ftruncate(writable.value, off_t(huge)) == 0);
        check(lcb_file_backed_source_index(retained, huge, {moved}) == 0);
        rejects([&] { lcb_file_backed_source_index(retained, 256, {moved}); });
        const int old = retained;
        check(lcb_identity_release() == 0);
        struct stat status;
        check(fstat(old, &status) == -1 && errno == EBADF);
        std::printf("PASS native identity %d checks, sparse >4GiB, alias/replacement/cursor/lifetime\n", checks);
        std::filesystem::remove_all(prefix);
        return 0;
    } catch (const std::exception & error) {
        lcb_identity_release();
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
#endif
