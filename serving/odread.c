// odread.c - the AirLLM-review probe: O_DIRECT read bandwidth of the served GGUF (seq/strided, N threads).
// Measured 2026-10-14 on the served Q6K GGUF: seq 1t 1685, seq 4t 2018, stride 1t 1085, stride 4t 1112 MB/s.
// Usage: cc -O2 -o odread odread.c -lpthread && ./odread <file> <seq|stride> <MiB> <threads>
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static const char* g_path;
static long g_file_mb;
static long g_total_mb;
static int g_stride;
static _Atomic long g_next;
static _Atomic long g_done_mb;
static _Atomic long g_open_err;

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec / 1e9;
}

static void* worker(void* arg) {
    (void) arg;
    int fd = open(g_path, O_RDONLY | O_DIRECT);
    if (fd < 0) { g_open_err = 1; return NULL; }
    void* buf; if (posix_memalign(&buf, 4096, 1 << 20)) return NULL;
    for (;;) {
        long i = g_next++;
        if (i >= g_total_mb) break;
        long off_mb = g_stride > 0 ? (i * g_stride) % (g_file_mb - 1) : i;
        if (pread(fd, buf, 1 << 20, (off_t) off_mb << 20) != (1 << 20)) break;
        g_done_mb++;
    }
    free(buf); close(fd);
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <file> <seq|stride> <mb> <threads>\n", argv[0]); return 2; }
    g_path = argv[1];
    int stride = strcmp(argv[2], "stride") == 0;
    g_total_mb = atol(argv[3]);
    int threads = atoi(argv[4]);
    int fd = open(g_path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    off_t sz = lseek(fd, 0, SEEK_END);
    close(fd);
    g_file_mb = (long) (sz >> 20);
    g_stride = stride ? 37 : 0;   // 37 MiB apart: the profile's pairs are scattered
    g_next = 0; g_done_mb = 0; g_open_err = 0;
    pthread_t th[16];
    double t0 = now_s();
    for (int i = 0; i < threads; ++i) pthread_create(&th[i], NULL, worker, NULL);
    for (int i = 0; i < threads; ++i) pthread_join(th[i], NULL);
    double dt = now_s() - t0;
    if (g_open_err) { fprintf(stderr, "a worker failed to open\n"); return 1; }
    printf("%s %s %ld MiB x %d threads: %.2f s, %.0f MB/s\n", g_path, argv[2], (long) g_done_mb, threads, dt,
           (double) g_done_mb / dt);
    return 0;
}
