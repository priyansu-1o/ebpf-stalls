#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/syscall.h>

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static uint64_t get_thread_cputime_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static uint64_t get_schedstat_wait_ns(void) {
    int fd = open("/proc/thread-self/schedstat", O_RDONLY);
    if (fd < 0) return 0;
    char buf[128];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return 0;
    buf[n] = '\0';
    unsigned long long cputime = 0, run_delay = 0, pcount = 0;
    if (sscanf(buf, "%llu %llu %llu", &cputime, &run_delay, &pcount) >= 2) {
        return (uint64_t)run_delay;
    }
    return 0;
}

static pid_t get_tid(void) {
    return (pid_t)syscall(SYS_gettid);
}

static const char *get_run_id(void) {
    const char *r = getenv("RUN_ID");
    return r ? r : "default_run";
}

/* 1. CPU Stall: Pure arithmetic spin loop */
void run_cpu_stall(int op_id, FILE *truth_fp, uint64_t duration_ms) {
    pid_t pid = getpid();
    pid_t tid = get_tid();
    uint64_t cpu_start = get_thread_cputime_ns();
    uint64_t wait_start = get_schedstat_wait_ns();
    uint64_t start = get_time_ns();
    uint64_t target_end = start + (duration_ms * 1000000ULL);

    volatile uint64_t counter = 0;
    while (get_time_ns() < target_end) {
        counter++;
    }
    uint64_t end = get_time_ns();
    uint64_t cpu_end = get_thread_cputime_ns();
    uint64_t wait_end = get_schedstat_wait_ns();

    uint64_t kernel_cpu_ns = (cpu_end >= cpu_start) ? (cpu_end - cpu_start) : 0;
    uint64_t kernel_sched_wait_ns = (wait_end >= wait_start) ? (wait_end - wait_start) : 0;

    fprintf(truth_fp,
            "{\"run_id\":\"%s\",\"op_id\":%d,\"pid\":%d,\"tid\":%d,\"start_ns\":%llu,\"end_ns\":%llu,\"syscall\":\"none\",\"bytes\":0,\"kernel_cpu_ns\":%llu,\"kernel_sched_wait_ns\":%llu,\"true_label\":\"CPU_SYSCALL\",\"label\":\"CPU_SYSCALL\"}\n",
            get_run_id(), op_id, (int)pid, (int)tid, (unsigned long long)start, (unsigned long long)end,
            (unsigned long long)kernel_cpu_ns, (unsigned long long)kernel_sched_wait_ns);
    fflush(truth_fp);
}

/* 2. Cold Synchronous I/O: O_DIRECT read from disk scratch file */
void run_io_wait(int op_id, FILE *truth_fp, const char *filepath, size_t block_size) {
    pid_t pid = getpid();
    pid_t tid = get_tid();
    
    // O_DIRECT requires aligned memory buffer
    void *buf = NULL;
    if (posix_memalign(&buf, 4096, block_size) != 0) {
        perror("posix_memalign");
        return;
    }

    int fd = open(filepath, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open O_DIRECT");
        free(buf);
        return;
    }

    uint64_t cpu_start = get_thread_cputime_ns();
    uint64_t wait_start = get_schedstat_wait_ns();
    uint64_t start = get_time_ns();
    ssize_t bytes_read = read(fd, buf, block_size);
    uint64_t end = get_time_ns();
    uint64_t cpu_end = get_thread_cputime_ns();
    uint64_t wait_end = get_schedstat_wait_ns();

    close(fd);
    free(buf);

    uint64_t kernel_cpu_ns = (cpu_end >= cpu_start) ? (cpu_end - cpu_start) : 0;
    uint64_t kernel_sched_wait_ns = (wait_end >= wait_start) ? (wait_end - wait_start) : 0;

    if (bytes_read > 0) {
        fprintf(truth_fp,
                "{\"run_id\":\"%s\",\"op_id\":%d,\"pid\":%d,\"tid\":%d,\"start_ns\":%llu,\"end_ns\":%llu,\"syscall\":\"read\",\"bytes\":%zd,\"kernel_cpu_ns\":%llu,\"kernel_sched_wait_ns\":%llu,\"true_label\":\"IO_WAIT\",\"label\":\"IO_WAIT\"}\n",
                get_run_id(), op_id, (int)pid, (int)tid, (unsigned long long)start, (unsigned long long)end, bytes_read,
                (unsigned long long)kernel_cpu_ns, (unsigned long long)kernel_sched_wait_ns);
        fflush(truth_fp);
    }
}

/* 3. Buffered Cold Read: Page cache dropped before, readahead active */
void run_cold_buffered_read(int op_id, FILE *truth_fp, const char *filepath, size_t block_size) {
    pid_t pid = getpid();
    pid_t tid = get_tid();
    char *buf = malloc(block_size);
    if (!buf) {
        perror("malloc");
        return;
    }

    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        perror("open cold buffered file");
        free(buf);
        return;
    }

    uint64_t cpu_start = get_thread_cputime_ns();
    uint64_t wait_start = get_schedstat_wait_ns();
    uint64_t start = get_time_ns();
    ssize_t bytes_read = read(fd, buf, block_size);
    uint64_t end = get_time_ns();
    uint64_t cpu_end = get_thread_cputime_ns();
    uint64_t wait_end = get_schedstat_wait_ns();

    close(fd);
    free(buf);

    uint64_t kernel_cpu_ns = (cpu_end >= cpu_start) ? (cpu_end - cpu_start) : 0;
    uint64_t kernel_sched_wait_ns = (wait_end >= wait_start) ? (wait_end - wait_start) : 0;

    if (bytes_read > 0) {
        fprintf(truth_fp,
                "{\"run_id\":\"%s\",\"op_id\":%d,\"pid\":%d,\"tid\":%d,\"start_ns\":%llu,\"end_ns\":%llu,\"syscall\":\"read\",\"bytes\":%zd,\"kernel_cpu_ns\":%llu,\"kernel_sched_wait_ns\":%llu,\"true_label\":\"IO_WAIT\",\"label\":\"IO_WAIT\"}\n",
                get_run_id(), op_id, (int)pid, (int)tid, (unsigned long long)start, (unsigned long long)end, bytes_read,
                (unsigned long long)kernel_cpu_ns, (unsigned long long)kernel_sched_wait_ns);
        fflush(truth_fp);
    }
}

/* 4. Warm Cached I/O: Buffered read from page-cache resident file */
void run_warm_read(int op_id, FILE *truth_fp, const char *filepath, size_t block_size) {
    pid_t pid = getpid();
    pid_t tid = get_tid();
    char *buf = malloc(block_size);

    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        perror("open warm file");
        free(buf);
        return;
    }

    uint64_t cpu_start = get_thread_cputime_ns();
    uint64_t wait_start = get_schedstat_wait_ns();
    uint64_t start = get_time_ns();
    ssize_t bytes_read = read(fd, buf, block_size);
    uint64_t end = get_time_ns();
    uint64_t cpu_end = get_thread_cputime_ns();
    uint64_t wait_end = get_schedstat_wait_ns();

    close(fd);
    free(buf);

    uint64_t kernel_cpu_ns = (cpu_end >= cpu_start) ? (cpu_end - cpu_start) : 0;
    uint64_t kernel_sched_wait_ns = (wait_end >= wait_start) ? (wait_end - wait_start) : 0;

    if (bytes_read > 0) {
        fprintf(truth_fp,
                "{\"run_id\":\"%s\",\"op_id\":%d,\"pid\":%d,\"tid\":%d,\"start_ns\":%llu,\"end_ns\":%llu,\"syscall\":\"read\",\"bytes\":%zd,\"kernel_cpu_ns\":%llu,\"kernel_sched_wait_ns\":%llu,\"true_label\":\"WARM_READ\",\"label\":\"WARM_READ\"}\n",
                get_run_id(), op_id, (int)pid, (int)tid, (unsigned long long)start, (unsigned long long)end, bytes_read,
                (unsigned long long)kernel_cpu_ns, (unsigned long long)kernel_sched_wait_ns);
        fflush(truth_fp);
    }
}

/* 5. Buffered Write + fsync: Dirty page write with synchronous flush */
void run_buffered_write(int op_id, FILE *truth_fp, const char *filepath, size_t block_size) {
    pid_t pid = getpid();
    pid_t tid = get_tid();
    char *buf = malloc(block_size);
    if (!buf) {
        perror("malloc");
        return;
    }
    memset(buf, 0x5a, block_size);

    int fd = open(filepath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open for buffered write");
        free(buf);
        return;
    }

    uint64_t cpu_start = get_thread_cputime_ns();
    uint64_t wait_start = get_schedstat_wait_ns();
    uint64_t start = get_time_ns();
    ssize_t written = write(fd, buf, block_size);
    int sync_ret = fsync(fd);
    uint64_t end = get_time_ns();
    uint64_t cpu_end = get_thread_cputime_ns();
    uint64_t wait_end = get_schedstat_wait_ns();

    close(fd);
    free(buf);

    uint64_t kernel_cpu_ns = (cpu_end >= cpu_start) ? (cpu_end - cpu_start) : 0;
    uint64_t kernel_sched_wait_ns = (wait_end >= wait_start) ? (wait_end - wait_start) : 0;

    if (written > 0 && sync_ret == 0) {
        fprintf(truth_fp,
                "{\"run_id\":\"%s\",\"op_id\":%d,\"pid\":%d,\"tid\":%d,\"start_ns\":%llu,\"end_ns\":%llu,\"syscall\":\"write_fsync\",\"bytes\":%zd,\"kernel_cpu_ns\":%llu,\"kernel_sched_wait_ns\":%llu,\"true_label\":\"IO_WAIT\",\"label\":\"IO_WAIT\"}\n",
                get_run_id(), op_id, (int)pid, (int)tid, (unsigned long long)start, (unsigned long long)end, written,
                (unsigned long long)kernel_cpu_ns, (unsigned long long)kernel_sched_wait_ns);
        fflush(truth_fp);
    }
}

/* 4. Scheduler Contention: Pinned threads competing on single core */
struct worker_arg {
    int core_id;
    int duration_ms;
    volatile int *stop;
};

void *spinner_thread(void *arg) {
    struct worker_arg *w = (struct worker_arg *)arg;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(w->core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    while (!(*w->stop)) {
        __asm__ __volatile__("" : : : "memory");
    }
    return NULL;
}

void run_sched_contend(int op_id, FILE *truth_fp, int num_competitors) {
    pid_t pid = getpid();
    pid_t tid = get_tid();
    int target_core = 0;

    // Pin main thread to core 0
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(target_core, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

    volatile int stop_workers = 0;
    pthread_t *threads = malloc(sizeof(pthread_t) * num_competitors);
    struct worker_arg warg = { .core_id = target_core, .duration_ms = 0, .stop = &stop_workers };

    for (int i = 0; i < num_competitors; i++) {
        pthread_create(&threads[i], NULL, spinner_thread, &warg);
    }

    // Give background spinners a brief moment to start hogging the core
    usleep(1000);

    uint64_t cpu_start = get_thread_cputime_ns();
    uint64_t wait_start = get_schedstat_wait_ns();
    uint64_t start = get_time_ns();
    // Yield repeatedly while competing with pinned background spinners
    for (int y = 0; y < 100; y++) {
        sched_yield();
        volatile uint64_t spin = 0;
        for (int k = 0; k < 10000; k++) {
            spin += k;
        }
    }
    uint64_t end = get_time_ns();
    uint64_t cpu_end = get_thread_cputime_ns();
    uint64_t wait_end = get_schedstat_wait_ns();

    stop_workers = 1;
    for (int i = 0; i < num_competitors; i++) {
        pthread_join(threads[i], NULL);
    }
    free(threads);

    uint64_t kernel_cpu_ns = (cpu_end >= cpu_start) ? (cpu_end - cpu_start) : 0;
    uint64_t kernel_sched_wait_ns = (wait_end >= wait_start) ? (wait_end - wait_start) : 0;

    fprintf(truth_fp,
            "{\"run_id\":\"%s\",\"op_id\":%d,\"pid\":%d,\"tid\":%d,\"start_ns\":%llu,\"end_ns\":%llu,\"syscall\":\"sched_yield\",\"bytes\":0,\"kernel_cpu_ns\":%llu,\"kernel_sched_wait_ns\":%llu,\"true_label\":\"SCHED_WAIT\",\"label\":\"SCHED_WAIT\"}\n",
            get_run_id(), op_id, (int)pid, (int)tid, (unsigned long long)start, (unsigned long long)end,
            (unsigned long long)kernel_cpu_ns, (unsigned long long)kernel_sched_wait_ns);
    fflush(truth_fp);
}

void run_mixed_contend(int op_id, FILE *truth_fp, const char *filepath, size_t block_size, int num_competitors) {
    pid_t pid = getpid();
    pid_t tid = get_tid();
    int target_core = 0;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(target_core, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

    volatile int stop_workers = 0;
    pthread_t *threads = malloc(sizeof(pthread_t) * num_competitors);
    struct worker_arg warg = { .core_id = target_core, .duration_ms = 0, .stop = &stop_workers };
    for (int i = 0; i < num_competitors; i++) {
        pthread_create(&threads[i], NULL, spinner_thread, &warg);
    }
    usleep(1000);

    void *buf = NULL;
    if (posix_memalign(&buf, 4096, block_size) != 0) {
        stop_workers = 1;
        for (int i = 0; i < num_competitors; i++) pthread_join(threads[i], NULL);
        free(threads);
        return;
    }

    int fd = open(filepath, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        free(buf);
        stop_workers = 1;
        for (int i = 0; i < num_competitors; i++) pthread_join(threads[i], NULL);
        free(threads);
        return;
    }

    uint64_t cpu_start = get_thread_cputime_ns();
    uint64_t wait_start = get_schedstat_wait_ns();
    uint64_t start = get_time_ns();
    ssize_t bytes_read = read(fd, buf, block_size);
    for (int y = 0; y < 50; y++) {
        sched_yield();
        volatile uint64_t spin = 0;
        for (int k = 0; k < 5000; k++) spin += k;
    }
    uint64_t end = get_time_ns();
    uint64_t cpu_end = get_thread_cputime_ns();
    uint64_t wait_end = get_schedstat_wait_ns();

    close(fd);
    free(buf);
    stop_workers = 1;
    for (int i = 0; i < num_competitors; i++) pthread_join(threads[i], NULL);
    free(threads);

    uint64_t kernel_cpu_ns = (cpu_end >= cpu_start) ? (cpu_end - cpu_start) : 0;
    uint64_t kernel_sched_wait_ns = (wait_end >= wait_start) ? (wait_end - wait_start) : 0;

    fprintf(truth_fp,
            "{\"run_id\":\"%s\",\"op_id\":%d,\"pid\":%d,\"tid\":%d,\"start_ns\":%llu,\"end_ns\":%llu,\"syscall\":\"read\",\"bytes\":%zd,\"kernel_cpu_ns\":%llu,\"kernel_sched_wait_ns\":%llu,\"true_label\":\"INDETERMINATE\",\"label\":\"INDETERMINATE\"}\n",
            get_run_id(), op_id, (int)pid, (int)tid, (unsigned long long)start, (unsigned long long)end, bytes_read > 0 ? bytes_read : 0,
            (unsigned long long)kernel_cpu_ns, (unsigned long long)kernel_sched_wait_ns);
    fflush(truth_fp);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <mode> <truth_file> <target_file_or_param> [block_size] [iterations]\n", argv[0]);
        fprintf(stderr, "Modes: cpu | io | cold_buffered | warm | buffered_write | sched | mixed\n");
        return 1;
    }

    char *mode = argv[1];
    char *truth_path = argv[2];
    char *param = argv[3];
    size_t bs = (argc >= 5) ? atol(argv[4]) : 1048576; // default 1MB
    int iters = (argc >= 6) ? atoi(argv[5]) : 1;
    if (iters < 1) iters = 1;

    FILE *fp = fopen(truth_path, "a");
    if (!fp) {
        perror("fopen truth_path");
        return 1;
    }

    for (int i = 1; i <= iters; i++) {
        if (strcmp(mode, "cpu") == 0) {
            int duration = atoi(param);
            run_cpu_stall(i, fp, duration);
        } else if (strcmp(mode, "io") == 0) {
            run_io_wait(i, fp, param, bs);
        } else if (strcmp(mode, "cold_buffered") == 0 || strcmp(mode, "buffered_read") == 0) {
            run_cold_buffered_read(i, fp, param, bs);
        } else if (strcmp(mode, "warm") == 0) {
            run_warm_read(i, fp, param, bs);
        } else if (strcmp(mode, "buffered_write") == 0 || strcmp(mode, "write") == 0) {
            run_buffered_write(i, fp, param, bs);
        } else if (strcmp(mode, "sched") == 0) {
            int competitors = atoi(param);
            run_sched_contend(i, fp, competitors);
        } else if (strcmp(mode, "mixed") == 0) {
            run_mixed_contend(i, fp, param, bs, 4);
        } else {
            fprintf(stderr, "Unknown mode: %s\n", mode);
            break;
        }
        if (i < iters) {
            usleep(10000); // 10ms gap between operations
        }
    }

    fclose(fp);
    return 0;
}
