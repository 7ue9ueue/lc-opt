// Runs one program the way the judge does: stdin and stdout are files, and the time
// covers the whole process from fork to exit.
// Usage: runner TIME_LIMIT_SEC INPUT OUTPUT PROGRAM
// Prints: VERDICT WALL_MS PEAK_RSS_KB, where VERDICT is OK, RE or TLE.
#define _GNU_SOURCE
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static pid_t child;

static void kill_child(int sig) {
    (void)sig;
    kill(child, SIGKILL);
}

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: runner TIME_LIMIT_SEC INPUT OUTPUT PROGRAM\n");
        return 2;
    }
    double limit_ms = atof(argv[1]) * 1e3;

    // Let a slow program run to twice the limit so the report shows how slow it is.
    double kill_ms = 2 * limit_ms;
    struct itimerval timer = {0};
    timer.it_value.tv_sec = (time_t)(kill_ms / 1e3);
    timer.it_value.tv_usec = (suseconds_t)((kill_ms - timer.it_value.tv_sec * 1e3) * 1e3);
    signal(SIGALRM, kill_child);

    double start = now_ms();
    child = fork();
    if (child == 0) {
        int in = open(argv[2], O_RDONLY);
        int out = open(argv[3], O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (in < 0 || out < 0) _exit(127);
        dup2(in, 0);
        dup2(out, 1);
        execl(argv[4], argv[4], (char *)NULL);
        _exit(127);
    }
    setitimer(ITIMER_REAL, &timer, NULL);

    int status;
    struct rusage usage;
    while (wait4(child, &status, 0, &usage) < 0) {
    }
    double wall_ms = now_ms() - start;

    const char *verdict = "OK";
    if (wall_ms > limit_ms) verdict = "TLE";
    else if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) verdict = "RE";
    printf("%s %.3f %ld\n", verdict, wall_ms, usage.ru_maxrss);
    return 0;
}
