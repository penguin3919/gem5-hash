#include <stdio.h>
#include <stdlib.h>

// bench_seq.c
#define N (32*1024*1024)

int main(void) {
    char *a = malloc(N);
    for (long i = 0; i < N; i += 4096) a[i] = 1;

    long sum = 0;
    for (long i = 0; i < 2000000; i++)
        sum += a[(i * 4096) % N];   // 4096 단위 → 순차 페이지 접근

    printf("%ld\n", sum);
    return 0;
}
