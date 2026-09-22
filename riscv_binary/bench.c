#include <stdlib.h>
#include <stdio.h>

#define N (32*1024*1024)   /* 32MB, 8192 페이지 */

int main(void) {
    char *a = malloc(N);
    for (long i = 0; i < N; i += 4096) a[i] = 1;   /* 미리 매핑 */

    long sum = 0;
    for (long i = 0; i < 2000000; i++)
        sum += a[(i * 4099) % N];                  /* 무작위 페이지 접근 */

    printf("%ld\n", sum);
    return 0;
}
