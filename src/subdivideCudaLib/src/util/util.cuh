#ifndef UTILS_CUH
#define UTILS_CUH

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
void printFloatArr(float* arr, int size);
void printIntArr(int* arr, int size);

#define FATAL(msg, ...)                                                      \
  do {                                                                       \
    fprintf(stderr, "[%s:%d] " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    exit(-1);                                                                \
  } while (0)


#endif // UTILS_CUH