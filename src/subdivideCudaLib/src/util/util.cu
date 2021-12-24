#include <stdlib.h>
#include <stdio.h>

#include "util.cuh"

// Courtesy of the Advanced Parallel Computing Course

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

void printFloatArr(float* arr, int size) {
    if(size == 0) {
        return;
    }
    for(int i = 0; i < size - 1; i++) {
        printf("%lf, ", arr[i]);
    }
    printf("%lf\n\n", arr[size - 1]);
}

void printIntArr(int* arr, int size) {
    if(size == 0) {
        return;
    }
    for(int i = 0; i < size - 1; i++) {
        printf("%d, ", arr[i]);
    }
    printf("%d\n\n", arr[size - 1]);
}