#include "list.cuh"

#include "stdlib.h"

List initEmptyList() {
    List list;
    list.size = 10;
    list.arr = (int*)malloc(list.size * sizeof(int));
    list.i = 0;
    return list;
}

void append(List* list, int item) {
    if(list->i == list->size) {
        list->size *= 2;
        list->arr = (int*)realloc(list->arr, list->size * sizeof(int));
    }
    list->arr[list->i] = item;
    list->i++;
}

int indexOf(List* list, int item) {
    for(int k = 0; k < list->i; k++) {
        if(list->arr[k] == item) {
            return k;
        }
    }
    return -1;
}

int listSize(List* list) {
    return list->i;
}