#pragma once

/*
A simple min-priority-queue implementation. Uses STL internally

This is for the most part an internal implementation detail of
drcal_traverse_sensor_links(), but could potentially be useful somewhere
on its own, so I'm exposing it here
 */

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    uint16_t idx_parent;
    bool done : 1;
    uint64_t cost : 47;
} node_t;
// I can't find a single static assertion invocation that works in both C++ and
// C. The below is ugly, but works
#ifdef __cplusplus
static_assert(sizeof(node_t) == 8, "node_t has expected size");
#else
_Static_assert(sizeof(node_t) == 8, "node_t has expected size");
#endif

typedef struct {
    uint16_t* buffer;  // each of these indexes an external node_t[] array,
                       // which contains the cost
    int size;
} drcal_heap_t;

bool drcal_heap_empty(drcal_heap_t* heap, node_t* nodes);
void drcal_heap_push(drcal_heap_t* heap, node_t* nodes, uint16_t x);
uint16_t drcal_heap_pop(drcal_heap_t* heap, node_t* nodes);
void drcal_heap_resort(drcal_heap_t* heap, node_t* nodes);
