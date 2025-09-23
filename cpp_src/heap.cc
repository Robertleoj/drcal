/*
  A simple min-priority-queue implementation. Uses STL internally
*/
#include <algorithm>
#include <functional>

extern "C" {
#include "heap.h"
}

struct Compare_nodes_greater {
    node_t* nodes;
    Compare_nodes_greater(
        node_t* _nodes
    )
        : nodes(_nodes) {}
    bool operator()(
        const uint16_t& idx_a,
        const uint16_t& idx_b
    ) const {
        return nodes[idx_a].cost > nodes[idx_b].cost;
    }
};

extern "C" bool drcal_heap_empty(
    drcal_heap_t* heap,
    node_t* nodes
) {
    return heap->size == 0;
}

extern "C" void drcal_heap_push(
    drcal_heap_t* heap,
    node_t* nodes,
    uint16_t x
) {
    heap->buffer[heap->size++] = x;
    std::push_heap(
        &heap->buffer[0],
        &heap->buffer[heap->size],
        Compare_nodes_greater(nodes)
    );
}

extern "C" uint16_t drcal_heap_pop(
    drcal_heap_t* heap,
    node_t* nodes
) {
    uint16_t x = heap->buffer[0];
    std::pop_heap(
        &heap->buffer[0],
        &heap->buffer[heap->size],
        Compare_nodes_greater(nodes)
    );
    heap->size--;
    return x;
}

extern "C" void drcal_heap_resort(
    drcal_heap_t* heap,
    node_t* nodes
) {
    std::make_heap(
        &heap->buffer[0],
        &heap->buffer[heap->size],
        Compare_nodes_greater(nodes)
    );
}
