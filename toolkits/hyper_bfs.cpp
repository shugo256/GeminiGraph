#include <stdio.h>
#include <stdlib.h>
#include <cstdint>

#include "core/hypergraph.hpp"

// sample command
// toolkits/hyper_bfs ../datasets/com-orkut.bin 2322299 15301901

namespace hyper_bfs
{
  template<typename SrcId>
  auto sparse_signal(Hypergraph<Empty> * graph) {
    return [&](SrcId src){
      graph->emit(src, src);
    };
  }

  template<typename SrcId, typename DstId, typename DstSubset>
  auto sparse_slot(Hypergraph<Empty> * graph, SrcId * parent_dst, DstSubset * active_dst) {
    return [&](SrcId src, SrcId msg, VertexAdjList<Empty> outgoing_adj){
      DstId activated = 0;
      for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
        DstId dst = ptr->neighbour;
        if (parent_dst[dst] == UINT32_MAX && cas(&parent_dst[dst], UINT32_MAX, src)) {
          active_dst->set_bit(dst);
          activated += 1;
        }
      }
      return activated;
    };
  }

  template<typename SrcId, typename DstId, typename SrcSubset>
  auto dense_signal(Hypergraph<Empty> * graph, SrcSubset * active_src) {
    return [&](DstId dst, VertexAdjList<Empty> incoming_adj) {
      for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
        SrcId src = ptr->neighbour;
        if (active_src->get_bit(src)) {
          graph->emit(dst, SrcId());
          break;
        }
      }
    };
  }

  template<typename SrcId, typename DstId, typename DstSubset>
  auto dense_slot(Hypergraph<Empty> * graph, SrcId * parent_dst, DstSubset * active_dst) {
    return [&](DstId dst, SrcId msg) {
      if (parent_dst[dst] == UINT32_MAX && cas(&parent_dst[dst], UINT32_MAX, msg)) {
        active_dst->set_bit(dst);
        return 1;
      }
      return 0;
    };
  }
} // namespace hyper_bfs


void compute(Hypergraph<Empty> * graph, VertexId root) {
  using namespace hyper_bfs;

  double exec_time = 0;
  exec_time -= get_time();

  HyperedgeId * parent_v = graph->alloc_vertex_array<HyperedgeId>();
  VertexId * parent_h = graph->alloc_hyperedge_array<VertexId>();
  graph->fill_vertex_array(parent_v, UINT32_MAX);
  graph->fill_hyperedge_array(parent_h, UINT32_MAX);
  parent_v[root] = root;

  VertexSubset * active_v = graph->alloc_vertex_subset();
  HyperedgeSubset * active_h = graph->alloc_hyperedge_subset();
  active_v->clear();
  active_v->set_bit(root);

  VertexId active_vertices = 1;
  
  for (int i_i=0;active_vertices>0;i_i++) {
    if (graph->partition_id==0) {
      printf("active(%d)>=%u\n", i_i, active_vertices);
    }
    active_h->clear();
    graph->prop_from_vertices<HyperedgeId, VertexId>(
      sparse_signal<VertexId>(graph),
      sparse_slot<VertexId, HyperedgeId>(graph, parent_h, active_h),
      dense_signal<VertexId, HyperedgeId>(graph, active_v),
      dense_slot<VertexId, HyperedgeId>(graph, parent_h, active_h),
      active_v
    );

    active_v->clear();
    active_vertices = graph->prop_from_hyperedges<VertexId, HyperedgeId>(
      sparse_signal<HyperedgeId>(graph),
      sparse_slot<HyperedgeId, VertexId>(graph, parent_v, active_v),
      dense_signal<HyperedgeId, VertexId>(graph, active_h),
      dense_slot<HyperedgeId, VertexId>(graph, parent_v, active_v),
      active_h
    );
  }

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  graph->gather_vertex_array(parent_v, 0);
  if (graph->partition_id==0) {
    VertexId found_vertices = 0;
    for (VertexId v_i=0;v_i<graph->vertices;v_i++) {
      if (parent_v[v_i] < UINT32_MAX) {
        found_vertices += 1;
      }
    }
    printf("found_vertices = %u\n", found_vertices);
  }

  graph->dealloc_vertex_array(parent_v);
  graph->dealloc_hyperedge_array(parent_h);
  delete active_v;
  delete active_h;
}

int main(int argc, char ** argv) {
  MPI_Instance mpi(&argc, &argv);

  if (argc<4) {
    printf("hyper_bfs [file] [vertices] [hyperedges] ([root])\n");
    exit(-1);
  }

  Hypergraph<Empty> * graph;
  graph = new Hypergraph<Empty>();
  graph->load_directed(argv[1], std::atoi(argv[2]), std::atoi(argv[3]));
  VertexId root = (argc == 4 ? 0 : std::atoi(argv[4]));

  compute(graph, root);
  for (int run=0;run<5;run++) {
    compute(graph, root);
  }

  delete graph;
  return 0;
}
