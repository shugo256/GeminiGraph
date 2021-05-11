#include <stdio.h>
#include <stdlib.h>

#include "core/hypergraph.hpp"

typedef float Weight;

namespace hyper_sssp
{
  template<typename SrcId>
  auto sparse_signal(Hypergraph<Weight> * graph, Weight * distance_src) {
    return [&](SrcId src){
      graph->emit(src, distance_src[src]);
    };
  }

  template<typename SrcId, typename DstId, typename DstSubset>
  auto sparse_slot(Weight * distance_dst, DstSubset * active_dst) {
    return [&](SrcId src, Weight msg, VertexAdjList<Weight> outgoing_adj){
      DstId activated = 0;
      for (AdjUnit<Weight> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
        DstId dst = ptr->neighbour;
        Weight relax_dist = msg + ptr->edge_data;
        if (relax_dist < distance_dst[dst]) {
          if (write_min(&distance_dst[dst], relax_dist)) {
            active_dst->set_bit(dst);
            activated += 1;
          }
        }
      }
      return activated;
    };
  }

  template<typename SrcId, typename DstId, typename SrcSubset>
  auto dense_signal(Hypergraph<Weight> * graph, Weight * distance_src, SrcSubset * active_src) {
    return [&](DstId dst, VertexAdjList<Weight> incoming_adj) {
      Weight msg = 1e9;
      for (AdjUnit<Weight> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
        SrcId src = ptr->neighbour;
        if (active_src->get_bit(src)) {
          Weight relax_dist = distance_src[src] + ptr->edge_data;
          if (relax_dist < msg) {
            msg = relax_dist;
          }
        }
      }
      if (msg < 1e9) graph->emit(dst, msg);
    };
  }

  template<typename DstId, typename DstSubset>
  auto dense_slot(Weight * distance_dst, DstSubset * active_dst) {
    return [&](DstId dst, Weight msg) {
      if (msg < distance_dst[dst]) {
        write_min(&distance_dst[dst], msg);
        active_dst->set_bit(dst);
        return 1;
      }
      return 0;
    };
  }
} // namespace hyper_sssp


void compute(Hypergraph<Weight> * graph, VertexId root) {
  using namespace hyper_sssp;

  double exec_time = 0;
  exec_time -= get_time();

  Weight * distance_v = graph->alloc_vertex_array<Weight>();
  Weight * distance_h = graph->alloc_hyperedge_array<Weight>();
  VertexSubset * active_v = graph->alloc_vertex_subset();
  HyperedgeSubset * active_h = graph->alloc_hyperedge_subset();
  active_v->clear();
  active_v->set_bit(root);
  graph->fill_vertex_array(distance_v, (Weight)1e9);
  graph->fill_hyperedge_array(distance_h, (Weight)1e9);
  distance_v[root] = (Weight)0;
  VertexId active_vertices = 1;
  
  for (int i_i=0;active_vertices>0;i_i++) {
    if (graph->partition_id==0) {
      printf("active(%d)>=%u\n", i_i, active_vertices);
    }
    active_h->clear();
    graph->prop_from_vertices<HyperedgeId,Weight>(
      sparse_signal<VertexId>(graph, distance_v),
      sparse_slot<VertexId, HyperedgeId>(distance_h, active_h),
      dense_signal<VertexId, HyperedgeId>(graph, distance_v, active_v),
      dense_slot<HyperedgeId>(distance_h, active_h),
      active_v
    );


    active_v->clear();
    graph->prop_from_hyperedges<VertexId,Weight>(
      sparse_signal<HyperedgeId>(graph, distance_h),
      sparse_slot<HyperedgeId, VertexId>(distance_v, active_v),
      dense_signal<HyperedgeId, VertexId>(graph, distance_h, active_h),
      dense_slot<VertexId>(distance_v, active_v),
      active_h
    );

    active_vertices = graph->process_vertices<VertexId>(
      [&](VertexId vid) { return 1; },
      active_v
    );
  }

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  graph->gather_vertex_array(distance_v, 0);
  if (graph->partition_id==0) {
    VertexId max_v_i = root;
    for (VertexId v_i=0;v_i<graph->vertices;v_i++) {
      if (distance_v[v_i] < 1e9 && distance_v[v_i] > distance_v[max_v_i]) {
        max_v_i = v_i;
      }
    }
    printf("distance_v[%u]=%f\n", max_v_i, distance_v[max_v_i]);
  }

  graph->dealloc_vertex_array(distance_v);
  graph->dealloc_hyperedge_array(distance_h);
  delete active_v;
  delete active_h;
}

int main(int argc, char ** argv) {
  MPI_Instance mpi(&argc, &argv);

  if (argc<4) {
    printf("hyper_sssp [file] [vertices] [hyperedges] ([root])\n");
    exit(-1);
  }

  Hypergraph<Weight> * graph;
  graph = new Hypergraph<Weight>();
  graph->load_directed(argv[1], std::atoi(argv[2]), std::atoi(argv[3]));
  VertexId root = (argc == 4 ? 0 : std::atoi(argv[4]));

  compute(graph, root);
  for (int run=0;run<5;run++) {
    compute(graph, root);
  }

  delete graph;
  return 0;
}
