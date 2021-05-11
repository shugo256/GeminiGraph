#include <stdio.h>
#include <stdlib.h>

#include "core/hypergraph.hpp"

// sample command
// toolkits/hyper_cc ../datasets/com-orkut.bin 2322299 15301901

using Label = VertexId;

namespace hyper_cc {

  namespace update {
    template<typename SrcId>
    auto sparse_signal(Hypergraph<Empty> * graph, Label * label_src) {
      return [&](SrcId src){
        graph->emit(src, label_src[src]);
      };
    }

    template<typename SrcId, typename DstId, typename DstSubset>
    auto sparse_slot(Hypergraph<Empty> * graph, Label * label_dst, DstSubset * active_dst) {
      return [&](SrcId src, Label msg, VertexAdjList<Empty> outgoing_adj){
        SrcId activated = 0;
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          DstId dst = ptr->neighbour;
          if (msg < label_dst[dst]) {
            write_min(&label_dst[dst], msg);
            active_dst->set_bit(dst);
            activated += 1;
          }
        }
        return activated;
      };
    }

    template<typename SrcId, typename DstId>
    auto dense_signal(Hypergraph<Empty> * graph, Label * label_src) {
      return [&](DstId dst, VertexAdjList<Empty> incoming_adj) {
        Label msg = UINT32_MAX;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          SrcId src = ptr->neighbour;
          if (label_src[src] < msg) {
            msg = label_src[src];
          }
        }
        if (msg < UINT32_MAX) {
          graph->emit(dst, msg);
        }
      };
    }

    template<typename DstId, typename DstSubset>
    auto dense_slot(Hypergraph<Empty> * graph, VertexId * label_dst, DstSubset * active_dst) {
      return [&](DstId dst, Label msg) {
        if (msg < label_dst[dst]) {
          write_min(&label_dst[dst], msg);
          active_dst->set_bit(dst);
          return 1;
        }
        return 0;
      };
    }
  } // namespace update
} // hyper_cc

void compute(Hypergraph<Empty> * graph) {
  using namespace hyper_cc;

  double exec_time = 0;
  exec_time -= get_time();

  Label * label_v = graph->alloc_vertex_array<Label>();
  Label * label_h = graph->alloc_hyperedge_array<Label>();
  VertexSubset * active_v = graph->alloc_vertex_subset();
  HyperedgeSubset * active_h = graph->alloc_hyperedge_subset();
  graph->fill_hyperedge_array(label_h, UINT32_MAX);
  active_v->fill();

  VertexId active_vertices = graph->process_vertices<VertexId>(
    [&](VertexId vtx){
      label_v[vtx] = vtx;
      return 1;
    },
    active_v
  );

  for (int i_i=0;active_vertices>0;i_i++) {
    if (graph->partition_id==0) {
      printf("active(%d)>=%u\n", i_i, active_vertices);
    }
    active_h->clear();
    graph->prop_from_vertices<HyperedgeId,VertexId>(
      update::sparse_signal<VertexId>(graph, label_v),
      update::sparse_slot<VertexId, HyperedgeId>(graph, label_h, active_h),
      update::dense_signal<VertexId, HyperedgeId>(graph, label_v),
      update::dense_slot<HyperedgeId>(graph, label_h, active_h),
      active_v
    );

    active_v->clear();
    graph->prop_from_hyperedges<VertexId,VertexId>(
      update::sparse_signal<HyperedgeId>(graph, label_h),
      update::sparse_slot<HyperedgeId, VertexId>(graph, label_v, active_v),
      update::dense_signal<HyperedgeId, VertexId>(graph, label_h),
      update::dense_slot<VertexId>(graph, label_v, active_v),
      active_h
    );

    active_vertices = graph->process_vertices<int>(
      [&](VertexId vid) { return 1; },
      active_v
    );
  }

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  graph->gather_vertex_array(label_v, 0);
  if (graph->partition_id==0) {
    VertexId * count = graph->alloc_vertex_array<VertexId>();
    graph->fill_vertex_array(count, 0u);
    for (VertexId v_i=0;v_i<graph->vertices;v_i++) {
      count[label_v[v_i]] += 1;
    }
    VertexId components = 0;
    for (VertexId v_i=0;v_i<graph->vertices;v_i++) {
      if (count[v_i] > 0) {
        components += 1;
      }
    }
    printf("components = %u\n", components);
  }
  
  graph->dealloc_vertex_array(label_v);
  graph->dealloc_hyperedge_array(label_h);
  delete active_v;
  delete active_h;
}

int main(int argc, char ** argv) {
  MPI_Instance mpi(&argc, &argv);

  if (argc<4) {
    printf("cc [file] [vertices] [hyperedges]\n");
    exit(-1);
  }

  Hypergraph<Empty> * graph;
  graph = new Hypergraph<Empty>();
  graph->load_undirected_from_directed(argv[1], std::atoi(argv[2]), std::atoi(argv[3]));

  compute(graph);
  for (int run=0;run<5;run++) {
    compute(graph);
  }

  delete graph;
  return 0;
}
