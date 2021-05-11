#include <stdio.h>
#include <stdlib.h>

#include "core/hypergraph.hpp"


// sample command
// toolkits/hyper_bpath ../datasets/com-orkut.bin 2322299 15301901

using ParentId = int32_t;
constexpr ParentId PAR_MAX = INT32_MAX;

namespace hyper_bpath
{
  template<typename SrcId>
  struct HyperBPathMessage {
    ParentId count;
    SrcId first;
    SrcId last;
  };

  template<typename SrcId>
  auto sparse_signal(Hypergraph<Empty> * graph) {
    return [&](SrcId src){
      graph->emit(src, HyperBPathMessage<SrcId>());
    };
  }

  template<typename SrcId, typename DstId, bool (*updateDst)(SrcId, DstId, ParentId*), typename DstSubset>
  auto sparse_slot(Hypergraph<Empty> * graph, ParentId * parent_dst, DstSubset * active_dst) {
    return [&](SrcId src, HyperBPathMessage<SrcId> msg, VertexAdjList<Empty> outgoing_adj){
      SrcId activated = 0;
      for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
        DstId dst = ptr->neighbour;
        if (updateDst(src, dst, parent_dst)) {
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
      HyperBPathMessage<SrcId> msg{0, UINT32_MAX, UINT32_MAX};
      for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
        SrcId src = ptr->neighbour;
        if (active_src->get_bit(src)) {
          msg.count++;
          if (msg.first == UINT32_MAX) {
            msg.first = src;
          }
          msg.last = src;
        }
      }
      if (msg.count > 0) {
        graph->emit(dst, msg);
      }
    };
  }

  namespace dense_slot {
    auto from_v(Hypergraph<Empty> * graph, ParentId * parent_h, HyperedgeSubset * active_h) {
      return [&](HyperedgeId dst, HyperBPathMessage<VertexId> msg) {
        if (parent_h[dst]+ msg.count == 0) {
          parent_h[dst] = msg.last;
          active_h->set_bit(dst);
          return 1;
        }
        return 0;
      };
    }
    auto from_h(Hypergraph<Empty> * graph, ParentId * parent_v, VertexSubset * active_v) {
      return [&](VertexId dst, HyperBPathMessage<HyperedgeId> msg) {
        if (parent_v[dst] == PAR_MAX) {
          parent_v[dst] = msg.first;
          active_v->set_bit(dst);
          return 1;
        }
        return 0;
      };
    }
  }

  // try updating dst's parent
  bool updateV(HyperedgeId src, VertexId dst, ParentId * parent_v) {
    return cas(&parent_v[dst], PAR_MAX, ParentId(src));
  }
  bool updateH(VertexId src, HyperedgeId dst, ParentId * parent_h) {
    if (parent_h[dst] < 0 && write_add(&parent_h[dst], 1) == 0) {
      parent_h[dst] = src;
      return 1;
    }
    return 0;
  }
} // namespace hyper_bpath


void compute(Hypergraph<Empty> * graph, VertexId root) {
  using namespace hyper_bpath;
  
  double exec_time = 0;
  exec_time -= get_time();

  VertexSubset * active_v = graph->alloc_vertex_subset();
  HyperedgeSubset * active_h = graph->alloc_hyperedge_subset();
  active_v->clear();
  active_v->set_bit(root);
  active_h->fill();

  VertexId active_vertices = 1;

  ParentId * parent_v = graph->alloc_vertex_array<ParentId>();
  ParentId * parent_h = graph->alloc_hyperedge_array<ParentId>();
  graph->fill_vertex_array(parent_v, PAR_MAX);
  graph->process_hyperedges<HyperedgeId>(
    [&](HyperedgeId hid){
      parent_h[hid] = -ParentId(graph->in_degree_h[hid]);
      return 1;
    },
    active_h
  );
  parent_v[root] = root;

  for (int i_i=0;active_vertices>0;i_i++) {
    if (graph->partition_id==0) {
      printf("active(%d)>=%u\n", i_i, active_vertices);
    }
    active_h->clear();
    graph->prop_from_vertices<VertexId, HyperBPathMessage<VertexId>>(
      sparse_signal<VertexId>(graph),
      sparse_slot<VertexId, HyperedgeId, updateH>(graph, parent_h, active_h),
      dense_signal<VertexId, HyperedgeId>(graph, active_v),
      dense_slot::from_v(graph, parent_h, active_h),
      active_v
    );

    active_v->clear();
    active_vertices = graph->prop_from_hyperedges<HyperedgeId, HyperBPathMessage<HyperedgeId>>(
      sparse_signal<HyperedgeId>(graph),
      sparse_slot<HyperedgeId, VertexId, updateV>(graph, parent_v, active_v),
      dense_signal<HyperedgeId, VertexId>(graph, active_h),
      dense_slot::from_h(graph, parent_v, active_v),
      active_h
    );
  }

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }
  
  graph->dealloc_vertex_array(parent_v);
  graph->dealloc_hyperedge_array(parent_h);
  delete active_v;
  delete active_h;
}

int main(int argc, char ** argv) {
  MPI_Instance mpi(&argc, &argv);

  if (argc<4) {
    printf("hyper_bpath [file] [vertices] [hyperedges] ([root])\n");
    exit(-1);
  }

  Hypergraph<Empty> * graph;
  graph = new Hypergraph<Empty>();
  graph->load_undirected_from_directed(argv[1], std::atoi(argv[2]), std::atoi(argv[3]));

  VertexId root = (argc == 4 ? 0 : std::atoi(argv[4]));

  compute(graph, root);
  for (int run=0;run<5;run++) {
    compute(graph, root);
  }

  delete graph;
  return 0;
}
