#include <stdio.h>
#include <stdlib.h>

#include "core/graph.hpp"

namespace hyper_kcore_bipartite
{
  auto filter_vertices(VertexId k, VertexId * degrees, VertexSubset * to_remove, VertexSubset * active_v) {
    return [&](VertexId vid) {
      if (degrees[vid] < k) {
        degrees[vid] = k-1;
        to_remove->set_bit(vid);
        active_v->reset_bit(vid);
        return 1;
      }
      return 0;
    };
  }

  namespace remove_hyperedges
  {
    auto sparse_signal(Graph<Empty> * graph) {
      return [&](VertexId src){
        graph->emit(src, Empty());
      };
    }
    auto sparse_slot(uint * flags, VertexSubset * active_h) {
      return [&](VertexId src, Empty msg, VertexAdjList<Empty> outgoing_adj){
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;
          if (flags[dst] == 0) {
            cas(&flags[dst], 0u, 1u);
            active_h->set_bit(dst);
          }
        }
        return 1;
      };
    }
    auto dense_signal(Graph<Empty> * graph, VertexSubset * to_remove) {
      return [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          if (to_remove->get_bit(src)) {
            graph->emit(dst, Empty());
            return;
          }
        }
      };
    }
    auto dense_slot(uint * flags, VertexSubset * active_h) {
      return [&](VertexId dst, Empty msg) {
        if (flags[dst] == 0) {
          flags[dst] = 1;
          active_h->set_bit(dst);
        }
        return 0;
      };
    }
  } // namespace remove_hyperedges

  namespace update_degrees
  {
    auto sparse_signal(Graph<Empty> * graph) {
      return [&](VertexId src){
        graph->emit(src, 1);
      };
    }
    auto sparse_slot(long k, VertexId * degrees) {
      return [&](VertexId src, VertexId msg, VertexAdjList<Empty> outgoing_adj){
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;
          if (degrees[dst] >= k) {
            write_sub(&degrees[dst], msg);
          }
        }
        return 1;
      };
    }
    auto dense_signal(Graph<Empty> * graph, VertexSubset * active_h) {
      return [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
        VertexId msg = 0;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          if (active_h->get_bit(src)) {
            msg++;
          }
        }
        if (msg > 0) {
          graph->emit(dst, msg);
        }
      };
    }
    auto dense_slot(long k, VertexId * degrees) {
      return [&](VertexId dst, VertexId msg) {
        if (degrees[dst] >= k) {
          degrees[dst] -= msg;
        }
        return 0;
      };
    }
  } // namespace update_degrees
  
  
} // namespace hyper_kcore_bipartite


void compute(Graph<Empty> * graph, VertexId hyper_vertices, VertexId root) {
  using namespace hyper_kcore_bipartite;

  double exec_time = 0;
  exec_time -= get_time();

  VertexSubset * active_v = graph->alloc_vertex_subset();
  VertexSubset * active_h = graph->alloc_vertex_subset();
  VertexSubset * to_remove = graph->alloc_vertex_subset();
  active_v->fill();
  graph->process_vertices<int>(
    [&](VertexId vtx) {
      if (vtx >= hyper_vertices) {
        active_v->reset_bit(vtx);
      }
      return 0;
    },
    active_v
  );

  VertexId * degrees = graph->alloc_vertex_array<VertexId>(); //v array
  uint32_t * flags = graph->alloc_vertex_array<uint32_t>(); // h array
  graph->process_vertices<int>(
    [&](VertexId vid) {
      degrees[vid] = graph->out_degree[vid];
      return 0;
    },
    active_v
  );
  graph->fill_vertex_array(flags, 0u);

  long k = 1;
  VertexId active_vertices = hyper_vertices;

  for (; ; k++) {
    while (true) {
      // deg_lessthen_k => deg_atleast_k
      to_remove->clear();
      VertexId remove_target_num = graph->process_vertices<VertexId>(
        filter_vertices(k, degrees, to_remove, active_v),
        active_v
      );
      active_vertices -= remove_target_num;

      if (remove_target_num == 0) {
        break;
      }

      // remove hyperedges
      active_h->clear();
      graph->process_edges<int, Empty>(
        remove_hyperedges::sparse_signal(graph),
        remove_hyperedges::sparse_slot(flags, active_h),
        remove_hyperedges::dense_signal(graph, to_remove),
        remove_hyperedges::dense_slot(flags, active_h),
        to_remove
      );

      // update degrees
      graph->process_edges<int, VertexId>(
        update_degrees::sparse_signal(graph),
        update_degrees::sparse_slot(k, degrees),
        update_degrees::dense_signal(graph, active_h),
        update_degrees::dense_slot(k, degrees),
        active_h
      );
    }
    if (active_vertices == 0) {
      break;
    }
  }

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
    printf("largest_core=%ld\n", k - 1);
  }

  graph->dealloc_vertex_array(degrees);
  graph->dealloc_vertex_array(flags);
  delete active_v;
  delete active_h;
  delete to_remove;
}

int main(int argc, char ** argv) {
  MPI_Instance mpi(&argc, &argv);

  if (argc<4) {
    printf("hyper_kcore_bipartite [file] [vertices] [hyperedges]\n");
    exit(-1);
  }

  Graph<Empty> * graph;
  graph = new Graph<Empty>();

  VertexId vertices = std::atoi(argv[2]);
  VertexId hyperedges = std::atoi(argv[3]);

  VertexId root = (argc == 4 ? 0 : std::atoi(argv[4]));
  graph->load_directed(argv[1], vertices + hyperedges);

  compute(graph, vertices, root);
  for (int run=0;run<5;run++) {
    compute(graph, vertices, root);
  }

  delete graph;
  return 0;
}
