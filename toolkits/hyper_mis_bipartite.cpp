#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "core/graph.hpp"

namespace hyper_mis_bipartite_corrected {
  constexpr int inverse_prob = 3;

  // copied from ligra/utils.cpp
  inline ulong hashInt(ulong a) {
    a = (a+0x7ed55d166bef7a1d) + (a<<12);
    a = (a^0xc761c23c510fa2dd) ^ (a>>9);
    a = (a+0x165667b183a9c0e1) + (a<<59);
    a = (a+0xd3a2646cab3487e3) ^ (a<<49);
    a = (a+0xfd7046c5ef9ab54c) + (a<<3);
    a = (a^0xb55a4f090dd4a67b) ^ (a>>32);
    return a;
  }

  auto random_sample(uint * flags, VertexId offset, uint round) {
    return [&](VertexId vtx) {
      ulong hash = hashInt((ulong)(vtx+offset));
      if (hash % inverse_prob == 0) {
        flags[vtx] = round;
      }
      return 1;
    };
  }

  auto degrees_reset(VertexId * round_degrees) {
    return [&](VertexId vtx) {
      round_degrees[vtx] = 0;
      return 1;
    };
  }

  namespace count_neighbors {
    auto sparse_signal(Graph<Empty> * graph, uint * flags, uint round) {
      return [&](VertexId src) {
        if (flags[src] == round)
          graph->emit(src, 0);
      };
    }
    auto sparse_slot(VertexId * round_degrees) {
      return [&](VertexId src, VertexId msg, VertexAdjList<Empty> outgoing_adj) {
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;
          write_add(&round_degrees[dst], (VertexId)1);
        }
        return 1;
      };
    }
    auto dense_signal(Graph<Empty> * graph, VertexSubset * active_v, uint * flags, uint round) {
      return [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
        VertexId sum = 0;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          if (active_v->get_bit(src) && flags[src] == round)
            sum++;
        }
        if (sum > 0)
          graph->emit(dst, sum);
      };
    }
    auto dense_slot(VertexId * round_degrees) {
      return [&](VertexId dst, VertexId msg) {
        write_add(&round_degrees[dst], msg);
        return 1;
      };
    }
  } // namespace count_neighbors

  auto check_independence(VertexId * round_degrees, VertexId * total_degrees, VertexSubset * full_edges) {
    return [&](VertexId hid) {
      if (round_degrees[hid] == total_degrees[hid]) {
        full_edges->set_bit(hid);
      }
      return 1;
    };
  }

  namespace reset_neighbors {
    auto sparse_signal(Graph<Empty> * graph) {
      return [&](VertexId src) {
        graph->emit(src, Empty());
      };
    }
    auto sparse_slot(uint * flags, uint round, VertexSubset * remaining_v) {
      return [&](VertexId src, Empty msg, VertexAdjList<Empty> outgoing_adj) {
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;
          if (remaining_v->get_bit(dst))
            cas(&flags[dst], round, 0u);
        }
        return 1;
      };
    }
    auto dense_signal(Graph<Empty> * graph, VertexSubset * full_edges, VertexSubset * remaining_v) {
      return [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
        if (!remaining_v->get_bit(dst)) return;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          if (full_edges->get_bit(src)) {
            graph->emit(dst, Empty());
            return;
          }
        }
        
      };
    }
    auto dense_slot(uint * flags, uint round) {
      return [&](VertexId dst, Empty msg) {
        if (flags[dst] == round) {
          flags[dst] = 0;
          return 1;
        }
        return 0;
      };
    }
  } // namespace reset_neighbors

  auto filter_hyperedges(VertexId * total_degrees, VertexSubset * active_h, VertexSubset * size_one_edges) {
    return [&](VertexId src) {
        if (total_degrees[src] <= 1) {
          active_h->reset_bit(src);
          if (total_degrees[src] == 1) {
            size_one_edges->set_bit(src);
          }
        }
        return 1;
      };
  }

  namespace exclude_lonely_vertices {
    auto sparse_signal(Graph<Empty> * graph) {
      return [&](VertexId src) {
        graph->emit(src, Empty());
      };
    }
    auto sparse_slot(uint * flags, VertexSubset * remaining_v) {
      return [&](VertexId src, Empty msg, VertexAdjList<Empty> outgoing_adj) {
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;
          if (remaining_v->get_bit(dst)) {
            cas(&flags[dst], 0u, 1u);
          }
        }
        return 1;
      };
    }
    auto dense_signal(Graph<Empty> * graph, VertexSubset * size_one_edges, VertexSubset * remaining_v) {
      return [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
        if (!remaining_v->get_bit(dst)) return;
        bool need_update = false;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          if (size_one_edges->get_bit(src)) {
            graph->emit(dst, Empty());
            return;
          }
        }
      };
    }
    auto dense_slot(uint * flags) {
      return [&](VertexId dst, Empty msg) {
        cas(&flags[dst], 0u, 1u);
        return 1;
      };
    }
  } // namespace exclude_lonely_vertices

  auto filter_vertices(uint * flags, VertexSubset * active_v, VertexSubset * remaining_v) {
    return [&](VertexId vtx) {
      if (flags[vtx] != 0 || !remaining_v->get_bit(vtx)) {
        active_v->reset_bit(vtx);
        return 0;
      }
      return 1;
    };
  }
} // namespace hyper_mis_bipartite


void compute(Graph<Empty> * graph, VertexId hyper_vertices) {
  using namespace hyper_mis_bipartite;

  double exec_time = 0;
  exec_time -= get_time();

  uint * flags = graph->alloc_vertex_array<uint>(); // undecided = 0, out = 1, in = anything else
  VertexId * total_degrees = graph->alloc_vertex_array<VertexId>();
  VertexId * round_degrees = graph->alloc_vertex_array<VertexId>();
  VertexSubset * active_v = graph->alloc_vertex_subset();
  VertexSubset * active_h = graph->alloc_vertex_subset();
  VertexSubset * full_edges = graph->alloc_vertex_subset();
  VertexSubset * size_one_edges = graph->alloc_vertex_subset();
  VertexSubset * remaining_v = graph->alloc_vertex_subset();
  graph->fill_vertex_array(flags, (uint)0);
  active_v->fill();
  active_h->fill();
  remaining_v->fill();
  graph->process_vertices<int>(
    [&](VertexId vtx) {
      if (vtx < hyper_vertices) {
        active_h->reset_bit(vtx);
      } else {
        active_v->reset_bit(vtx);
        remaining_v->reset_bit(vtx);
      }
      return 0;
    },
    active_v
  );
  
  graph->process_vertices<int>(
    [&](VertexId htx) {
      total_degrees[htx] = graph->out_degree[htx];
      return 0;
    },
    active_h
  );

  VertexId active_vertex_num = hyper_vertices;
  VertexId rand_offset = 0;
  uint round=2;
  for (;active_vertex_num>0;round++) {
    // random sample
    rand_offset += graph->process_vertices<VertexId>(
      random_sample(flags, rand_offset, round),
      active_v
    );

    graph->process_vertices<VertexId>(
      [&](VertexId vtx) {
        return (flags[vtx] == round);
      },
      active_v
    );

    // reset round_degrees
    graph->process_vertices<VertexId>(
      degrees_reset(round_degrees),
      active_h
    );

    // count neighbors (assuming graph is undirected)
    graph->process_edges<int, VertexId>(
      count_neighbors::sparse_signal(graph, flags, round),
      count_neighbors::sparse_slot(round_degrees),
      count_neighbors::dense_signal(graph, active_v, flags, round),
      count_neighbors::dense_slot(round_degrees),
      active_v
    );

    // check independence
    full_edges->clear();
    graph->process_vertices<VertexId>(
      check_independence(round_degrees, total_degrees, full_edges),
      active_h
    );

    // reset neighbors
    graph->process_edges<int, Empty>(
      reset_neighbors::sparse_signal(graph),
      reset_neighbors::sparse_slot(flags, round, remaining_v),
      reset_neighbors::dense_signal(graph, full_edges, remaining_v),
      reset_neighbors::dense_slot(flags, round),
      full_edges,
      remaining_v
    );

    // delete vertices
    graph->filter_from(
      [&](VertexId vid) { return flags[vid] < 2; },
      remaining_v,
      active_h,
      total_degrees
    );

    // filter hyperedges
    size_one_edges->clear();
    graph->process_vertices<int>(
      filter_hyperedges(total_degrees, active_h, size_one_edges),
      active_h
    );
    graph->process_edges<int, Empty>(
      exclude_lonely_vertices::sparse_signal(graph),
      exclude_lonely_vertices::sparse_slot(flags, remaining_v),
      exclude_lonely_vertices::dense_signal(graph, size_one_edges, remaining_v),
      exclude_lonely_vertices::dense_slot(flags),
      size_one_edges,
      remaining_v
    );

    // filter_vertices
    active_vertex_num = graph->process_vertices<VertexId>(
      filter_vertices(flags, active_v, remaining_v),
      active_v
    );

    // std::cerr << round << ' ' << active_vertex_num << std::endl;
  }

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  graph->gather_vertex_array(flags, 0);
  if (graph->partition_id==0) {
    VertexId set_size = 0;
    for (VertexId i=0; i<hyper_vertices; i++) {
      if (flags[i] > 1) {
        // if (set_size < 10) std::cerr << i << std::endl;
        set_size++;
      }
    }
    printf("set_size=%ld, round=%ld\n", set_size, round);
  }

  graph->dealloc_vertex_array(flags);
  graph->dealloc_vertex_array(total_degrees);
  graph->dealloc_vertex_array(round_degrees);

  delete active_v;
  delete active_h;
  delete full_edges;
  delete size_one_edges;
  delete remaining_v;
}


int main(int argc, char ** argv) {
  MPI_Instance mpi(&argc, &argv);

  if (argc<4) {
    printf("hyper_mis_bipartite [file] [vertices] [hyperedges]\n");
    exit(-1);
  }

  VertexId vertices = std::atoi(argv[2]);
  VertexId hyperedges = std::atoi(argv[3]);

  Graph<Empty> * graph;
  graph = new Graph<Empty>();
  
  graph->load_directed(argv[1], vertices + hyperedges);
  compute(graph, vertices);
  for (int run=0;run<5;run++) {
    compute(graph, vertices);
  }

  delete graph;
  return 0;
}