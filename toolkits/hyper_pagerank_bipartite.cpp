#include <stdio.h>
#include <stdlib.h>

#include "core/graph.hpp"

#include <math.h>

namespace bipartite_hyper_pagerank {
  constexpr double damping = 0.85;

  auto reset_rank(double * rank) {
    return [&](VertexId i) {
      rank[i] = 0;
      return 1;
    };
  }

  namespace prop_rank
  {
    auto sparse_signal(Graph<Empty> * graph, double * rank) {
      return [&](VertexId src){
        graph->emit(src, rank[src] / double(graph->out_degree[src]));
      };
    }

    auto sparse_slot(double * rank_dst) {
      return [&](VertexId src, double msg, VertexAdjList<Empty> outgoing_adj){
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;
          write_add(&rank_dst[dst], msg);
        }
        return 1;
      };
    }

    auto dense_signal(Graph<Empty> * graph, double * rank) {
      return [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
        double msg = 0;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          msg += rank[src] / double(graph->out_degree[src]);
        }
        if (msg > 0) graph->emit(dst, msg);
      };
    }

    auto dense_slot(double * rank_dst) {
      return [&](VertexId dst, double msg) {
        rank_dst[dst] += msg;
        return 0;
      };
    }
  } // namespace prop_rank

  auto update_vertex_rank(Graph<Empty> * graph, double * rank, VertexId hyper_vertices) {
    return [&](VertexId vid) {
      rank[vid] *= damping;
      rank[vid] += (1. - damping) * (1. / double(hyper_vertices));
      return 1;
    };
  }
} // namespace bipartite_hyper_pagerank


void compute(Graph<Empty> * graph, VertexId hyper_vertices, int iterations) {
  using namespace bipartite_hyper_pagerank;

  double exec_time = 0;
  exec_time -= get_time();

  VertexSubset * all = graph->alloc_vertex_subset();
  VertexSubset * all_v = graph->alloc_vertex_subset();
  VertexSubset * all_h = graph->alloc_vertex_subset();
  all->fill();
  all_v->clear();
  all_h->clear();

  graph->process_vertices<int>(
    [&](VertexId vtx) {
      if (vtx < hyper_vertices) {
        all_v->set_bit(vtx);
      } else {
        all_h->set_bit(vtx);
      }
      return 0;
    },
    all
  );

  double * rank = graph->alloc_vertex_array<double>();
  graph->fill_vertex_array(rank, 1. / double(hyper_vertices));

  for (int i_i=0;i_i<iterations;i_i++) {
    graph->process_vertices<int>(reset_rank(rank), all_h);
    graph->process_edges<int, double>(
      prop_rank::sparse_signal(graph, rank),
      prop_rank::sparse_slot(rank),
      prop_rank::dense_signal(graph, rank),
      prop_rank::dense_slot(rank),
      all_v
    );

    graph->process_vertices<int>(reset_rank(rank), all_v);
    graph->process_edges<int, double>(
      prop_rank::sparse_signal(graph, rank),
      prop_rank::sparse_slot(rank),
      prop_rank::dense_signal(graph, rank),
      prop_rank::dense_slot(rank),
      all_h
    );

    graph->process_vertices<int>(update_vertex_rank(graph, rank, hyper_vertices), all_v);
  }

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  double pr_sum = graph->process_vertices<double>(
    [&](VertexId vtx) {
      return rank[vtx];
    },
    all
  );
  if (graph->partition_id==0) {
    printf("pr_sum=%lf\n", pr_sum);
  }

  graph->gather_vertex_array(rank, 0);
  if (graph->partition_id==0) {
    VertexId max_v_i = 0;
    for (VertexId v_i=0;v_i<hyper_vertices;v_i++) {
      if (rank[v_i] > rank[max_v_i]) max_v_i = v_i;
    }
    printf("pr[%u]=%lf\n", max_v_i, rank[max_v_i]);

    VertexId max_h_i = hyper_vertices;
    for (VertexId h_i=hyper_vertices;h_i<graph->vertices;h_i++) {
      if (rank[h_i] > rank[max_h_i]) max_h_i = h_i;
    }
    printf("pr[%u]=%lf\n", max_h_i, rank[max_h_i]);
  }

  graph->dealloc_vertex_array(rank);
  delete all;
  delete all_v;
  delete all_h;
}

int main(int argc, char ** argv) {
  MPI_Instance mpi(&argc, &argv);

  if (argc<4) {
    printf("pagerank [file] [vertices] [hyperedges] ([iterations])\n");
    exit(-1);
  }
  
  VertexId vertices = std::atoi(argv[2]);
  VertexId hyperedges = std::atoi(argv[3]);

  Graph<Empty> * graph;
  graph = new Graph<Empty>();

  graph->load_directed(argv[1], vertices + hyperedges);
  int iterations = (argc == 4 ? 20 : std::atoi(argv[3]));

  compute(graph, vertices, iterations);
  for (int run=0;run<5;run++) {
    compute(graph, vertices, iterations);
  }

  delete graph;
  return 0;
}
