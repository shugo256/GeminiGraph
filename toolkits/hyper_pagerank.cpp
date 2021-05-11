#include <stdio.h>
#include <stdlib.h>

#include "core/hypergraph.hpp"

#include <math.h>

namespace hyper_pagerank {
  constexpr double damping = 0.85;

  template<typename Id>
  auto reset_rank(double * rank) {
    return [&](Id i) {
      rank[i] = 0;
      return 1;
    };
  }

  namespace prop_rank
  {
    template<typename SrcId, typename DstId>
    auto sparse_signal(Hypergraph<Empty> * graph, double * rank_src, DstId * out_degree_src) {
      return [&](SrcId src){
        graph->emit(src, rank_src[src] / double(out_degree_src[src]));
      };
    }

    template<typename SrcId, typename DstId>
    auto sparse_slot(double * rank_dst) {
      return [&](SrcId src, double msg, VertexAdjList<Empty> outgoing_adj){
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          DstId dst = ptr->neighbour;
          write_add(&rank_dst[dst], msg);
        }
        return 1;
      };
    }

    template<typename SrcId, typename DstId>
    auto dense_signal(Hypergraph<Empty> * graph, double * rank_src, DstId * out_degree_src) {
      return [&](DstId dst, VertexAdjList<Empty> incoming_adj) {
        double msg = 0;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          SrcId src = ptr->neighbour;
          msg += rank_src[src] / double(out_degree_src[src]);
        }
        if (msg > 0) graph->emit(dst, msg);
      };
    }

    template<typename DstId>
    auto dense_slot(double * rank_dst) {
      return [&](DstId dst, double msg) {
        rank_dst[dst] += msg;
        return 0;
      };
    }
  } // namespace prop_rank

  auto update_vertex_rank(Hypergraph<Empty> * graph, double * rank_v) {
    return [&](VertexId vid) {
      rank_v[vid] *= damping;
      rank_v[vid] += (1. - damping) * (1. / double(graph->vertices));
      return 1;
    };
  }
} // namespace hyper_pagerank

const double d = (double)0.85;

void compute(Hypergraph<Empty> * graph, int iterations) {
  using namespace hyper_pagerank;

  double exec_time = 0;
  exec_time -= get_time();

  double * rank_v = graph->alloc_vertex_array<double>();
  double * rank_h = graph->alloc_hyperedge_array<double>();
  VertexSubset * all_v = graph->alloc_vertex_subset();
  HyperedgeSubset * all_h = graph->alloc_hyperedge_subset();
  graph->fill_vertex_array(rank_v, 1. / double(graph->vertices));
  all_v->fill();
  all_h->fill();

  for (int i_i=0;i_i<iterations;i_i++) {
    graph->process_hyperedges<int>(
      reset_rank<HyperedgeId>(rank_h),
      all_h
    );
    graph->prop_from_vertices<int, double>(
      prop_rank::sparse_signal<VertexId, HyperedgeId>(graph, rank_v, graph->out_degree_v),
      prop_rank::sparse_slot<VertexId, HyperedgeId>(rank_h),
      prop_rank::dense_signal<VertexId, HyperedgeId>(graph, rank_v, graph->out_degree_v),
      prop_rank::dense_slot<HyperedgeId>(rank_h),
      all_v
    );
    graph->process_vertices<int>(
      reset_rank<VertexId>(rank_v),
      all_v
    );
    graph->prop_from_hyperedges<int, double>(
      prop_rank::sparse_signal<HyperedgeId, VertexId>(graph, rank_h, graph->out_degree_h),
      prop_rank::sparse_slot<HyperedgeId, VertexId>(rank_v),
      prop_rank::dense_signal<HyperedgeId, VertexId>(graph, rank_h, graph->out_degree_h),
      prop_rank::dense_slot<VertexId>(rank_v),
      all_h
    );
    graph->process_vertices<int>(
      update_vertex_rank(graph, rank_v),
      all_v
    );
  }

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  double pr_sum = graph->process_vertices<double>(
    [&](VertexId vtx) {
      return rank_v[vtx];
    },
    all_v
  ) + graph->process_hyperedges<double>(
    [&](VertexId vtx) {
      return rank_h[vtx];
    },
    all_h
  );
  if (graph->partition_id==0) {
    printf("pr_sum=%lf\n", pr_sum);
  }

  graph->gather_vertex_array(rank_v, 0);
  graph->gather_hyperedge_array(rank_h, 0);
  if (graph->partition_id==0) {
    VertexId max_v_i = 0;
    for (VertexId v_i=0;v_i<graph->vertices;v_i++) {
      if (rank_v[v_i] > rank_v[max_v_i]) max_v_i = v_i;
    }
    printf("pr[%u]=%lf\n", max_v_i, rank_v[max_v_i]);

    HyperedgeId max_h_i = 0;
    for (HyperedgeId h_i=0;h_i<graph->hyperedges;h_i++) {
      if (rank_h[h_i] > rank_h[max_h_i]) max_h_i = h_i;
    }
    printf("pr[%u]=%lf\n", graph->vertices + max_h_i, rank_h[max_h_i]);
  }

  graph->dealloc_vertex_array(rank_v);
  graph->dealloc_hyperedge_array(rank_h);
  delete all_v;
  delete all_h;
}

int main(int argc, char ** argv) {
  MPI_Instance mpi(&argc, &argv);

  if (argc<4) {
    printf("pagerank [file] [vertices] [hyperedges] ([iterations])\n");
    exit(-1);
  }

  Hypergraph<Empty> * graph;
  graph = new Hypergraph<Empty>();
  graph->load_directed(argv[1], std::atoi(argv[2]), std::atoi(argv[3]));
  int iterations = (argc == 4 ? 20 : std::atoi(argv[4]));

  compute(graph, iterations);
  for (int run=0;run<5;run++) {
    compute(graph, iterations);
  }

  delete graph;
  return 0;
}
