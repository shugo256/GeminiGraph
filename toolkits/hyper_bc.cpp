#include <stdio.h>
#include <stdlib.h>

#include "core/hypergraph.hpp"

// sample command
// toolkits/hyper_bc ../datasets/com-orkut.bin 2322299 15301901

namespace hyper_bc
{
  namespace forward
  {
    template<typename SrcId>
    auto sparse_signal(Hypergraph<Empty> * graph, double * num_paths_src) {
      return [&](SrcId src){
        graph->emit(src, num_paths_src[src]);
      };
    }

    template<typename SrcId, typename DstId, typename DstSubset>
    auto sparse_slot(Hypergraph<Empty> * graph, double * num_paths_dst, DstSubset * visited_dst, DstSubset * active_dst) {
      return [&](SrcId src, double msg, VertexAdjList<Empty> outgoing_adj){
        DstId activated = 0;
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          DstId dst = ptr->neighbour;
          if (!visited_dst->get_bit(dst)) {
            write_add(&num_paths_dst[dst], msg);
            active_dst->set_bit(dst);
            activated += 1;
          }
        }
        return activated;
      };
    }

    template<typename SrcId, typename DstId, typename SrcSubset, typename DstSubset>
    auto dense_signal(Hypergraph<Empty> * graph, double * num_paths_src, DstSubset * visited_dst, SrcSubset * active_src) {
      return [&](DstId dst, VertexAdjList<Empty> incoming_adj) {
        // visited_dst is shared entirely with every partition
        // because it will be passed to prop_to_xx as dense_selective
        if (visited_dst->get_bit(dst)) return;

        double msg = 0.0;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          SrcId src = ptr->neighbour;
          if (active_src->get_bit(src)) {
            msg += num_paths_src[src];
          }
        }
        if (msg > 0) {
          graph->emit(dst, msg);
        }
      };
    }

    template<typename DstId, typename DstSubset>
    auto dense_slot(Hypergraph<Empty> * graph, double * num_paths_dst, DstSubset * visited_dst, DstSubset * active_dst) {
      return [&](DstId dst, double msg) {
        if (!visited_dst->get_bit(dst)) {
          write_add(&num_paths_dst[dst], msg);
          active_dst->set_bit(dst);
          return 1;
        }
        return 0;
      };
    }

    template<typename ElemId, typename ElemSubset>
    auto mark_visited(ElemSubset * visited) {
      return [&](ElemId elem) {
        visited->set_bit(elem);
        return 1;
      };
    }
  } // namespace forward

  namespace backward
  {
    template<typename SrcId, double msgFromSrc(SrcId, double *, double *)>
    auto sparse_signal(Hypergraph<Empty> * graph, double * dependencies_src, double * num_paths_v) {
      return [&](SrcId src){
        graph->emit(src, msgFromSrc(src, dependencies_src, num_paths_v));
      };
    }

    template<typename SrcId, typename DstId, double updateDst(DstId, double, double *), typename DstSubset>
    auto sparse_slot(Hypergraph<Empty> * graph, double * dependencies_dst, double * num_paths_v, DstSubset * visited_dst) {
      return [&](SrcId src, double msg, VertexAdjList<Empty> outgoing_adj){
        DstId activated = 0;
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          DstId dst = ptr->neighbour;
          if (!visited_dst->get_bit(dst)) {
            write_add(&dependencies_dst[dst], updateDst(dst, msg, num_paths_v));
            activated += 1;
          }
        }
        return activated;
      };
    }

    template<typename SrcId, typename DstId, double msgFromSrc(SrcId, double *, double *), typename SrcSubset, typename DstSubset>
    auto dense_signal(Hypergraph<Empty> * graph, double * dependencies_src, double * num_paths_v, DstSubset * visited_dst, SrcSubset * active_src) {
      return [&](DstId dst, VertexAdjList<Empty> incoming_adj) {
        // visited_dst is shared entirely with every partition
        // because it will be passed to prop_to_xx as dense_selective
        if (visited_dst->get_bit(dst)) return;

        double msg = 0.0;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          SrcId src = ptr->neighbour;
          if (active_src->get_bit(src)) {
            msg += msgFromSrc(src, dependencies_src, num_paths_v);
          }
        }
        if (msg > 0) {
          graph->emit(dst, msg);
        }
      };
    }

    template<typename DstId, double updateDst(DstId, double, double *), typename DstSubset>
    auto dense_slot(Hypergraph<Empty> * graph, double * dependencies_dst, double * num_paths_v, DstSubset * visited_dst) {
      return [&](DstId dst, double msg) {
        if (!visited_dst->get_bit(dst)) {
          write_add(&dependencies_dst[dst], updateDst(dst, msg, num_paths_v));
          return 1;
        }
        return 0;
      };
    }

    template<typename ElemId, typename ElemSubset>
    auto mark_visited_v(double * dependencies_v, ElemSubset * visited_v) {
      return [&](ElemId elem) {
        visited_v->set_bit(elem);
        dependencies_v[elem] += 1;
        return 1;
      };
    }
  } // namespace backward
  
} // namespace hyper_bc

double msgFromV(VertexId src, double * dependencies_v, double * num_paths_v) {
  return dependencies_v[src] / num_paths_v[src];
}
double updateH(HyperedgeId dst, double msg, double * ignore) {
  return msg;
}
double msgFromH(HyperedgeId src, double * dependencies_h, double * ignore) {
  return dependencies_h[src];
}
double updateV(VertexId dst, double msg, double * num_paths_v) {
  return msg * num_paths_v[dst];
}


void compute(Hypergraph<Empty> * graph, VertexId root) {
  double exec_time = 0;
  exec_time -= get_time();

  double * num_paths_v = graph->alloc_vertex_array<double>();
  double * num_paths_h = graph->alloc_hyperedge_array<double>();
  graph->fill_vertex_array(num_paths_v, 0.0);
  graph->fill_hyperedge_array(num_paths_h, 0.0);
  num_paths_v[root] = 1.0;

  VertexSubset * visited_v = graph->alloc_vertex_subset();
  HyperedgeSubset * visited_h = graph->alloc_hyperedge_subset();
  visited_v->clear();
  visited_h->clear();
  visited_v->set_bit(root);

  VertexSubset * active_v = graph->alloc_vertex_subset();
  HyperedgeSubset * active_h;
  active_v->clear();
  active_v->set_bit(root);
  VertexId active_vertices = 1;
  HyperedgeId active_hyperedges = 0;

  std::vector<Bitmap *> levels;
  levels.push_back(active_v);

  long i_i = 0;;

  if (graph->partition_id==0) {
    printf("forward\n");
  }
  for (i_i=0;active_vertices>0;i_i++) {
    if (graph->partition_id==0) {
      printf("active(%d)>=%u\n", i_i, active_vertices);
    }
    active_h = graph->alloc_hyperedge_subset();
    active_h->clear();
    graph->prop_from_vertices<HyperedgeId,double>(
      hyper_bc::forward::sparse_signal<VertexId>(graph, num_paths_v),
      hyper_bc::forward::sparse_slot<VertexId, HyperedgeId>(graph, num_paths_h, visited_h, active_h),
      hyper_bc::forward::dense_signal<VertexId, HyperedgeId>(graph, num_paths_v, visited_h, active_v),
      hyper_bc::forward::dense_slot<HyperedgeId>(graph, num_paths_h, visited_h, active_h),
      active_v, visited_h
    );
    active_hyperedges = graph->process_hyperedges<HyperedgeId>(
      hyper_bc::forward::mark_visited<HyperedgeId>(visited_h),
      active_h
    );
    levels.push_back(active_h);

    if (active_hyperedges == 0) break;
    
    active_v = graph->alloc_vertex_subset();
    active_v->clear();
    graph->prop_from_hyperedges<VertexId,double>(
      hyper_bc::forward::sparse_signal<HyperedgeId>(graph, num_paths_h),
      hyper_bc::forward::sparse_slot<HyperedgeId, VertexId>(graph, num_paths_v, visited_v, active_v),
      hyper_bc::forward::dense_signal<HyperedgeId, VertexId>(graph, num_paths_h, visited_v, active_h),
      hyper_bc::forward::dense_slot<VertexId>(graph, num_paths_v, visited_v, active_v),
      active_h, visited_v
    );
    active_vertices = graph->process_vertices<VertexId>(
      hyper_bc::forward::mark_visited<VertexId>(visited_v),
      active_v
    );
    levels.push_back(active_v);

    if (active_vertices == 0) break;
  }

  // the last phase should end on hyperedges
  if (levels.size() % 2 == 1) {
    delete levels.back();
    levels.pop_back();
  }
  delete levels.back();
  levels.pop_back();

  double * dependencies_v = graph->alloc_vertex_array<double>();
  double * dependencies_h = graph->alloc_hyperedge_array<double>();
  graph->fill_vertex_array(dependencies_v, 0.0);
  graph->fill_hyperedge_array(dependencies_h, 0.0);

  visited_v->clear();
  visited_h->clear();
  graph->transpose();

  if (graph->partition_id==0) {
    printf("backward\n");
  }
  while (levels.size() > 1) {
    graph->process_vertices<VertexId>(
      hyper_bc::backward::mark_visited_v<VertexId>(dependencies_v, visited_v),
      levels.back()
    );
    graph->prop_from_vertices<HyperedgeId,double>(
      hyper_bc::backward::sparse_signal<VertexId, msgFromV>(graph, dependencies_v, num_paths_v),
      hyper_bc::backward::sparse_slot<VertexId, HyperedgeId, updateH>(graph, dependencies_h, nullptr, visited_h),
      hyper_bc::backward::dense_signal<VertexId, HyperedgeId, msgFromV>(graph, dependencies_v, num_paths_v, visited_h, levels.back()),
      hyper_bc::backward::dense_slot<HyperedgeId, updateH>(graph, dependencies_h, nullptr, visited_h),
      levels.back(), visited_h
    );
    delete levels.back();
    levels.pop_back();
    
    graph->process_hyperedges<HyperedgeId>(
      // same as forward
      hyper_bc::forward::mark_visited<HyperedgeId>(visited_h),
      levels.back()
    );
    graph->prop_from_hyperedges<VertexId,double>(
      hyper_bc::backward::sparse_signal<HyperedgeId, msgFromH>(graph, dependencies_h, nullptr),
      hyper_bc::backward::sparse_slot<HyperedgeId, VertexId, updateV>(graph, dependencies_v, num_paths_v, visited_v),
      hyper_bc::backward::dense_signal<HyperedgeId, VertexId, msgFromH>(graph, dependencies_h, nullptr, visited_v, levels.back()),
      hyper_bc::backward::dense_slot<VertexId, updateV>(graph, dependencies_v, num_paths_v, visited_v),
      levels.back(), visited_v
    );
    delete levels.back();
    levels.pop_back();
  }
  // delete first level
  delete levels.back();
  levels.pop_back();
  
  graph->transpose();

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  // graph->gather_vertex_array(dependencies_v, 0);
  // if (graph->partition_id==0) {
  //   for (VertexId v_i=0;v_i<20;v_i++) {
  //     printf("%d %lf %lf\n", v_i, num_paths_v[v_i], dependencies_v[v_i]);
  //   }
  // }

  graph->dealloc_vertex_array(num_paths_v);
  graph->dealloc_hyperedge_array(num_paths_h);
  graph->dealloc_vertex_array(dependencies_v);
  graph->dealloc_hyperedge_array(dependencies_h);
  delete visited_v;
  delete visited_h;
}

int main(int argc, char ** argv) {
  MPI_Instance mpi(&argc, &argv);

  if (argc<4) {
    printf("hyper_bc [file] [vertices] [hyperedges] ([root])\n");
    exit(-1);
  }

  Hypergraph<Empty> * graph;
  graph = new Hypergraph<Empty>();
  VertexId root = (argc == 4 ? 0 : std::atoi(argv[4]));
  graph->load_directed(argv[1], std::atoi(argv[2]), std::atoi(argv[3]));

  compute(graph, root);
  for (int run=0;run<5;run++) {
    compute(graph, root);
  }

  delete graph;
  return 0;
}
