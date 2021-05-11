/*
Copyright (c) 2015-2016 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef HYPERGRAPH_HPP
#define HYPERGRAPH_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <malloc.h>
#include <sys/mman.h>
#include <numa.h>
#include <omp.h>

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <functional>

#include "core/atomic.hpp"
#include "core/bitmap.hpp"
#include "core/constants.hpp"
#include "core/filesystem.hpp"
#include "core/mpi.hpp"
#include "core/time.hpp"
#include "core/type.hpp"


// types for bipartite representation
using BipartiteVertexId = uint32_t;
using BipartiteVertexSubset = Bitmap;
using BipartiteEdgeId = uint64_t;

// types for hypergraph APIs
using VertexId = BipartiteVertexId;
using VertexSubset = BipartiteVertexSubset;
using HyperedgeId = BipartiteVertexId;
using HyperedgeSubset = BipartiteVertexSubset;

enum ThreadStatus {
  WORKING,
  STEALING
};

enum MessageTag {
  ShuffleGraph,
  PassMessage,
  GatherVertexArray
};

struct ThreadState {
  BipartiteVertexId curr;
  BipartiteVertexId end;
  ThreadStatus status;
};

struct MessageBuffer {
  size_t capacity;
  int count; // the actual size (i.e. bytes) should be sizeof(element) * count
  char * data;
  MessageBuffer () {
    capacity = 0;
    count = 0;
    data = NULL;
  }
  void init (int socket_id) {
    capacity = 4096;
    count = 0;
    data = (char*)numa_alloc_onnode(capacity, socket_id);
  }
  void resize(size_t new_capacity) {
    if (new_capacity > capacity) {
      char * new_data = (char*)numa_realloc(data, capacity, new_capacity);
      assert(new_data!=NULL);
      data = new_data;
      capacity = new_capacity;
    }
  }
};

template <typename MsgData>
struct MsgUnit {
  VertexId vertex;
  MsgData msg_data;
} __attribute__((packed));


template <typename EdgeData = Empty>
class Hypergraph {
public:
  int partition_id;
  int partitions;

  size_t alpha;

  int threads;
  int sockets;
  int threads_per_socket;

  size_t edge_data_size;
  size_t unit_size;
  size_t edge_unit_size;

  bool symmetric;
  bool flipped;
  VertexId vertices;
  HyperedgeId hyperedges;
  BipartiteVertexId bipartite_vertices;
  BipartiteEdgeId edges; // number of edges in the bipartite representation
  BipartiteEdgeId edges_from_v;
  BipartiteEdgeId edges_from_h;

  HyperedgeId * out_degree_v; // HyperedgeId [vertices]; numa-aware
  HyperedgeId * in_degree_v; // HyperedgeId [vertices]; numa-aware
  VertexId * out_degree_h; // VertexId [hyperedges]; numa-aware
  VertexId * in_degree_h; // VertexId [hyperedges]; numa-aware

private:
  VertexId * partition_offset_v; // VertexId [partitions+1]
  VertexId * local_partition_offset_v; // VertexId [sockets+1]
  HyperedgeId * partition_offset_h; // HyperedgeId [partitions+1]
  HyperedgeId * local_partition_offset_h; // HyperedgeId [sockets+1]

  VertexId owned_vertices;
  HyperedgeId owned_hyperedges;
  BipartiteEdgeId * outgoing_edges_v; // BipartiteEdgeId [sockets]
  BipartiteEdgeId * outgoing_edges_h; // BipartiteEdgeId [sockets]
  BipartiteEdgeId * incoming_edges_v; // BipartiteEdgeId [sockets]
  BipartiteEdgeId * incoming_edges_h; // BipartiteEdgeId [sockets]

  Bitmap ** incoming_adj_bitmap_v;
  Bitmap ** incoming_adj_bitmap_h;
  BipartiteEdgeId ** incoming_adj_index_v; // BipartiteEdgeId [sockets] [vertices+1]; numa-aware
  BipartiteEdgeId ** incoming_adj_index_h; // BipartiteEdgeId [sockets] [hyperedges+1]; numa-aware
  AdjUnit<EdgeData> ** incoming_adj_list_v; // AdjUnit<EdgeData> [sockets] [vertices+1]; numa-aware
  AdjUnit<EdgeData> ** incoming_adj_list_h; // AdjUnit<EdgeData> [sockets] [hyperedges+1]; numa-aware
  Bitmap ** outgoing_adj_bitmap_v;
  Bitmap ** outgoing_adj_bitmap_h;
  BipartiteEdgeId ** outgoing_adj_index_v; // BipartiteEdgeId [sockets] [vertices+1]; numa-aware
  BipartiteEdgeId ** outgoing_adj_index_h; // BipartiteEdgeId [sockets] [hyperedges+1]; numa-aware
  AdjUnit<EdgeData> ** outgoing_adj_list_v; // AdjUnit<EdgeData> [sockets] [vertices+1]; numa-aware
  AdjUnit<EdgeData> ** outgoing_adj_list_h; // AdjUnit<EdgeData> [sockets] [hyperedges+1]; numa-aware

  VertexId * compressed_incoming_adj_vertices;
  VertexId * compressed_incoming_adj_hyperedges;
  CompressedAdjIndexUnit ** compressed_incoming_adj_index_v; // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware
  CompressedAdjIndexUnit ** compressed_incoming_adj_index_h; // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware
  VertexId * compressed_outgoing_adj_vertices;
  VertexId * compressed_outgoing_adj_hyperedges;
  CompressedAdjIndexUnit ** compressed_outgoing_adj_index_v; // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware
  CompressedAdjIndexUnit ** compressed_outgoing_adj_index_h; // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware

  ThreadState ** thread_state; // ThreadState* [threads]; numa-aware
  ThreadState ** tuned_vertex_chunks_dense; // ThreadState [partitions][threads];
  ThreadState ** tuned_hyperedge_chunks_dense; // ThreadState [partitions][threads];
  ThreadState ** tuned_vertex_chunks_sparse; // ThreadState [partitions][threads];
  ThreadState ** tuned_hyperedge_chunks_sparse; // ThreadState [partitions][threads];

  size_t local_send_buffer_limit;
  MessageBuffer ** local_send_buffer; // MessageBuffer* [threads]; numa-aware

  int current_send_part_id;
  MessageBuffer *** send_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware
  MessageBuffer *** recv_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware

  // used in filter functions;
  Bitmap * tmp_bitmap;

public:
  Hypergraph() {
    threads = numa_num_configured_cpus();
    // threads = omp_get_max_threads();
    sockets = numa_num_configured_nodes();
    threads_per_socket = threads / sockets;
    flipped = false;

    tmp_bitmap = nullptr;

    init();
  }

  inline int get_socket_id(int thread_id) {
    return thread_id / threads_per_socket;
  }

  inline int get_socket_offset(int thread_id) {
    return thread_id % threads_per_socket;
  }

  void init() {
    edge_data_size = std::is_same<EdgeData, Empty>::value ? 0 : sizeof(EdgeData);
    unit_size = sizeof(VertexId) + edge_data_size;
    edge_unit_size = sizeof(VertexId) + unit_size;

    assert( numa_available() != -1 );
    assert( sizeof(unsigned long) == 8 ); // assume unsigned long is 64-bit

    char nodestring[sockets*2+1];
    nodestring[0] = '0';
    for (int s_i=1;s_i<sockets;s_i++) {
      nodestring[s_i*2-1] = ',';
      nodestring[s_i*2] = '0'+s_i;
    }
    struct bitmask * nodemask = numa_parse_nodestring(nodestring);
    numa_set_interleave_mask(nodemask);

    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    thread_state = new ThreadState * [threads];
    local_send_buffer_limit = 16;
    local_send_buffer = new MessageBuffer * [threads];
    for (int t_i=0;t_i<threads;t_i++) {
      thread_state[t_i] = (ThreadState*)numa_alloc_onnode( sizeof(ThreadState), get_socket_id(t_i));
      local_send_buffer[t_i] = (MessageBuffer*)numa_alloc_onnode( sizeof(MessageBuffer), get_socket_id(t_i));
      local_send_buffer[t_i]->init(get_socket_id(t_i));
    }
    #pragma omp parallel for
    for (int t_i=0;t_i<threads;t_i++) {
      int s_i = get_socket_id(t_i);
      assert(numa_run_on_node(s_i)==0);
      #ifdef PRINT_DEBUG_MESSAGES
      printf("thread-%d bound to socket-%d\n", t_i, s_i);
      #endif
    }
    #ifdef PRINT_DEBUG_MESSAGES
    printf("threads=%d*%d\n", sockets, threads_per_socket);
    printf("interleave on %s\n", nodestring);
    #endif

    MPI_Comm_rank(MPI_COMM_WORLD, &partition_id);
    MPI_Comm_size(MPI_COMM_WORLD, &partitions);
    send_buffer = new MessageBuffer ** [partitions];
    recv_buffer = new MessageBuffer ** [partitions];
    for (int i=0;i<partitions;i++) {
      send_buffer[i] = new MessageBuffer * [sockets];
      recv_buffer[i] = new MessageBuffer * [sockets];
      for (int s_i=0;s_i<sockets;s_i++) {
        send_buffer[i][s_i] = (MessageBuffer*)numa_alloc_onnode( sizeof(MessageBuffer), s_i);
        send_buffer[i][s_i]->init(s_i);
        recv_buffer[i][s_i] = (MessageBuffer*)numa_alloc_onnode( sizeof(MessageBuffer), s_i);
        recv_buffer[i][s_i]->init(s_i);
      }
    }

    alpha = 8 * (partitions - 1);

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // vertex set <-> hyperedge set
  void flip_bipartite_graph(bool flip_public_vars=true) {
    flipped ^= 1;

    if (flip_public_vars) {
      std::swap(vertices, hyperedges);
      std::swap(edges_from_v, edges_from_h);
      std::swap(out_degree_v, out_degree_h);
      std::swap(in_degree_v, in_degree_h);
    }

    std::swap(partition_offset_v, partition_offset_h);
    std::swap(local_partition_offset_v, local_partition_offset_h);

    std::swap(owned_vertices, owned_hyperedges);
    std::swap(outgoing_edges_v, outgoing_edges_h);
    std::swap(incoming_edges_v, incoming_edges_h);

    std::swap(incoming_adj_bitmap_v, incoming_adj_bitmap_h);
    std::swap(incoming_adj_index_v, incoming_adj_index_h);
    std::swap(incoming_adj_list_v, incoming_adj_list_h);
    std::swap(outgoing_adj_bitmap_v, outgoing_adj_bitmap_h);
    std::swap(outgoing_adj_index_v, outgoing_adj_index_h);
    std::swap(outgoing_adj_list_v, outgoing_adj_list_h);

    std::swap(compressed_incoming_adj_vertices, compressed_incoming_adj_hyperedges);
    std::swap(compressed_incoming_adj_index_v, compressed_incoming_adj_index_h);
    std::swap(compressed_outgoing_adj_vertices, compressed_outgoing_adj_hyperedges);
    std::swap(compressed_outgoing_adj_index_v, compressed_outgoing_adj_index_h);

    std::swap(tuned_vertex_chunks_dense, tuned_hyperedge_chunks_dense);
    std::swap(tuned_vertex_chunks_sparse, tuned_hyperedge_chunks_sparse);
  }

  // fill a vertex array with a specific value
  template<typename T>
  void fill_vertex_array(T * array, T value) {
    #pragma omp parallel for
    for (VertexId v_i=partition_offset_v[partition_id];v_i<partition_offset_v[partition_id+1];v_i++) {
      array[v_i] = value;
    }
  }

  // fill a hyperedge array with a specific value
  template<typename T>
  void fill_hyperedge_array(T * array, T value) {
    flip_bipartite_graph();
    fill_vertex_array(array, value);
    flip_bipartite_graph();
  }

  // allocate a numa-aware array
  template<typename T>
  T * alloc_vertex_array() {
    char * array = (char *)mmap(NULL, sizeof(T) * vertices, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(array!=NULL);
    for (int s_i=0;s_i<sockets;s_i++) {
      numa_tonode_memory(array + sizeof(T) * local_partition_offset_v[s_i], sizeof(T) * (local_partition_offset_v[s_i+1] - local_partition_offset_v[s_i]), s_i);
    }
    return (T*)array;
  }
  template<typename T>
  T * alloc_hyperedge_array() {
    flip_bipartite_graph();
    auto allocated_array = alloc_vertex_array<T>();
    flip_bipartite_graph();
    return allocated_array;
  }

  // deallocate array
  template<typename T>
  T * dealloc_vertex_array(T * array) {
    numa_free(array, sizeof(T) * vertices);
  }
  template<typename T>
  T * dealloc_hyperedge_array(T * array) {
    numa_free(array, sizeof(T) * hyperedges);
  }

  // allocate a numa-oblivious array
  template<typename T>
  T * alloc_interleaved_vertex_array() {
    T * array = (T *)numa_alloc_interleaved( sizeof(T) * vertices );
    assert(array!=NULL);
    return array;
  }
  template<typename T>
  T * alloc_interleaved_hyperedge_array() {
    flip_bipartite_graph();
    auto allocated_array = alloc_interleaved_vertex_array<T>();
    flip_bipartite_graph();
    return allocated_array;
  }

  // dump array to path
  template<typename T>
  void dump_vertex_array(T * array, std::string path) {
    long file_length = sizeof(T) * vertices;
    if (!file_exists(path) || file_size(path) != file_length) {
      if (partition_id==0) {
        FILE * fout = fopen(path.c_str(), "wb");
        char * buffer = new char [PAGESIZE];
        for (long offset=0;offset<file_length;) {
          if (file_length - offset >= PAGESIZE) {
            fwrite(buffer, 1, PAGESIZE, fout);
            offset += PAGESIZE;
          } else {
            fwrite(buffer, 1, file_length - offset, fout);
            offset += file_length - offset;
          }
        }
        fclose(fout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    int fd = open(path.c_str(), O_RDWR);
    assert(fd!=-1);
    long offset = sizeof(T) * partition_offset_v[partition_id];
    long end_offset = sizeof(T) * partition_offset_v[partition_id+1];
    void * data = (void *)array;
    assert(lseek(fd, offset, SEEK_SET)!=-1);
    while (offset < end_offset) {
      long bytes = write(fd, data + offset, end_offset - offset);
      assert(bytes!=-1);
      offset += bytes;
    }
    assert(close(fd)==0);
  }
  template<typename T>
  void dump_hyperedge_array(T * array, std::string path) {
    flip_bipartite_graph();
    dump_vertex_array(array, path);
    flip_bipartite_graph();
  }

  template<typename T>
  void restore_vertex_array(T * array, std::string path) {
    long file_length = sizeof(T) * vertices;
    if (!file_exists(path) || file_size(path) != file_length) {
      assert(false);
    }
    int fd = open(path.c_str(), O_RDWR);
    assert(fd!=-1);
    long offset = sizeof(T) * partition_offset_v[partition_id];
    long end_offset = sizeof(T) * partition_offset_v[partition_id+1];
    void * data = (void *)array;
    assert(lseek(fd, offset, SEEK_SET)!=-1);
    while (offset < end_offset) {
      long bytes = read(fd, data + offset, end_offset - offset);
      assert(bytes!=-1);
      offset += bytes;
    }
    assert(close(fd)==0);
  }
  template<typename T>
  void restore_hyperedge_array(T * array, std::string path) {
    flip_bipartite_graph();
    restore_vertex_array(array, path);
    flip_bipartite_graph();
  }

  // gather array
  template<typename T>
  void gather_vertex_array(T * array, int root) {
    if (partition_id!=root) {
      MPI_Send(array + partition_offset_v[partition_id], sizeof(T) * owned_vertices, MPI_CHAR, root, GatherVertexArray, MPI_COMM_WORLD);
    } else {
      for (int i=0;i<partitions;i++) {
        if (i==partition_id) continue;
        MPI_Status recv_status;
        MPI_Recv(array + partition_offset_v[i], sizeof(T) * (partition_offset_v[i + 1] - partition_offset_v[i]), MPI_CHAR, i, GatherVertexArray, MPI_COMM_WORLD, &recv_status);
        int length;
        MPI_Get_count(&recv_status, MPI_CHAR, &length);
        assert(length == sizeof(T) * (partition_offset_v[i + 1] - partition_offset_v[i]));
      }
    }
  }
  template<typename T>
  void gather_hyperedge_array(T * array, int root) {
    flip_bipartite_graph();
    gather_vertex_array(array, root);
    flip_bipartite_graph();
  }

  // allocate new subset
  VertexSubset * alloc_vertex_subset() {
    return new VertexSubset(vertices);
  }
  HyperedgeSubset * alloc_hyperedge_subset() {
    return new HyperedgeSubset(hyperedges);
  }

  int get_partition_id_v(VertexId v_i){
    for (int i=0;i<partitions;i++) {
      if (v_i >= partition_offset_v[i] && v_i < partition_offset_v[i+1]) {
        return i;
      }
    }
    assert(false);
  }
  int get_partition_id_h(HyperedgeId h_i){
    for (int i=0;i<partitions;i++) {
      if (h_i >= partition_offset_h[i] && h_i < partition_offset_h[i+1]) {
        return i;
      }
    }
    assert(false);
  }

  int get_local_partition_id_v(VertexId v_i){
    for (int s_i=0;s_i<sockets;s_i++) {
      if (v_i >= local_partition_offset_v[s_i] && v_i < local_partition_offset_v[s_i+1]) {
        return s_i;
      }
    }
    assert(false);
  }
  int get_local_partition_id_h(HyperedgeId h_i){
    for (int s_i=0;s_i<sockets;s_i++) {
      if (h_i >= local_partition_offset_h[s_i] && h_i < local_partition_offset_h[s_i+1]) {
        return s_i;
      }
    }
    assert(false);
  }

  template<typename ID>
  ID set_partition_offset(
      ID * partition_offset, int partition_num, ID bv_num, ID * out_degree, 
      BipartiteEdgeId e_num, ID start_offset, ID end_offset) {

    partition_offset[0] = start_offset;

    BipartiteEdgeId remained_amount = e_num + BipartiteEdgeId(bv_num) * alpha;

    for (int i=0;i<partition_num;i++) {
      BipartiteEdgeId remained_partitions = partition_num - i;
      BipartiteEdgeId expected_chunk_size = remained_amount / remained_partitions;

      if (remained_partitions==1) {
        partition_offset[i+1] = end_offset;
      } else {
        BipartiteEdgeId got_edges = 0;
        for (ID bv_i=partition_offset[i];bv_i<end_offset;bv_i++) {
          got_edges += out_degree[bv_i] + alpha;
          if (got_edges > expected_chunk_size) {
            partition_offset[i+1] = bv_i;
            break;
          }
        }
        partition_offset[i+1] = (partition_offset[i+1]) / PAGESIZE * PAGESIZE; // aligned with pages
      }
      for (VertexId v_i=partition_offset[i];v_i<partition_offset[i+1];v_i++) {
        remained_amount -= out_degree[v_i] + alpha;
      }
    }
    assert(partition_offset[partition_num]==end_offset);

    ID owned_bv_num = partition_offset[partition_id+1] - partition_offset[partition_id];

    return owned_bv_num;
  }
  template<typename ID>
  ID set_partition_offset(
      ID * partition_offset, int partition_num, ID bv_num, ID * out_degree, BipartiteEdgeId e_num) {
    return set_partition_offset(partition_offset, partition_num, bv_num, out_degree, e_num, ID(0), bv_num);
  }

  void set_compressed_outgoing_adj_datas_v() {
    for (int s_i=0;s_i<sockets;s_i++) {
      outgoing_edges_v[s_i] = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (outgoing_adj_bitmap_v[s_i]->get_bit(v_i)) {
          outgoing_edges_v[s_i] += outgoing_adj_index_v[s_i][v_i];
          compressed_outgoing_adj_vertices[s_i] += 1;
        }
      }
      compressed_outgoing_adj_index_v[s_i] = (CompressedAdjIndexUnit*)numa_alloc_onnode( sizeof(CompressedAdjIndexUnit) * (compressed_outgoing_adj_vertices[s_i] + 1) , s_i );
      compressed_outgoing_adj_index_v[s_i][0].index = 0;
      BipartiteEdgeId last_e_i = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (outgoing_adj_bitmap_v[s_i]->get_bit(v_i)) {
          outgoing_adj_index_v[s_i][v_i] = last_e_i + outgoing_adj_index_v[s_i][v_i];
          last_e_i = outgoing_adj_index_v[s_i][v_i];
          compressed_outgoing_adj_index_v[s_i][compressed_outgoing_adj_vertices[s_i]].vertex = v_i;
          compressed_outgoing_adj_vertices[s_i] += 1;
          compressed_outgoing_adj_index_v[s_i][compressed_outgoing_adj_vertices[s_i]].index = last_e_i;
        }
      }
      for (VertexId p_v_i=0;p_v_i<compressed_outgoing_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_outgoing_adj_index_v[s_i][p_v_i].vertex;
        outgoing_adj_index_v[s_i][v_i] = compressed_outgoing_adj_index_v[s_i][p_v_i].index;
        outgoing_adj_index_v[s_i][v_i+1] = compressed_outgoing_adj_index_v[s_i][p_v_i+1].index;
      }
    }
  }
  void set_compressed_outgoing_adj_datas_h() {
    flip_bipartite_graph();
    set_compressed_outgoing_adj_datas_v();
    flip_bipartite_graph();
  }
  void set_compressed_incoming_adj_datas_v() {
    transpose();
    set_compressed_outgoing_adj_datas_v();
    transpose();
  }
  void set_compressed_incoming_adj_datas_h() {
    transpose();
    set_compressed_outgoing_adj_datas_h();
    transpose();
  }

  // load a directed graph and make it undirected
  // とりあえずhyperedgeは{hid + vertices}で表す
  void load_undirected_from_directed(std::string path, VertexId vertices, HyperedgeId hyperedges) {
    double prep_time = 0;
    prep_time -= MPI_Wtime();

    symmetric = true;

    MPI_Datatype vid_t = get_mpi_data_type<VertexId>();
    MPI_Datatype eid_t = get_mpi_data_type<BipartiteEdgeId>();

    this->vertices = vertices;
    this->hyperedges = hyperedges;
    this->bipartite_vertices = vertices + hyperedges;
    long total_bytes = file_size(path.c_str());
    this->edges = total_bytes / edge_unit_size;
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("|V| = %u, |H| = %u, |E| = %lu\n", vertices, hyperedges, edges);
    }
    #endif

    BipartiteEdgeId read_edges = edges / partitions;
    if (partition_id==partitions-1) {
      read_edges += edges % partitions;
    }
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = edge_unit_size * (edges / partitions * partition_id);
    long read_bytes;
    int fin = open(path.c_str(), O_RDONLY);
    EdgeUnit<EdgeData> * read_edge_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE];

    out_degree_v = alloc_interleaved_vertex_array<HyperedgeId>();
    out_degree_h = alloc_interleaved_hyperedge_array<VertexId>();
    for (VertexId v_i=0;v_i<vertices;v_i++) {
      out_degree_v[v_i] = 0;
    }
    for (HyperedgeId h_i=0;h_i<hyperedges;h_i++) {
      out_degree_h[h_i] = 0;
    }
    assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
    read_bytes = 0;
    edges_from_v = edges_from_h = 0;
    while (read_bytes < bytes_to_read) {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      } else {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes>=0);
      read_bytes += curr_read_bytes;
      BipartiteEdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      // #pragma omp parallel for
      for (BipartiteEdgeId e_i=0;e_i<curr_read_edges;e_i++) {
        BipartiteVertexId src = read_edge_buffer[e_i].src;
        BipartiteVertexId dst = read_edge_buffer[e_i].dst;
        auto src_degree = (src < vertices ? &out_degree_v[src] : &out_degree_h[src-vertices]);
        auto dst_degree = (dst < vertices ? &out_degree_v[dst] : &out_degree_h[dst-vertices]);
        __sync_fetch_and_add(src_degree, 1);
        __sync_fetch_and_add(dst_degree, 1);
        src < vertices ? edges_from_v++ : edges_from_h++;
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, out_degree_v, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, out_degree_h, hyperedges, vid_t, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, &edges_from_v, 1, eid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &edges_from_h, 1, eid_t, MPI_SUM, MPI_COMM_WORLD);
    assert(edges_from_v + edges_from_h == edges);

    // locality-aware chunking
    // verticesとhyperedgesで別々に行う
    // 結局その方が、process_edges系のapiはHyperedgeとそこから出る辺、Vertexとそこから出る辺とは別々に
    // 扱うから、その時にいいのでは？と言うのと、一緒くたにすると発生するextraなけいさん達の存在もある
    partition_offset_v = new VertexId [partitions + 1];
    partition_offset_h = new HyperedgeId [partitions + 1];

    owned_vertices   = set_partition_offset(partition_offset_v, partitions, vertices,   out_degree_v, edges_from_v);
    owned_hyperedges = set_partition_offset(partition_offset_h, partitions, hyperedges, out_degree_h, edges_from_h);

    // check consistency of partition boundaries
    VertexId * global_partition_offset_v = new VertexId [partitions + 1];
    HyperedgeId * global_partition_offset_h = new HyperedgeId [partitions + 1];
    MPI_Allreduce(partition_offset_v, global_partition_offset_v, partitions + 1, vid_t, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(partition_offset_h, global_partition_offset_h, partitions + 1, vid_t, MPI_MAX, MPI_COMM_WORLD);
    for (int i=0;i<=partitions;i++) {
      assert(partition_offset_v[i] == global_partition_offset_v[i]);
      assert(partition_offset_h[i] == global_partition_offset_h[i]);
    }

    delete [] global_partition_offset_v;
    delete [] global_partition_offset_h;

    // #ifdef PRINT_DEBUG_MESSAGES
    // // TODO: hypergraphむけに書き換える
    // if (partition_id==0) {
    //   for (int i=0;i<partitions;i++) {
    //     BipartiteEdgeId part_out_edges = 0;
    //     for (VertexId v_i=partition_offset_v[i];v_i<partition_offset_v[i+1];v_i++) {
    //       part_out_edges += out_degree_v[v_i];
    //     }
    //     printf("|V'_%d| = %u |E_%d| = %lu\n", i, partition_offset_v[i+1] - partition_offset_v[i], i, part_out_edges);
    //   }
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // #endif
    {
      // NUMA-aware sub-chunking
      local_partition_offset_v = new VertexId [sockets + 1];
      local_partition_offset_h = new HyperedgeId [sockets + 1];
      BipartiteEdgeId part_out_edges_v = 0;
      BipartiteEdgeId part_out_edges_h = 0;
      for (VertexId v_i=partition_offset_v[partition_id];v_i<partition_offset_v[partition_id+1];v_i++) {
        part_out_edges_v += out_degree_v[v_i];
      }
      for (HyperedgeId h_i=partition_offset_h[partition_id];h_i<partition_offset_h[partition_id+1];h_i++) {
        part_out_edges_h += out_degree_h[h_i];
      }
      
      set_partition_offset(
        local_partition_offset_v, sockets, owned_vertices, 
        out_degree_v, part_out_edges_v, partition_offset_v[partition_id], partition_offset_v[partition_id+1]);
      set_partition_offset(
        local_partition_offset_h, sockets, owned_hyperedges, 
        out_degree_h, part_out_edges_h, partition_offset_h[partition_id], partition_offset_h[partition_id+1]);
    }

    HyperedgeId * filtered_out_degree_v = alloc_vertex_array<HyperedgeId>();
    VertexId * filtered_out_degree_h = alloc_hyperedge_array<VertexId>();
    for (VertexId v_i=partition_offset_v[partition_id];v_i<partition_offset_v[partition_id+1];v_i++) {
      filtered_out_degree_v[v_i] = out_degree_v[v_i];
    }
    for (HyperedgeId h_i=partition_offset_h[partition_id];h_i<partition_offset_h[partition_id+1];h_i++) {
      filtered_out_degree_h[h_i] = out_degree_h[h_i];
    }
    numa_free(out_degree_v, sizeof(HyperedgeId) * vertices);
    numa_free(out_degree_h, sizeof(VertexId) * hyperedges);

    out_degree_v = filtered_out_degree_v;
    out_degree_h = filtered_out_degree_h;
    in_degree_v = out_degree_v;
    in_degree_h = out_degree_h;

    int * buffered_edges = new int [partitions];
    std::vector<char> * send_buffer = new std::vector<char> [partitions];
    for (int i=0;i<partitions;i++) {
      send_buffer[i].resize(edge_unit_size * CHUNKSIZE);
    }
    EdgeUnit<EdgeData> * recv_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE];

    // constructing symmetric edges
    BipartiteEdgeId recv_outgoing_edges = 0;
    outgoing_edges_v = new BipartiteEdgeId [sockets];
    outgoing_edges_h = new BipartiteEdgeId [sockets];
    outgoing_adj_index_v = new BipartiteEdgeId* [sockets];
    outgoing_adj_index_h = new BipartiteEdgeId* [sockets];
    outgoing_adj_list_v = new AdjUnit<EdgeData>* [sockets];
    outgoing_adj_list_h = new AdjUnit<EdgeData>* [sockets];
    outgoing_adj_bitmap_v = new Bitmap * [sockets];
    outgoing_adj_bitmap_h = new Bitmap * [sockets];
    for (int s_i=0;s_i<sockets;s_i++) {
      outgoing_adj_bitmap_v[s_i] = new Bitmap (vertices);
      outgoing_adj_bitmap_h[s_i] = new Bitmap (hyperedges);
      outgoing_adj_bitmap_v[s_i]->clear();
      outgoing_adj_bitmap_h[s_i]->clear();
      outgoing_adj_index_v[s_i] = (BipartiteEdgeId*)numa_alloc_onnode(sizeof(BipartiteEdgeId) * (vertices+1), s_i);
      outgoing_adj_index_h[s_i] = (BipartiteEdgeId*)numa_alloc_onnode(sizeof(BipartiteEdgeId) * (hyperedges+1), s_i);
    }
    {
      std::thread recv_thread_dst([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // #pragma omp parallel for
          for (BipartiteEdgeId e_i=0;e_i<recv_edges;e_i++) {
            BipartiteVertexId src = recv_buffer[e_i].src;
            BipartiteVertexId dst = recv_buffer[e_i].dst;
            if (src < vertices) {
              // vertex -> hyperedge
              dst -= vertices;
              assert(dst >= partition_offset_h[partition_id] && dst < partition_offset_h[partition_id+1]);
              int dst_part = get_local_partition_id_h(dst);
              if (!outgoing_adj_bitmap_v[dst_part]->get_bit(src)) {
                outgoing_adj_bitmap_v[dst_part]->set_bit(src);
                outgoing_adj_index_v[dst_part][src] = 0;
              }
              __sync_fetch_and_add(&outgoing_adj_index_v[dst_part][src], 1);
            } else {
              // hyperedge -> vertex
              src -= vertices;
              assert(dst >= partition_offset_v[partition_id] && dst < partition_offset_v[partition_id+1]);
              int dst_part = get_local_partition_id_v(dst);
              if (!outgoing_adj_bitmap_h[dst_part]->get_bit(src)) {
                outgoing_adj_bitmap_h[dst_part]->set_bit(src);
                outgoing_adj_index_h[dst_part][src] = 0;
              }
              __sync_fetch_and_add(&outgoing_adj_index_h[dst_part][src], 1);
            }
          }
          recv_outgoing_edges += recv_edges;
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        BipartiteEdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (BipartiteEdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          BipartiteVertexId dst = read_edge_buffer[e_i].dst;
          int i = dst < vertices ? get_partition_id_v(dst) : get_partition_id_h(dst-vertices);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
        for (BipartiteEdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          // std::swap(read_edge_buffer[e_i].src, read_edge_buffer[e_i].dst);
          BipartiteVertexId tmp = read_edge_buffer[e_i].src;
          read_edge_buffer[e_i].src = read_edge_buffer[e_i].dst;
          read_edge_buffer[e_i].dst = tmp;
        }
        for (BipartiteEdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          BipartiteVertexId dst = read_edge_buffer[e_i].dst;
          int i = dst < vertices ? get_partition_id_v(dst) : get_partition_id_h(dst-vertices);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
      #ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu symmetric edges\n", partition_id, recv_outgoing_edges);
      #endif
    }
    compressed_outgoing_adj_vertices = new VertexId [sockets];
    compressed_outgoing_adj_hyperedges = new HyperedgeId [sockets];
    compressed_outgoing_adj_index_v = new CompressedAdjIndexUnit * [sockets];
    compressed_outgoing_adj_index_h = new CompressedAdjIndexUnit * [sockets];
    
    set_compressed_outgoing_adj_datas_v();
    set_compressed_outgoing_adj_datas_h();

    for (int s_i=0;s_i<sockets;s_i++) {
      // #ifdef PRINT_DEBUG_MESSAGES
      // printf("part(%d) E_%d has %lu symmetric edges\n", partition_id, s_i, outgoing_edges[s_i]);
      // #endif
      outgoing_adj_list_v[s_i] = (AdjUnit<EdgeData>*)numa_alloc_onnode(unit_size * outgoing_edges_v[s_i], s_i);
      outgoing_adj_list_h[s_i] = (AdjUnit<EdgeData>*)numa_alloc_onnode(unit_size * outgoing_edges_h[s_i], s_i);
    }
    {
      std::thread recv_thread_dst([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          #pragma omp parallel for
          for (BipartiteEdgeId e_i=0;e_i<recv_edges;e_i++) {
            BipartiteVertexId src = recv_buffer[e_i].src;
            BipartiteVertexId dst = recv_buffer[e_i].dst;
            if (src < vertices) {
              // vertex -> hyperedge
              dst -= vertices;
              assert(dst >= partition_offset_h[partition_id] && dst < partition_offset_h[partition_id+1]);
              int dst_part = get_local_partition_id_h(dst);
              BipartiteEdgeId pos = __sync_fetch_and_add(&outgoing_adj_index_v[dst_part][src], 1);
              outgoing_adj_list_v[dst_part][pos].neighbour = dst;
              if (!std::is_same<EdgeData, Empty>::value) {
                outgoing_adj_list_v[dst_part][pos].edge_data = recv_buffer[e_i].edge_data;
              }
            } else {
              // hyperedge -> vertex
              src -= vertices;
              assert(dst >= partition_offset_v[partition_id] && dst < partition_offset_v[partition_id+1]);
              int dst_part = get_local_partition_id_v(dst);
              BipartiteEdgeId pos = __sync_fetch_and_add(&outgoing_adj_index_h[dst_part][src], 1);
              outgoing_adj_list_h[dst_part][pos].neighbour = dst;
              if (!std::is_same<EdgeData, Empty>::value) {
                outgoing_adj_list_h[dst_part][pos].edge_data = recv_buffer[e_i].edge_data;
              }
            }
          }
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        BipartiteEdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (BipartiteEdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          BipartiteVertexId dst = read_edge_buffer[e_i].dst;
          int i = dst < vertices ? get_partition_id_v(dst) : get_partition_id_h(dst-vertices);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
        for (BipartiteEdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          // std::swap(read_edge_buffer[e_i].src, read_edge_buffer[e_i].dst);
          BipartiteVertexId tmp = read_edge_buffer[e_i].src;
          read_edge_buffer[e_i].src = read_edge_buffer[e_i].dst;
          read_edge_buffer[e_i].dst = tmp;
        }
        for (BipartiteEdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          BipartiteVertexId dst = read_edge_buffer[e_i].dst;
          int i = dst < vertices ? get_partition_id_v(dst) : get_partition_id_h(dst-vertices);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
    }
    for (int s_i=0;s_i<sockets;s_i++) {
      for (VertexId p_v_i=0;p_v_i<compressed_outgoing_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_outgoing_adj_index_v[s_i][p_v_i].vertex;
        outgoing_adj_index_v[s_i][v_i] = compressed_outgoing_adj_index_v[s_i][p_v_i].index;
        outgoing_adj_index_v[s_i][v_i+1] = compressed_outgoing_adj_index_v[s_i][p_v_i+1].index;
      }
      for (HyperedgeId p_h_i=0;p_h_i<compressed_outgoing_adj_hyperedges[s_i];p_h_i++) {
        HyperedgeId h_i = compressed_outgoing_adj_index_h[s_i][p_h_i].vertex;
        outgoing_adj_index_h[s_i][h_i] = compressed_outgoing_adj_index_h[s_i][p_h_i].index;
        outgoing_adj_index_h[s_i][h_i+1] = compressed_outgoing_adj_index_h[s_i][p_h_i+1].index;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    incoming_edges_v = outgoing_edges_v;
    incoming_adj_index_v = outgoing_adj_index_v;
    incoming_adj_list_v = outgoing_adj_list_v;
    incoming_adj_bitmap_v = outgoing_adj_bitmap_v;
    compressed_incoming_adj_vertices = compressed_outgoing_adj_vertices;
    compressed_incoming_adj_index_v = compressed_outgoing_adj_index_v;

    incoming_edges_h = outgoing_edges_h;
    incoming_adj_index_h = outgoing_adj_index_h;
    incoming_adj_list_h = outgoing_adj_list_h;
    incoming_adj_bitmap_h = outgoing_adj_bitmap_h;
    compressed_incoming_adj_hyperedges = compressed_outgoing_adj_hyperedges;
    compressed_incoming_adj_index_h = compressed_outgoing_adj_index_h;

    MPI_Barrier(MPI_COMM_WORLD);

    delete [] buffered_edges;
    delete [] send_buffer;
    delete [] read_edge_buffer;
    delete [] recv_buffer;
    close(fin);

    tune_chunks();
    tuned_vertex_chunks_sparse = tuned_vertex_chunks_dense;
    tuned_hyperedge_chunks_sparse = tuned_hyperedge_chunks_dense;

    prep_time += MPI_Wtime();

    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("preprocessing cost: %.2lf (s)\n", prep_time);
    }
    #endif
  }

  // reverse all the bipartite edges
  void transpose() {
    std::swap(out_degree_v, in_degree_v);
    std::swap(outgoing_edges_v, incoming_edges_v);
    std::swap(outgoing_adj_index_v, incoming_adj_index_v);
    std::swap(outgoing_adj_bitmap_v, incoming_adj_bitmap_v);
    std::swap(outgoing_adj_list_v, incoming_adj_list_v);
    std::swap(tuned_vertex_chunks_dense, tuned_vertex_chunks_sparse);
    std::swap(compressed_outgoing_adj_vertices, compressed_incoming_adj_vertices);
    std::swap(compressed_outgoing_adj_index_v, compressed_incoming_adj_index_v);

    std::swap(out_degree_h, in_degree_h);
    std::swap(outgoing_edges_h, incoming_edges_h);
    std::swap(outgoing_adj_index_h, incoming_adj_index_h);
    std::swap(outgoing_adj_bitmap_h, incoming_adj_bitmap_h);
    std::swap(outgoing_adj_list_h, incoming_adj_list_h);
    std::swap(tuned_hyperedge_chunks_dense, tuned_hyperedge_chunks_sparse);
    std::swap(compressed_outgoing_adj_hyperedges, compressed_incoming_adj_hyperedges);
    std::swap(compressed_outgoing_adj_index_h, compressed_incoming_adj_index_h);
  }

  // load a directed graph from path
  void load_directed(std::string path, VertexId vertices, HyperedgeId hyperedges) {
    double prep_time = 0;
    prep_time -= MPI_Wtime();

    symmetric = false;

    MPI_Datatype vid_t = get_mpi_data_type<BipartiteVertexId>();
    MPI_Datatype eid_t = get_mpi_data_type<BipartiteEdgeId>();

    this->vertices = vertices;
    this->hyperedges = hyperedges;
    this->bipartite_vertices = vertices + hyperedges;
    long total_bytes = file_size(path.c_str());
    this->edges = total_bytes / edge_unit_size;
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("|V| = %u, |E| = %lu\n", vertices, edges);
    }
    #endif

    BipartiteEdgeId read_edges = edges / partitions;
    if (partition_id==partitions-1) {
      read_edges += edges % partitions;
    }
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = edge_unit_size * (edges / partitions * partition_id);
    long read_bytes;
    int fin = open(path.c_str(), O_RDONLY);
    EdgeUnit<EdgeData> * read_edge_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE];

    out_degree_v = alloc_interleaved_vertex_array<HyperedgeId>();
    out_degree_h = alloc_interleaved_hyperedge_array<VertexId>();
    for (VertexId v_i=0;v_i<vertices;v_i++) {
      out_degree_v[v_i] = 0;
    }
    for (HyperedgeId h_i=0;h_i<hyperedges;h_i++) {
      out_degree_h[h_i] = 0;
    }
    assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
    read_bytes = 0;
    edges_from_v = edges_from_h = 0;
    while (read_bytes < bytes_to_read) {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      } else {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes>=0);
      read_bytes += curr_read_bytes;
      BipartiteEdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      #pragma omp parallel for
      for (BipartiteEdgeId e_i=0;e_i<curr_read_edges;e_i++) {
        BipartiteVertexId src = read_edge_buffer[e_i].src;
        BipartiteVertexId dst = read_edge_buffer[e_i].dst;
        auto src_degree = (src < vertices ? &out_degree_v[src] : &out_degree_h[src-vertices]);
        __sync_fetch_and_add(src_degree, 1);
        __sync_fetch_and_add((src < vertices ? &edges_from_v : &edges_from_h), 1);
      }
    }
    
    MPI_Allreduce(MPI_IN_PLACE, out_degree_v, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, out_degree_h, hyperedges, vid_t, MPI_SUM, MPI_COMM_WORLD);
    
    MPI_Allreduce(MPI_IN_PLACE, &edges_from_v, 1, eid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &edges_from_h, 1, eid_t, MPI_SUM, MPI_COMM_WORLD);
    assert(edges_from_v + edges_from_h == edges);

    // locality-aware chunking
    partition_offset_v = new VertexId [partitions + 1];
    partition_offset_h = new HyperedgeId [partitions + 1];
    
    owned_vertices   = set_partition_offset(partition_offset_v, partitions, vertices,   out_degree_v, edges_from_v);
    owned_hyperedges = set_partition_offset(partition_offset_h, partitions, hyperedges, out_degree_h, edges_from_h);

    // check consistency of partition boundaries
    VertexId * global_partition_offset_v = new VertexId [partitions + 1];
    HyperedgeId * global_partition_offset_h = new HyperedgeId [partitions + 1];
    MPI_Allreduce(partition_offset_v, global_partition_offset_v, partitions + 1, vid_t, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(partition_offset_h, global_partition_offset_h, partitions + 1, vid_t, MPI_MAX, MPI_COMM_WORLD);
    for (int i=0;i<=partitions;i++) {
      assert(partition_offset_v[i] == global_partition_offset_v[i]);
      assert(partition_offset_h[i] == global_partition_offset_h[i]);
    }
    MPI_Allreduce(partition_offset_v, global_partition_offset_v, partitions + 1, vid_t, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(partition_offset_h, global_partition_offset_h, partitions + 1, vid_t, MPI_MIN, MPI_COMM_WORLD);
    for (int i=0;i<=partitions;i++) {
      assert(partition_offset_v[i] == global_partition_offset_v[i]);
      assert(partition_offset_h[i] == global_partition_offset_h[i]);
    }

    delete [] global_partition_offset_v;
    delete [] global_partition_offset_h;

    // #ifdef PRINT_DEBUG_MESSAGES
    // if (partition_id==0) {
    //   for (int i=0;i<partitions;i++) {
    //     BipartiteEdgeId part_out_edges = 0;
    //     for (VertexId v_i=partition_offset_v[i];v_i<partition_offset_v[i+1];v_i++) {
    //       part_out_edges += out_degree_v[v_i];
    //     }
    //     printf("|V'_%d| = %u |E^dense_%d| = %lu\n", i, partition_offset_v[i+1] - partition_offset_v[i], i, part_out_edges);
    //   }
    // }
    // #endif

    {
      // NUMA-aware sub-chunking
      local_partition_offset_v = new VertexId [sockets + 1];
      local_partition_offset_h = new HyperedgeId [sockets + 1];
      BipartiteEdgeId part_out_edges_v = 0;
      BipartiteEdgeId part_out_edges_h = 0;
      for (VertexId v_i=partition_offset_v[partition_id];v_i<partition_offset_v[partition_id+1];v_i++) {
        part_out_edges_v += out_degree_v[v_i];
      }
      for (HyperedgeId h_i=partition_offset_h[partition_id];h_i<partition_offset_h[partition_id+1];h_i++) {
        part_out_edges_h += out_degree_h[h_i];
      }
      
      set_partition_offset(
        local_partition_offset_v, sockets, owned_vertices, 
        out_degree_v, part_out_edges_v, partition_offset_v[partition_id], partition_offset_v[partition_id+1]);
      set_partition_offset(
        local_partition_offset_h, sockets, owned_hyperedges, 
        out_degree_h, part_out_edges_h, partition_offset_h[partition_id], partition_offset_h[partition_id+1]);
    }

    HyperedgeId * filtered_out_degree_v = alloc_vertex_array<HyperedgeId>();
    VertexId * filtered_out_degree_h = alloc_hyperedge_array<VertexId>();
    for (VertexId v_i=partition_offset_v[partition_id];v_i<partition_offset_v[partition_id+1];v_i++) {
      filtered_out_degree_v[v_i] = out_degree_v[v_i];
    }
    for (HyperedgeId h_i=partition_offset_h[partition_id];h_i<partition_offset_h[partition_id+1];h_i++) {
      filtered_out_degree_h[h_i] = out_degree_h[h_i];
    }
    numa_free(out_degree_v, sizeof(HyperedgeId) * vertices);
    numa_free(out_degree_h, sizeof(VertexId) * hyperedges);
    
    out_degree_v = filtered_out_degree_v;
    out_degree_h = filtered_out_degree_h;
    in_degree_v = alloc_vertex_array<HyperedgeId>();
    in_degree_h = alloc_hyperedge_array<VertexId>();
    for (VertexId v_i=partition_offset_v[partition_id];v_i<partition_offset_v[partition_id+1];v_i++) {
      in_degree_v[v_i] = 0;
    }
    for (HyperedgeId h_i=partition_offset_h[partition_id];h_i<partition_offset_h[partition_id+1];h_i++) {
      in_degree_h[h_i] = 0;
    }

    int * buffered_edges = new int [partitions];
    std::vector<char> * send_buffer = new std::vector<char> [partitions];
    for (int i=0;i<partitions;i++) {
      send_buffer[i].resize(edge_unit_size * CHUNKSIZE);
    }
    EdgeUnit<EdgeData> * recv_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE];

    BipartiteEdgeId recv_outgoing_edges = 0;
    outgoing_edges_v = new BipartiteEdgeId [sockets];
    outgoing_edges_h = new BipartiteEdgeId [sockets];
    outgoing_adj_index_v = new BipartiteEdgeId* [sockets];
    outgoing_adj_index_h = new BipartiteEdgeId* [sockets];
    outgoing_adj_list_v = new AdjUnit<EdgeData>* [sockets];
    outgoing_adj_list_h = new AdjUnit<EdgeData>* [sockets];
    outgoing_adj_bitmap_v = new Bitmap * [sockets];
    outgoing_adj_bitmap_h = new Bitmap * [sockets];
    for (int s_i=0;s_i<sockets;s_i++) {
      outgoing_adj_bitmap_v[s_i] = new Bitmap (vertices);
      outgoing_adj_bitmap_h[s_i] = new Bitmap (hyperedges);
      outgoing_adj_bitmap_v[s_i]->clear();
      outgoing_adj_bitmap_h[s_i]->clear();
      outgoing_adj_index_v[s_i] = (BipartiteEdgeId*)numa_alloc_onnode(sizeof(BipartiteEdgeId) * (vertices+1), s_i);
      outgoing_adj_index_h[s_i] = (BipartiteEdgeId*)numa_alloc_onnode(sizeof(BipartiteEdgeId) * (hyperedges+1), s_i);
    }
    {
      std::thread recv_thread_dst([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // #pragma omp parallel for
          for (BipartiteEdgeId e_i=0;e_i<recv_edges;e_i++) {
            BipartiteVertexId src = recv_buffer[e_i].src;
            BipartiteVertexId dst = recv_buffer[e_i].dst;
            if (src < vertices) {
              // vertex -> hyperedge
              dst -= vertices;
              assert(dst >= partition_offset_h[partition_id] && dst < partition_offset_h[partition_id+1]);
              assert(0 <= src && src < vertices);
              int dst_part = get_local_partition_id_h(dst);
              if (!outgoing_adj_bitmap_v[dst_part]->get_bit(src)) {
                outgoing_adj_bitmap_v[dst_part]->set_bit(src);
                outgoing_adj_index_v[dst_part][src] = 0;
              }
              __sync_fetch_and_add(&outgoing_adj_index_v[dst_part][src], 1);
              __sync_fetch_and_add(&in_degree_h[dst], 1);
            } else {
              src -= vertices;
              // hyperedge -> vertex
              assert(dst >= partition_offset_v[partition_id] && dst < partition_offset_v[partition_id+1]);
              assert(0 <= src && src < hyperedges);
              int dst_part = get_local_partition_id_v(dst);
              if (!outgoing_adj_bitmap_h[dst_part]->get_bit(src)) {
                outgoing_adj_bitmap_h[dst_part]->set_bit(src);
                outgoing_adj_index_h[dst_part][src] = 0;
              }
              __sync_fetch_and_add(&outgoing_adj_index_h[dst_part][src], 1);
              __sync_fetch_and_add(&in_degree_v[dst], 1);
            }
          }
          recv_outgoing_edges += recv_edges;
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        BipartiteEdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (BipartiteEdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          BipartiteVertexId dst = read_edge_buffer[e_i].dst;
          int i = dst < vertices ? get_partition_id_v(dst) : get_partition_id_h(dst-vertices);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
      #ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu sparse mode edges\n", partition_id, recv_outgoing_edges);
      #endif
    }
    
    compressed_outgoing_adj_vertices = new VertexId [sockets];
    compressed_outgoing_adj_hyperedges = new HyperedgeId [sockets];
    compressed_outgoing_adj_index_v = new CompressedAdjIndexUnit * [sockets];
    compressed_outgoing_adj_index_h = new CompressedAdjIndexUnit * [sockets];
    
    set_compressed_outgoing_adj_datas_v();
    set_compressed_outgoing_adj_datas_h();

    for (int s_i=0;s_i<sockets;s_i++) {
      // #ifdef PRINT_DEBUG_MESSAGES
      // printf("part(%d) E_%d has %lu sparse mode edges\n", partition_id, s_i, outgoing_edges[s_i]);
      // #endif
      outgoing_adj_list_v[s_i] = (AdjUnit<EdgeData>*)numa_alloc_onnode(unit_size * outgoing_edges_v[s_i], s_i);
      outgoing_adj_list_h[s_i] = (AdjUnit<EdgeData>*)numa_alloc_onnode(unit_size * outgoing_edges_h[s_i], s_i);
    }

    {
      std::thread recv_thread_dst([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          #pragma omp parallel for
          for (BipartiteEdgeId e_i=0;e_i<recv_edges;e_i++) {
            BipartiteVertexId src = recv_buffer[e_i].src;
            BipartiteVertexId dst = recv_buffer[e_i].dst;
            if (src < vertices) {
              // vertex -> hyperedge
              dst -= vertices;
              assert(dst >= partition_offset_h[partition_id] && dst < partition_offset_h[partition_id+1]);
              assert(0 <= src && src < vertices);
              int dst_part = get_local_partition_id_h(dst);
              BipartiteEdgeId pos = __sync_fetch_and_add(&outgoing_adj_index_v[dst_part][src], 1);
              outgoing_adj_list_v[dst_part][pos].neighbour = dst;
              if (!std::is_same<EdgeData, Empty>::value) {
                outgoing_adj_list_v[dst_part][pos].edge_data = recv_buffer[e_i].edge_data;
              }
            } else {
              // hyperedge -> vertex
              src -= vertices;
              assert(dst >= partition_offset_v[partition_id] && dst < partition_offset_v[partition_id+1]);
              assert(0 <= src && src < hyperedges);
              int dst_part = get_local_partition_id_v(dst);
              BipartiteEdgeId pos = __sync_fetch_and_add(&outgoing_adj_index_h[dst_part][src], 1);
              outgoing_adj_list_h[dst_part][pos].neighbour = dst;
              if (!std::is_same<EdgeData, Empty>::value) {
                outgoing_adj_list_h[dst_part][pos].edge_data = recv_buffer[e_i].edge_data;
              }
            }
          }
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        BipartiteEdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (BipartiteEdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          BipartiteVertexId dst = read_edge_buffer[e_i].dst;
          int i = dst < vertices ? get_partition_id_v(dst) : get_partition_id_h(dst-vertices);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
    }
    for (int s_i=0;s_i<sockets;s_i++) {
      for (VertexId p_v_i=0;p_v_i<compressed_outgoing_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_outgoing_adj_index_v[s_i][p_v_i].vertex;
        outgoing_adj_index_v[s_i][v_i] = compressed_outgoing_adj_index_v[s_i][p_v_i].index;
        outgoing_adj_index_v[s_i][v_i+1] = compressed_outgoing_adj_index_v[s_i][p_v_i+1].index;
      }
      for (HyperedgeId p_h_i=0;p_h_i<compressed_outgoing_adj_hyperedges[s_i];p_h_i++) {
        HyperedgeId h_i = compressed_outgoing_adj_index_h[s_i][p_h_i].vertex;
        outgoing_adj_index_h[s_i][h_i] = compressed_outgoing_adj_index_h[s_i][p_h_i].index;
        outgoing_adj_index_h[s_i][h_i+1] = compressed_outgoing_adj_index_h[s_i][p_h_i+1].index;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    BipartiteEdgeId recv_incoming_edges = 0;
    incoming_edges_v = new BipartiteEdgeId [sockets];
    incoming_edges_h = new BipartiteEdgeId [sockets];
    incoming_adj_index_v = new BipartiteEdgeId* [sockets];
    incoming_adj_index_h = new BipartiteEdgeId* [sockets];
    incoming_adj_list_v = new AdjUnit<EdgeData>* [sockets];
    incoming_adj_list_h = new AdjUnit<EdgeData>* [sockets];
    incoming_adj_bitmap_v = new Bitmap * [sockets];
    incoming_adj_bitmap_h = new Bitmap * [sockets];
    for (int s_i=0;s_i<sockets;s_i++) {
      incoming_adj_bitmap_v[s_i] = new Bitmap (vertices);
      incoming_adj_bitmap_h[s_i] = new Bitmap (hyperedges);
      incoming_adj_bitmap_v[s_i]->clear();
      incoming_adj_bitmap_h[s_i]->clear();
      incoming_adj_index_v[s_i] = (BipartiteEdgeId*)numa_alloc_onnode(sizeof(BipartiteEdgeId) * (vertices+1), s_i);
      incoming_adj_index_h[s_i] = (BipartiteEdgeId*)numa_alloc_onnode(sizeof(BipartiteEdgeId) * (hyperedges+1), s_i);
    }
    {
      std::thread recv_thread_src([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // #pragma omp parallel for
          for (BipartiteEdgeId e_i=0;e_i<recv_edges;e_i++) {
            BipartiteVertexId src = recv_buffer[e_i].src;
            BipartiteVertexId dst = recv_buffer[e_i].dst;
            if (src < vertices) {
              // vertex -> hyperedge
              dst -= vertices;
              assert(src >= partition_offset_v[partition_id] && src < partition_offset_v[partition_id+1]);
              int src_part = get_local_partition_id_v(src);
              if (!incoming_adj_bitmap_h[src_part]->get_bit(dst)) {
                incoming_adj_bitmap_h[src_part]->set_bit(dst);
                incoming_adj_index_h[src_part][dst] = 0;
              }
              __sync_fetch_and_add(&incoming_adj_index_h[src_part][dst], 1);
            } else {
              // hyperedge -> vertex
              src -= vertices;
              assert(src >= partition_offset_h[partition_id] && src < partition_offset_h[partition_id+1]);
              int src_part = get_local_partition_id_h(src);
              if (!incoming_adj_bitmap_v[src_part]->get_bit(dst)) {
                incoming_adj_bitmap_v[src_part]->set_bit(dst);
                incoming_adj_index_v[src_part][dst] = 0;
              }
              __sync_fetch_and_add(&incoming_adj_index_v[src_part][dst], 1);
            }
          }
          recv_incoming_edges += recv_edges;
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        BipartiteEdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (BipartiteEdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          BipartiteVertexId src = read_edge_buffer[e_i].src;
          int i = src < vertices ? get_partition_id_v(src) : get_partition_id_h(src-vertices);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_src.join();
      #ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu dense mode edges\n", partition_id, recv_incoming_edges);
      #endif
    }
    compressed_incoming_adj_vertices = new VertexId [sockets];
    compressed_incoming_adj_hyperedges = new HyperedgeId [sockets];
    compressed_incoming_adj_index_v = new CompressedAdjIndexUnit * [sockets];
    compressed_incoming_adj_index_h = new CompressedAdjIndexUnit * [sockets];
    
    set_compressed_incoming_adj_datas_v();
    set_compressed_incoming_adj_datas_h();

    for (int s_i=0;s_i<sockets;s_i++) {
      // #ifdef PRINT_DEBUG_MESSAGES
      // printf("part(%d) E_%d has %lu symmetric edges\n", partition_id, s_i, outgoing_edges[s_i]);
      // #endif
      incoming_adj_list_v[s_i] = (AdjUnit<EdgeData>*)numa_alloc_onnode(unit_size * incoming_edges_v[s_i], s_i);
      incoming_adj_list_h[s_i] = (AdjUnit<EdgeData>*)numa_alloc_onnode(unit_size * incoming_edges_h[s_i], s_i);
    }
    {
      std::thread recv_thread_src([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          #pragma omp parallel for
          for (BipartiteEdgeId e_i=0;e_i<recv_edges;e_i++) {
            BipartiteVertexId src = recv_buffer[e_i].src;
            BipartiteVertexId dst = recv_buffer[e_i].dst;

            if (src < vertices) {
              // vertex -> hyperedge
              dst -= vertices;
              assert(src >= partition_offset_v[partition_id] && src < partition_offset_v[partition_id+1]);
              int src_part = get_local_partition_id_v(src);
              BipartiteEdgeId pos = __sync_fetch_and_add(&incoming_adj_index_h[src_part][dst], 1);
              incoming_adj_list_h[src_part][pos].neighbour = src;
              if (!std::is_same<EdgeData, Empty>::value) {
                incoming_adj_list_h[src_part][pos].edge_data = recv_buffer[e_i].edge_data;
              }
            } else {
              // hyperedge -> vertex
              src -= vertices;
              assert(src >= partition_offset_h[partition_id] && src < partition_offset_h[partition_id+1]);
              int src_part = get_local_partition_id_h(src);
              BipartiteEdgeId pos = __sync_fetch_and_add(&incoming_adj_index_v[src_part][dst], 1);
              incoming_adj_list_v[src_part][pos].neighbour = src;
              if (!std::is_same<EdgeData, Empty>::value) {
                incoming_adj_list_v[src_part][pos].edge_data = recv_buffer[e_i].edge_data;
              }
            }
            
          }
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        BipartiteEdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (BipartiteEdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          BipartiteVertexId src = read_edge_buffer[e_i].src;
          int i = src < vertices ? get_partition_id_v(src) : get_partition_id_h(src-vertices);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_src.join();
    }
    for (int s_i=0;s_i<sockets;s_i++) {
      for (VertexId p_v_i=0;p_v_i<compressed_incoming_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_incoming_adj_index_v[s_i][p_v_i].vertex;
        incoming_adj_index_v[s_i][v_i] = compressed_incoming_adj_index_v[s_i][p_v_i].index;
        incoming_adj_index_v[s_i][v_i+1] = compressed_incoming_adj_index_v[s_i][p_v_i+1].index;
      }
      for (HyperedgeId p_h_i=0;p_h_i<compressed_incoming_adj_hyperedges[s_i];p_h_i++) {
        HyperedgeId h_i = compressed_incoming_adj_index_h[s_i][p_h_i].vertex;
        incoming_adj_index_h[s_i][h_i] = compressed_incoming_adj_index_h[s_i][p_h_i].index;
        incoming_adj_index_h[s_i][h_i+1] = compressed_incoming_adj_index_h[s_i][p_h_i+1].index;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    delete [] buffered_edges;
    delete [] send_buffer;
    delete [] read_edge_buffer;
    delete [] recv_buffer;
    close(fin);

    transpose();
    tune_chunks();
    transpose();
    tune_chunks();

    prep_time += MPI_Wtime();

    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("preprocessing cost: %.2lf (s)\n", prep_time);
    }
    #endif
  }

  void tune_chunks() {
    tune_vertex_chunks();
    tune_hyperedge_chunks();
  }
  void tune_vertex_chunks() {
    tuned_vertex_chunks_dense = new ThreadState * [partitions];
    int current_send_part_id = partition_id;
    for (int step=0;step<partitions;step++) {
      current_send_part_id = (current_send_part_id + 1) % partitions;
      int i = current_send_part_id;
      tuned_vertex_chunks_dense[i] = new ThreadState [threads];
      BipartiteEdgeId remained_edges;
      int remained_partitions;
      VertexId last_p_v_i;
      VertexId end_p_v_i;
      for (int t_i=0;t_i<threads;t_i++) {
        tuned_vertex_chunks_dense[i][t_i].status = WORKING;
        int s_i = get_socket_id(t_i);
        int s_j = get_socket_offset(t_i);
        if (s_j==0) {
          VertexId p_v_i = 0;
          while (p_v_i<compressed_incoming_adj_vertices[s_i]) {
            VertexId v_i = compressed_incoming_adj_index_v[s_i][p_v_i].vertex;
            if (v_i >= partition_offset_v[i]) {
              break;
            }
            p_v_i++;
          }
          last_p_v_i = p_v_i;
          while (p_v_i<compressed_incoming_adj_vertices[s_i]) {
            VertexId v_i = compressed_incoming_adj_index_v[s_i][p_v_i].vertex;
            if (v_i >= partition_offset_v[i+1]) {
              break;
            }
            p_v_i++;
          }
          end_p_v_i = p_v_i;
          remained_edges = 0;
          for (VertexId p_v_i=last_p_v_i;p_v_i<end_p_v_i;p_v_i++) {
            remained_edges += compressed_incoming_adj_index_v[s_i][p_v_i+1].index - compressed_incoming_adj_index_v[s_i][p_v_i].index;
            remained_edges += alpha;
          }
        }
        tuned_vertex_chunks_dense[i][t_i].curr = last_p_v_i;
        tuned_vertex_chunks_dense[i][t_i].end = last_p_v_i;
        remained_partitions = threads_per_socket - s_j;
        BipartiteEdgeId expected_chunk_size = remained_edges / remained_partitions;
        if (remained_partitions==1) {
          tuned_vertex_chunks_dense[i][t_i].end = end_p_v_i;
        } else {
          BipartiteEdgeId got_edges = 0;
          for (VertexId p_v_i=last_p_v_i;p_v_i<end_p_v_i;p_v_i++) {
            got_edges += compressed_incoming_adj_index_v[s_i][p_v_i+1].index - compressed_incoming_adj_index_v[s_i][p_v_i].index + alpha;
            if (got_edges >= expected_chunk_size) {
              tuned_vertex_chunks_dense[i][t_i].end = p_v_i;
              last_p_v_i = tuned_vertex_chunks_dense[i][t_i].end;
              break;
            }
          }
          got_edges = 0;
          for (VertexId p_v_i=tuned_vertex_chunks_dense[i][t_i].curr;p_v_i<tuned_vertex_chunks_dense[i][t_i].end;p_v_i++) {
            got_edges += compressed_incoming_adj_index_v[s_i][p_v_i+1].index - compressed_incoming_adj_index_v[s_i][p_v_i].index + alpha;
          }
          remained_edges -= got_edges;
        }
      }
    }
  }
  void tune_hyperedge_chunks() {
    flip_bipartite_graph();
    tune_vertex_chunks();
    flip_bipartite_graph();
  }

  // process vertices
  template<typename R>
  R process_vertices(std::function<R(VertexId)> process, Bitmap * active) {
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    R reducer = 0;
    size_t basic_chunk = 64;
    for (int t_i=0;t_i<threads;t_i++) {
      int s_i = get_socket_id(t_i);
      int s_j = get_socket_offset(t_i);
      VertexId partition_size = local_partition_offset_v[s_i+1] - local_partition_offset_v[s_i];
      thread_state[t_i]->curr = local_partition_offset_v[s_i] + partition_size / threads_per_socket  / basic_chunk * basic_chunk * s_j;
      thread_state[t_i]->end = local_partition_offset_v[s_i] + partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j+1);
      if (s_j == threads_per_socket - 1) {
        thread_state[t_i]->end = local_partition_offset_v[s_i+1];
      }
      thread_state[t_i]->status = WORKING;
    }
    #pragma omp parallel reduction(+:reducer)
    {
      R local_reducer = 0;
      int thread_id = omp_get_thread_num();
      while (true) {
        VertexId bv_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
        if (bv_i >= thread_state[thread_id]->end) break;
        unsigned long word = active->data[WORD_OFFSET(bv_i)];
        while (word != 0) {
          if (word & 1) {
            local_reducer += process(bv_i);
          }
          bv_i++;
          word = word >> 1;
        }
      }
      thread_state[thread_id]->status = STEALING;
      for (int t_offset=1;t_offset<threads;t_offset++) {
        int t_i = (thread_id + t_offset) % threads;
        while (thread_state[t_i]->status!=STEALING) {
          VertexId bv_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
          if (bv_i >= thread_state[t_i]->end) continue;
          unsigned long word = active->data[WORD_OFFSET(bv_i)];
          while (word != 0) {
            if (word & 1) {
              local_reducer += process(bv_i);
            }
            bv_i++;
            word = word >> 1;
          }
        }
      }
      reducer += local_reducer;
    }
    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("process_vertices took %lf (s)\n", stream_time);
    }
    #endif
    return global_reducer;
  }
  // process hyperedges
  template<typename R>
  R process_hyperedges(std::function<R(HyperedgeId)> process, Bitmap * active) {
    flip_bipartite_graph(false);
    R ret_val = process_vertices(process, active);
    flip_bipartite_graph(false);
    return ret_val;
  }

  template<typename M>
  void flush_local_send_buffer(int t_i) {
    int s_i = get_socket_id(t_i);
    int pos = __sync_fetch_and_add(&send_buffer[current_send_part_id][s_i]->count, local_send_buffer[t_i]->count);
    memcpy(send_buffer[current_send_part_id][s_i]->data + sizeof(MsgUnit<M>) * pos, local_send_buffer[t_i]->data, sizeof(MsgUnit<M>) * local_send_buffer[t_i]->count);
    local_send_buffer[t_i]->count = 0;
  }

  // emit a message to a vertex's master (dense) / mirror (sparse)
  template<typename M>
  void emit(VertexId vtx, M msg) {
    int t_i = omp_get_thread_num();
    MsgUnit<M> * buffer = (MsgUnit<M>*)local_send_buffer[t_i]->data;
    buffer[local_send_buffer[t_i]->count].vertex = vtx;
    buffer[local_send_buffer[t_i]->count].msg_data = msg;
    local_send_buffer[t_i]->count += 1;
    if (local_send_buffer[t_i]->count==local_send_buffer_limit) {
      flush_local_send_buffer<M>(t_i);
    }
  }


  // process edges
  template<typename R, typename M>
  R prop_from_vertices(std::function<void(VertexId)> sparse_signal, std::function<R(VertexId, M, VertexAdjList<EdgeData>)> sparse_slot, std::function<void(HyperedgeId, VertexAdjList<EdgeData>)> dense_signal, std::function<R(HyperedgeId, M)> dense_slot, Bitmap * active, Bitmap * dense_selective = nullptr) {
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    for (int t_i=0;t_i<threads;t_i++) {
      local_send_buffer[t_i]->resize( sizeof(MsgUnit<M>) * local_send_buffer_limit );
      local_send_buffer[t_i]->count = 0;
    }
    R reducer = 0;
    auto out_degree = flipped ? out_degree_h : out_degree_v;
    BipartiteEdgeId active_edges = process_vertices<BipartiteEdgeId>(
      [&](VertexId vtx){
        return (BipartiteEdgeId)out_degree[vtx];
      },
      active
    );
    bool sparse = (active_edges < edges / 20);
    if (sparse) {
      for (int i=0;i<partitions;i++) {
        for (int s_i=0;s_i<sockets;s_i++) {
          recv_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * (partition_offset_v[i+1] - partition_offset_v[i]) * sockets );
          send_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * owned_vertices * sockets );
          send_buffer[i][s_i]->count = 0;
          recv_buffer[i][s_i]->count = 0;
        }
      }
    } else {
      for (int i=0;i<partitions;i++) {
        for (int s_i=0;s_i<sockets;s_i++) {
          recv_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * owned_hyperedges * sockets );
          send_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * (partition_offset_h[i+1] - partition_offset_h[i]) * sockets );
          send_buffer[i][s_i]->count = 0;
          recv_buffer[i][s_i]->count = 0;
        }
      }
    }
    size_t basic_chunk = 64;
    if (sparse) {
      #ifdef PRINT_DEBUG_MESSAGES
      if (partition_id==0) {
        printf("sparse mode\n");
      }
      #endif
      int * recv_queue = new int [partitions];
      int recv_queue_size = 0;
      std::mutex recv_queue_mutex;

      current_send_part_id = partition_id;
      #pragma omp parallel for
      for (VertexId begin_v_i=partition_offset_v[partition_id];begin_v_i<partition_offset_v[partition_id+1];begin_v_i+=basic_chunk) {
        VertexId v_i = begin_v_i;
        unsigned long word = active->data[WORD_OFFSET(v_i)];
        while (word != 0) {
          if (word & 1) {
            sparse_signal(v_i);
          }
          v_i++;
          word = word >> 1;
        }
      }
      #pragma omp parallel for
      for (int t_i=0;t_i<threads;t_i++) {
        flush_local_send_buffer<M>(t_i);
      }
      recv_queue[recv_queue_size] = partition_id;
      recv_queue_mutex.lock();
      recv_queue_size += 1;
      recv_queue_mutex.unlock();
      std::thread send_thread([&](){
        for (int step=1;step<partitions;step++) {
          int i = (partition_id - step + partitions) % partitions;
          for (int s_i=0;s_i<sockets;s_i++) {
            MPI_Send(send_buffer[partition_id][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[partition_id][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&](){
        for (int step=1;step<partitions;step++) {
          int i = (partition_id + step) % partitions;
          for (int s_i=0;s_i<sockets;s_i++) {
            MPI_Status recv_status;
            MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
            MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
            MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
          }
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
      });
      for (int step=0;step<partitions;step++) {
        while (true) {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size<=step);
          recv_queue_mutex.unlock();
          if (!condition) break;
          __asm volatile ("pause" ::: "memory");
        }
        int i = recv_queue[step];
        MessageBuffer ** used_buffer;
        if (i==partition_id) {
          used_buffer = send_buffer[i];
        } else {
          used_buffer = recv_buffer[i];
        }
        for (int s_i=0;s_i<sockets;s_i++) {
          MsgUnit<M> * buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          size_t buffer_size = used_buffer[s_i]->count;
          for (int t_i=0;t_i<threads;t_i++) {
            // int s_i = get_socket_id(t_i);
            int s_j = get_socket_offset(t_i);
            VertexId partition_size = buffer_size;
            thread_state[t_i]->curr = partition_size / threads_per_socket  / basic_chunk * basic_chunk * s_j;
            thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j+1);
            if (s_j == threads_per_socket - 1) {
              thread_state[t_i]->end = buffer_size;
            }
            thread_state[t_i]->status = WORKING;
          }
          #pragma omp parallel reduction(+:reducer)
          {
            R local_reducer = 0;
            int thread_id = omp_get_thread_num();
            int s_i = get_socket_id(thread_id);
            while (true) {
              VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
              if (b_i >= thread_state[thread_id]->end) break;
              VertexId begin_b_i = b_i;
              VertexId end_b_i = b_i + basic_chunk;
              if (end_b_i>thread_state[thread_id]->end) {
                end_b_i = thread_state[thread_id]->end;
              }
              for (b_i=begin_b_i;b_i<end_b_i;b_i++) {
                VertexId v_i = buffer[b_i].vertex;
                M msg_data = buffer[b_i].msg_data;
                if (outgoing_adj_bitmap_v[s_i]->get_bit(v_i)) {
                  local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list_v[s_i] + outgoing_adj_index_v[s_i][v_i], outgoing_adj_list_v[s_i] + outgoing_adj_index_v[s_i][v_i+1]));
                }
              }
            }
            thread_state[thread_id]->status = STEALING;
            for (int t_offset=1;t_offset<threads;t_offset++) {
              int t_i = (thread_id + t_offset) % threads;
              if (thread_state[t_i]->status==STEALING) continue;
              while (true) {
                VertexId b_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
                if (b_i >= thread_state[t_i]->end) break;
                VertexId begin_b_i = b_i;
                VertexId end_b_i = b_i + basic_chunk;
                if (end_b_i>thread_state[t_i]->end) {
                  end_b_i = thread_state[t_i]->end;
                }
                int s_i = get_socket_id(t_i);
                for (b_i=begin_b_i;b_i<end_b_i;b_i++) {
                  VertexId v_i = buffer[b_i].vertex;
                  M msg_data = buffer[b_i].msg_data;
                  if (outgoing_adj_bitmap_v[s_i]->get_bit(v_i)) {
                    local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list_v[s_i] + outgoing_adj_index_v[s_i][v_i], outgoing_adj_list_v[s_i] + outgoing_adj_index_v[s_i][v_i+1]));
                  }
                }
              }
            }
            reducer += local_reducer;
          }
        }
      }
      send_thread.join();
      recv_thread.join();
      delete [] recv_queue;
    } else {
      // dense selective bitmap
      if (dense_selective!=nullptr && partitions>1) {
        double sync_time = 0;
        sync_time -= get_time();
        std::thread send_thread([&](){
          for (int step=1;step<partitions;step++) {
            int recipient_id = (partition_id + step) % partitions;
            MPI_Send(dense_selective->data + WORD_OFFSET(partition_offset_h[partition_id]), owned_hyperedges / 64, MPI_UNSIGNED_LONG, recipient_id, PassMessage, MPI_COMM_WORLD);
          }
        });
        std::thread recv_thread([&](){
          for (int step=1;step<partitions;step++) {
            int sender_id = (partition_id - step + partitions) % partitions;
            MPI_Recv(dense_selective->data + WORD_OFFSET(partition_offset_h[sender_id]), (partition_offset_h[sender_id + 1] - partition_offset_h[sender_id]) / 64, MPI_UNSIGNED_LONG, sender_id, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        });
        send_thread.join();
        recv_thread.join();
        MPI_Barrier(MPI_COMM_WORLD);
        sync_time += get_time();
        #ifdef PRINT_DEBUG_MESSAGES
        if (partition_id==0) {
          printf("sync_time = %lf\n", sync_time);
        }
        #endif
      }
      #ifdef PRINT_DEBUG_MESSAGES
      if (partition_id==0) {
        printf("dense mode\n");
      }
      #endif
      int * send_queue = new int [partitions];
      int * recv_queue = new int [partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&](){
        for (int step=0;step<partitions;step++) {
          if (step==partitions-1) {
            break;
          }
          while (true) {
            send_queue_mutex.lock();
            bool condition = (send_queue_size<=step);
            send_queue_mutex.unlock();
            if (!condition) break;
            __asm volatile ("pause" ::: "memory");
          }
          int i = send_queue[step];
          for (int s_i=0;s_i<sockets;s_i++) {
            MPI_Send(send_buffer[i][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&](){
        std::vector<std::thread> threads;
        for (int step=1;step<partitions;step++) {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i){
            for (int s_i=0;s_i<sockets;s_i++) {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
            }
          }, i);
        }
        for (int step=1;step<partitions;step++) {
          int i = (partition_id - step + partitions) % partitions;
          threads[step-1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });
      current_send_part_id = partition_id;
      for (int step=0;step<partitions;step++) {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        for (int t_i=0;t_i<threads;t_i++) {
          *thread_state[t_i] = tuned_hyperedge_chunks_dense[i][t_i];
        }
        #pragma omp parallel
        {
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          HyperedgeId final_p_h_i = thread_state[thread_id]->end;
          while (true) {
            HyperedgeId begin_p_h_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_h_i >= final_p_h_i) break;
            HyperedgeId end_p_h_i = begin_p_h_i + basic_chunk;
            if (end_p_h_i > final_p_h_i) {
              end_p_h_i = final_p_h_i;
            }
            for (HyperedgeId p_h_i = begin_p_h_i; p_h_i < end_p_h_i; p_h_i ++) {
              HyperedgeId h_i = compressed_incoming_adj_index_h[s_i][p_h_i].vertex;
              dense_signal(h_i, VertexAdjList<EdgeData>(incoming_adj_list_h[s_i] + compressed_incoming_adj_index_h[s_i][p_h_i].index, incoming_adj_list_h[s_i] + compressed_incoming_adj_index_h[s_i][p_h_i+1].index));
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset=1;t_offset<threads;t_offset++) {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status!=STEALING) {
              HyperedgeId begin_p_h_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_h_i >= thread_state[t_i]->end) break;
              HyperedgeId end_p_h_i = begin_p_h_i + basic_chunk;
              if (end_p_h_i > thread_state[t_i]->end) {
                end_p_h_i = thread_state[t_i]->end;
              }
              for (HyperedgeId p_h_i = begin_p_h_i; p_h_i < end_p_h_i; p_h_i ++) {
                HyperedgeId h_i = compressed_incoming_adj_index_h[s_i][p_h_i].vertex;
                dense_signal(h_i, VertexAdjList<EdgeData>(incoming_adj_list_h[s_i] + compressed_incoming_adj_index_h[s_i][p_h_i].index, incoming_adj_list_h[s_i] + compressed_incoming_adj_index_h[s_i][p_h_i+1].index));
              }
            }
          }
        }
        #pragma omp parallel for
        for (int t_i=0;t_i<threads;t_i++) {
          flush_local_send_buffer<M>(t_i);
        }
        if (i!=partition_id) {
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
      }
      for (int step=0;step<partitions;step++) {
        while (true) {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size<=step);
          recv_queue_mutex.unlock();
          if (!condition) break;
          __asm volatile ("pause" ::: "memory");
        }
        int i = recv_queue[step];
        MessageBuffer ** used_buffer;
        if (i==partition_id) {
          used_buffer = send_buffer[i];
        } else {
          used_buffer = recv_buffer[i];
        }
        for (int t_i=0;t_i<threads;t_i++) {
          int s_i = get_socket_id(t_i);
          int s_j = get_socket_offset(t_i);
          VertexId partition_size = used_buffer[s_i]->count;
          thread_state[t_i]->curr = partition_size / threads_per_socket  / basic_chunk * basic_chunk * s_j;
          thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j+1);
          if (s_j == threads_per_socket - 1) {
            thread_state[t_i]->end = used_buffer[s_i]->count;
          }
          thread_state[t_i]->status = WORKING;
        }
        #pragma omp parallel reduction(+:reducer)
        {
          R local_reducer = 0;
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          MsgUnit<M> * buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          while (true) {
            VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (b_i >= thread_state[thread_id]->end) break;
            VertexId begin_b_i = b_i;
            VertexId end_b_i = b_i + basic_chunk;
            if (end_b_i>thread_state[thread_id]->end) {
              end_b_i = thread_state[thread_id]->end;
            }
            for (b_i=begin_b_i;b_i<end_b_i;b_i++) {
              HyperedgeId h_i = buffer[b_i].vertex;
              M msg_data = buffer[b_i].msg_data;
              local_reducer += dense_slot(h_i, msg_data);
            }
          }
          thread_state[thread_id]->status = STEALING;
          reducer += local_reducer;
        }
      }
      send_thread.join();
      recv_thread.join();
      delete [] send_queue;
      delete [] recv_queue;
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("process_edges took %lf (s)\n", stream_time);
    }
    #endif
    return global_reducer;
  }
  template<typename R, typename M>
  R prop_from_hyperedges(std::function<void(VertexId)> sparse_signal, std::function<R(VertexId, M, VertexAdjList<EdgeData>)> sparse_slot, std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal, std::function<R(VertexId, M)> dense_slot, Bitmap * active, Bitmap * dense_selective = nullptr) {
    flip_bipartite_graph(false);
    R ret_val = prop_from_vertices(sparse_signal, sparse_slot, dense_signal, dense_slot, active, dense_selective);
    flip_bipartite_graph(false);
    return ret_val;
  }

  void filter_hyperedges_from_vertices(std::function<bool(HyperedgeId)> filter, HyperedgeSubset * remaining_h, VertexSubset * active_v, HyperedgeId * remaining_out_degree_v=nullptr, HyperedgeId * remaining_in_degree_v=nullptr) {
    if (tmp_bitmap == nullptr) {
      tmp_bitmap = new Bitmap(std::max(vertices, hyperedges));
    }

    // alternative for VertexFilterNgh step1: delete hyperedge i if filter(i) == false
    Bitmap * now_deleted_h = tmp_bitmap;
    now_deleted_h->clear();

    prop_from_vertices<int, Empty>(
      [&](VertexId src) {
        emit(src, Empty());
      },
      [&](VertexId src, Empty msg, VertexAdjList<Empty> outgoing_adj) {
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          HyperedgeId dst = ptr->neighbour;
          if (!filter(dst) && remaining_h->get_bit(dst)) {
            remaining_h->reset_bit(dst);
            now_deleted_h->set_bit(dst);
          }
        }
        return 0;
      },
      [&](HyperedgeId dst, VertexAdjList<Empty> incoming_adj) {
        if (!remaining_h->get_bit(dst)) return;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          if (active_v->get_bit(src)) {
            emit(dst, Empty());
            return;
          }
        }
      },
      [&](HyperedgeId dst, Empty msg) {
        if (!filter(dst)) {
          remaining_h->reset_bit(dst);
          now_deleted_h->set_bit(dst);
        }
        return 0;
      },
      active_v,
      remaining_h
    );

    // alternative for VertexFilterNgh step2: update remaining degrees
    if (remaining_out_degree_v != nullptr) {
      prop_from_hyperedges<int, HyperedgeId>(
        [&](HyperedgeId src) {
          emit(src, 1);
        },
        [&](HyperedgeId src, HyperedgeId msg, VertexAdjList<Empty> outgoing_adj) {
          for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
            VertexId dst = ptr->neighbour;
            write_sub(&remaining_out_degree_v[dst], msg);
          }
          return 0;
        },
        [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
          HyperedgeId sum = 0;
          for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
            HyperedgeId src = ptr->neighbour;
            if (now_deleted_h->get_bit(src))
              sum++;
          }
          if (sum > 0)
            emit(dst, sum);
        },
        [&](VertexId dst, VertexId msg) {
          write_sub(&remaining_out_degree_v[dst], msg);
          return 1;
        },
        now_deleted_h
      );
    }

    if (remaining_in_degree_v != nullptr) {
      transpose();
      prop_from_hyperedges<int, HyperedgeId>(
        [&](HyperedgeId src) {
          emit(src, 1);
        },
        [&](HyperedgeId src, HyperedgeId msg, VertexAdjList<Empty> outgoing_adj) {
          for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
            VertexId dst = ptr->neighbour;
            write_sub(&remaining_in_degree_v[dst], msg);
          }
          return 0;
        },
        [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
          HyperedgeId sum = 0;
          for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
            HyperedgeId src = ptr->neighbour;
            if (now_deleted_h->get_bit(src))
              sum++;
          }
          if (sum > 0)
            emit(dst, sum);
        },
        [&](VertexId dst, VertexId msg) {
          write_sub(&remaining_in_degree_v[dst], msg);
          return 1;
        },
        now_deleted_h
      );
      transpose();
    }
  }

  void filter_vertices_from_hyperedges(std::function<bool(VertexId)> filter, VertexSubset * remaining_v, HyperedgeSubset * active_h, VertexId * remaining_out_degree_h=nullptr, VertexId * remaining_in_degree_h=nullptr) {
    flip_bipartite_graph(false);
    filter_hyperedges_from_vertices(filter, remaining_v, active_h, remaining_out_degree_h, remaining_in_degree_h);
    flip_bipartite_graph(false);
  }
};

#endif

