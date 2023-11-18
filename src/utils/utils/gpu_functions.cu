#include <cuda_runtime.h>
#include "gpu_functions.h"
// fastSig()
#include "sigmoid.h"
// randn()
#include "rand_helper.h"
#include <stdio.h>
using namespace std;

#define MAX_ALPHA 101
#define FULL_MASK 0xffffffff
__inline__ __device__ 
double warpAllReduceSum(double val) {

  for (int i =1; i<32; i*=2){
    val+=__shfl_xor_sync(FULL_MASK,val,i);
  }
  return val;
}

__device__ vid_t get_negative_sample(vid_t num_vertices, void* seed){
  return randn(seed) % num_vertices;
}
/////// For testing ///////
__device__ bool u_has_edge_v(vid_t u, vid_t v, vid_t* d_V, vid_t* d_A){
  for (unsigned long long i =0; i< d_V[u+1]-d_V[u];i++){
    if (d_A[i+d_V[u]]==v) return true;
  }
  return false;
}

__device__ bool u_in_part_x(vid_t u, int part_id, int vids_per_bin, int num_vertices){
  vid_t min = vids_per_bin*part_id;
  vid_t max = vids_per_bin*(part_id+1);
  if (max>num_vertices) max = num_vertices;
  if (u>=max || u<min) return false;
  else return true;
}
//////////////////////////

//*********************************************************
// BLOOM FILTER
#define BITS_PER_INT 32

__device__ void setBit(unsigned int* array, int bit) {
    atomicOr(&array[bit / BITS_PER_INT], 1 << (bit % BITS_PER_INT));
}

/**
* Calculates the djb2 hash.
* @param str The string being hashed.
* @param start The starting point of the word in the array.
* @return Returns the djb2 hash in unsigned int format.
*/
__device__ unsigned int djb2Hash(unsigned int* str, int start) {
    unsigned int hash = 5381;
    unsigned int c;
    while (str[start] != ',') {
        c = str[start];
        hash = ((hash << 5) + hash) + c;
        start++;
    }
    return hash;
}

/**
* Calculates the sdbm hash.
* @param str The string being hashed.
* @param start The starting point of the word in the array.
* @return Returns the sdbm hash in unsigned int format.
*/
__device__ unsigned int sdbmHash(unsigned int* str, int start) {
    unsigned int hash = 0;
    unsigned int c = 0;
    while (str[start] != ',') {
        c = str[start];
        hash = c + (hash << 6) + (hash << 16) - hash;
        start++;
    }
    return hash;
}


__device__ float CountCommonAndTotalBits(unsigned int* vertexList, unsigned int vertex1, unsigned int vertex2, unsigned int* bloomFilter) {
  // Calculate hash values using both hash functions for both vertices
    unsigned int hash1_vertex1 = djb2Hash(vertexList, vertex1) % FILTER_SIZE;
    unsigned int hash2_vertex1 = sdbmHash(vertexList, vertex1) % FILTER_SIZE;

    unsigned int hash1_vertex2 = djb2Hash(vertexList, vertex2) % FILTER_SIZE;
    unsigned int hash2_vertex2 = sdbmHash(vertexList, vertex2) % FILTER_SIZE;

    float resultRatio;
    // Count the common bits
    int commonBits = __popc(bloomFilter[hash1_vertex1] & bloomFilter[hash2_vertex1] & bloomFilter[hash1_vertex2] & bloomFilter[hash2_vertex2]);

    // Count the total bits (union)
    int totalBits = __popc(bloomFilter[hash1_vertex1] | bloomFilter[hash2_vertex1] | bloomFilter[hash1_vertex2] | bloomFilter[hash2_vertex2]);

    // Calculate the ratio and store the result
    resultRatio = static_cast<float>(commonBits) / static_cast<float>(totalBits);
    return resultRatio;
}


__global__ void BloomFilter_Insert(unsigned int* vertexList, unsigned int* edgeList, unsigned int* bloomFilter, int numVertices, int numEdges){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < numEdges) 
    {
        // Extract the source and destination vertices for the current edge
        unsigned int srcVertex = edgeList[tid * 2];
        unsigned int destVertex = edgeList[tid * 2 + 1];

        // Calculate hash values using both hash functions
        unsigned int hash1 = djb2Hash(vertexList, srcVertex) % FILTER_SIZE;
        unsigned int hash2 = sdbmHash(vertexList, destVertex) % FILTER_SIZE;

        // Set the corresponding bits in the Bloom filter to true
        setBit(bloomFilter, hash1);
        setBit(bloomFilter, hash2);

        tid += blockDim.x * gridDim.x;
      }
}

// BLOOM FILTER ENDS
//**********************************************************


__device__ void single_sample_update(emb_t * vEmbeddings, emb_t * uEmbeddings, int dimensions, int positive, double learning_rate, int id, double bias, float* d_sigmoid_lookup_table, double negative_weight, int WARP_SIZE, unsigned int source, unsigned int generated_sample, unsigned int* bloomFilter, unsigned int * vertexList, int bf){
  int start = id % WARP_SIZE;
  double myscore = 0;
  for (int i=start ; i < dimensions; i += WARP_SIZE){
    myscore += vEmbeddings[i] * uEmbeddings[i];
  }
  double g = warpAllReduceSum(myscore);
  g-=bias;

  // call for bloom filter
  double fake_jaccard = CountCommonAndTotalBits(vertexList, source, generated_sample, bloomFilter);
  double f = (positive - fastSig(g, d_sigmoid_lookup_table)) * learning_rate * fake_jaccard;
  
  // perform update on embedding of graphs
  double nw = 1;
  if (positive == 0){
    nw = negative_weight;
  }
  for (int i = start; i < dimensions; i += WARP_SIZE){
    float u = uEmbeddings[i];
    float v = vEmbeddings[i];
    vEmbeddings[i] += u * f * nw;
    uEmbeddings[i] += v * f * nw;
  }
}

__device__ void single_sample_update(emb_t * vEmbeddings, emb_t * uEmbeddings, int dimensions, int positive, double learning_rate, int id, double bias, float* d_sigmoid_lookup_table, double negative_weight, int WARP_SIZE, int bf){
  int start = id % WARP_SIZE;
  double myscore = 0;
  for (int i=start ; i < dimensions; i += WARP_SIZE){
    myscore += vEmbeddings[i] * uEmbeddings[i];
  }
  double g = warpAllReduceSum(myscore);
  g-=bias;

  double f = (positive - fastSig(g, d_sigmoid_lookup_table)) * learning_rate;
  
  // perform update on embedding of graphs
  double nw = 1;
  if (positive == 0){
    nw = negative_weight;
  }
  for (int i = start; i < dimensions; i += WARP_SIZE){
    float u = uEmbeddings[i];
    float v = vEmbeddings[i];
    vEmbeddings[i] += u * f * nw;
    uEmbeddings[i] += v * f * nw;
  }
}


//wedge parametresi eklendi, exec komutunda 1 ya da 0 verilmesine gore wedge sampling yapcak
__device__ unsigned int get_positive_sample_ppr_device(unsigned int source, unsigned int * V, unsigned int * A, void* seed, int alpha, int wedge){


  unsigned int result = source;
  unsigned int numNeighbours = V[result+1] - V[result];
  unsigned long randNum = randn(seed);
  if (alpha == 0){
    if (numNeighbours == 0){
      return result;
    }
     else {
      if (wedge == 0){
        return A[V[result] + (randNum%numNeighbours)];
      }
      else{ //wedge sampling aciksa
      // KOMSUSU YERINE KOMSUSUNUN KOMSUSUNU DONCEM WEDGE ICIN |WEDGE PATH|=2
        unsigned int w = A[V[result] + (randNum%numNeighbours)];
        unsigned int numNeighboursNeighbours = V[w+1] - V[w];
        unsigned long randNumNum = randn(seed);


        return A[V[w] + (randNumNum%numNeighboursNeighbours)];
      }
    }
  }
  while(randNum % MAX_ALPHA < alpha){
    if (numNeighbours == 0){
      return result;
    }
    else {
      result = A[V[result]+(randNum%numNeighbours)];
    }
    randNum = randn(seed);
  }
  return result;
} 




__global__ void Embedding_Kernel(unsigned int d_num_vertices, int d_num_epoch, unsigned int * d_V, unsigned int * d_A,  emb_t * d_embeddings, int d_dim, int d_s, double d_lr, float* d_sigmoid_lookup_table, int ep_start, int total_batches, int alpha, int wedge, double negative_weight, unsigned int* bloomFilter, int WARP_SIZE, int WARPS_PER_BLOCK, int NUM_WARPS){
  //A warp per vertex strategy
  const unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
  const unsigned int warp_num = id/WARP_SIZE;
  const double nce_bias = logf(d_num_vertices);
  const double nce_neg_bias=logf(d_num_vertices/(d_s));
  double lr = d_lr;
  //seed for random number generation
  seed mySeed;
  void *sd = &mySeed; // This is an abstraction to allow us to pass any seed to the functions (for other random function tests later)
  mySeed.x = (1+warp_num*ep_start)*123456789; // the values of the seed are a function of the thread id
  mySeed.y = (1+warp_num*ep_start)*362436069; // (i wasn't very creative with these functions but I didn't see any decrease in ML performance)
  mySeed.z = (1+warp_num*ep_start)*521288629;
  //
  unsigned int source, p_sample, n_sample;
  // Every block has 128*number of warps within it shared memory floats
  // each warp will store its embeddings somewhere in this big array
  // myemb_s size = 128*(NUM_THREADS/WARP_SIZE)
  // warp_id_in_block = warp_num%WARPS_PER_BLOCK
  const int dims = d_dim;
  extern __shared__ emb_t emb_s[];
  emb_t* myemb_s = &emb_s[(warp_num%WARPS_PER_BLOCK)*dims];
  emb_t* myemb_g;
  for (int ep = 0; ep < d_num_epoch; ep++){
    for (int i = warp_num; i< d_num_vertices; i+=NUM_WARPS){
      source = i;
      p_sample = get_positive_sample_ppr_device(source, d_V, d_A, sd, alpha, wedge);//generate positive sample
      if(p_sample != UINT_MAX) {
        myemb_g = d_embeddings+(dims*source);
        for (int j =id%WARP_SIZE; j<dims; j+=WARP_SIZE)
          myemb_s[j] = myemb_g[j];

        single_sample_update(myemb_s, d_embeddings+dims*p_sample, dims, 1, lr, id, nce_bias, d_sigmoid_lookup_table, negative_weight, WARP_SIZE, source, p_sample, bloomFilter, d_V, bf); //update one positive sample
        // generated sample: either p_sample or n_sample will be given inside.
        
        for(unsigned int k = 0; k < d_s; k++){
          n_sample = get_negative_sample(d_num_vertices, sd);//get negative sample
          single_sample_update(myemb_s, d_embeddings+dims*n_sample, dims, 0, lr, id, nce_neg_bias, d_sigmoid_lookup_table, negative_weight, WARP_SIZE, source, n_sample, bloomFilter, d_V, bf);
        }
        for (int l =id%WARP_SIZE; l<dims; l+=WARP_SIZE)
          myemb_g[l] = myemb_s[l];
      }
    }
  }
}


__global__ void Big_Graphs_Embedding_Kernel(emb_t *source_bin, emb_t* dest_bin, long long vertices_per_part, int num_vertices, int starting_ep, int batch_ep,vid_t* vids, double d_lr, int dim, int neg_s, float* sig_table, int alpha, int wedge, int bf, int source_part_id, int dest_part_id , int WARP_SIZE, int WARPS_PER_BLOCK, int NUM_WARPS){
  const unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
  const unsigned int warp_num = id/WARP_SIZE;
  const double nce_bias = logf(1.0*num_vertices);
  const double nce_neg_bias=logf(1.0*num_vertices/float(neg_s));
  emb_t* myemb_g;
  long long size_s2d=vids[0];
  long long size_d2s = vids[1];
  vids = &vids[2];
  long long w;
#ifdef SHARED_MEMORY
  extern __shared__ emb_t emb_s[];
  emb_t* myemb_s = &emb_s[(warp_num%WARPS_PER_BLOCK)*dim];
#else
  emb_t*& myemb_s = myemb_g;
#endif
  seed mySeed;
  mySeed.x = (1+warp_num+starting_ep)*123456789; // the values of the seed are a function of the thread id
  mySeed.y = (1+warp_num+starting_ep)*362436069; // (i wasn't very creative with these functions but I didn't see any decrease in ML performance)
  mySeed.z = (1+warp_num+starting_ep)*521288629;
  vid_t source, p_sample, ns;
  double lr = d_lr;
  vid_t ns_limit = vertices_per_part;
  if ((dest_part_id+1)*vertices_per_part>num_vertices){
    ns_limit = num_vertices-vertices_per_part*dest_part_id;
  }
  for (w = warp_num; w<size_s2d; w+=NUM_WARPS){
    // p_sample is the absolute id of the target
    source = vids[w*2];
#ifdef _DEBUG_GPU
    if (!u_in_part_x(source, source_part_id, vertices_per_part, num_vertices))
      printf("PROBLEM: s2d - Found a positive sample while processing %d, %d that is incorrect\n", source_part_id, dest_part_id);
#endif
    myemb_g = source_bin+(dim*(source-vertices_per_part*source_part_id));
#ifdef SHARED_MEMORY
    for (int j =id%WARP_SIZE; j<dim; j+=WARP_SIZE)
      myemb_s[j] = myemb_g[j];
#endif
    p_sample =vids[w*2+1];
#ifdef _DEBUG_GPU
    if (!u_in_part_x(p_sample, dest_part_id, vertices_per_part, num_vertices))
      printf("PROBLEM: s2d - Found a positive sample while processing %d, %d that is incorrect\n", source_part_id, dest_part_id);
#endif
    single_sample_update(myemb_s, dest_bin+(p_sample-vertices_per_part*dest_part_id)*dim, dim, 1, lr, id, nce_bias, sig_table, 1, WARP_SIZE, source, p_sample, bf);
    for (int s = 0;s <neg_s; s++){
      ns = get_negative_sample(ns_limit, &mySeed);
      single_sample_update(myemb_s, dest_bin+ns*dim, dim, 0, lr, id, nce_neg_bias, sig_table, 1, WARP_SIZE, source, p_sample, bf);
    }
#ifdef SHARED_MEMORY
    for (int j =id%WARP_SIZE; j<dim; j+=WARP_SIZE)
      myemb_g[j] = myemb_s[j];
#endif
  }
  if (source_part_id!=dest_part_id){
    ns_limit = vertices_per_part;
    if ((source_part_id+1)*vertices_per_part>num_vertices){
      ns_limit = num_vertices-vertices_per_part*source_part_id;
    }
    long long offset = 2*(size_s2d);
    for (w = offset+(warp_num*2); w<offset+(size_d2s*2); w+=2*NUM_WARPS){
      // p_sample is the absolute id of the target
      source = vids[(w)];
#ifdef _DEBUG_GPU
      if (!u_in_part_x(source, dest_part_id, vertices_per_part, num_vertices))
        printf("PROBLEM: d2s - Found a sourcewhile processing %d, %d that is incorrect\n", source_part_id, dest_part_id);
#endif
      myemb_g = dest_bin+(dim*(source-vertices_per_part*dest_part_id));
#ifdef SHARED_MEMORY
      for (int j =id%WARP_SIZE; j<dim; j+=WARP_SIZE)
        myemb_s[j] = myemb_g[j];
#endif
      p_sample =vids[w+1];
#ifdef _DEBUG_GPU
      if (!u_in_part_x(p_sample,source_part_id, vertices_per_part, num_vertices))
        printf("PROBLEM: d2s - Found a positive sample while processing %d, %d that is incorrect\n", source_part_id, dest_part_id);
#endif
      single_sample_update(myemb_s, source_bin+(p_sample-vertices_per_part*source_part_id)*dim, dim, 1, lr, id, nce_bias, sig_table, 1, WARP_SIZE, source, p_sample, bf);
      for (int s = 0;s <neg_s; s++){
        ns = get_negative_sample(ns_limit, &mySeed);
        single_sample_update(myemb_s, source_bin+ns*dim, dim, 0, lr, id, nce_neg_bias, sig_table, 1, WARP_SIZE, source, p_sample, bf);
      }
#ifdef SHARED_MEMORY
      for (int j =id%WARP_SIZE; j<dim; j+=WARP_SIZE)
        myemb_g[j] = myemb_s[j];
#endif
    }
  }  
  //if (id == 0) printf("finished %d %d\n",source_part_id, dest_part_id); 
}
const unsigned kFullMask = 0xFFFFFFFF;
  template <class T>
    __device__ T WarpBroadcast(T value, int lane_id) {
#if __CUDACC_VER_MAJOR__ >= 9
      return __shfl_sync(kFullMask, value, lane_id);
#else
      return __shfl(value, lane_id);
#endif
    }
  template <class T>
    __device__ T WarpReduce(T value) {
#pragma unroll
      for (int delta = 1; delta < 32; delta *= 2)
#if __CUDACC_VER_MAJOR__ >= 9
        value += __shfl_down_sync(kFullMask, value, delta);
#else
      value += __shfl_down(value, delta);
#endif
      return value;
    }

#define MAX_ALPHA 101
#define FULL_MASK 0xffffffff
#define NEGATIVE_WEIGHT 5
__device__ void single_sample_update_pos(emb_t * vEmbeddings, emb_t * uEmbeddings, float learning_rate, int id, float bias, int dimension, int WARP_SIZE){
   int start = id % WARP_SIZE;
   //double myscore = 0;
   float x = 0;
   for (int i=start ; i < dimension; i += WARP_SIZE){
     x += vEmbeddings[i] * uEmbeddings[i];
   }
   x = WarpBroadcast(WarpReduce(x), 0);
   x-=bias;
   float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
   prob = prob - 1;
   // perform update on embedding of graphs
   float lol = prob  * learning_rate;
   for (int i = start; i < dimension; i += WARP_SIZE){
     float u = uEmbeddings[i];
     float v = vEmbeddings[i];
     vEmbeddings[i] -= u * lol;
     uEmbeddings[i] -= v * lol;
   }
}
__device__ void single_sample_update_neg(emb_t * vEmbeddings, emb_t * uEmbeddings, float learning_rate, int id, float bias, int dimension, float negative_weight, int WARP_SIZE){
   int start = id % WARP_SIZE;
   //double myscore = 0;
   float x = 0;
   for (int i=start ; i < dimension; i += WARP_SIZE){
     x += vEmbeddings[i] * uEmbeddings[i];
   }
   x = WarpBroadcast(WarpReduce(x), 0);
   x-=bias;
   float prob = x > 0 ? 1 / (1 + exp(-x)) : exp(x) / (exp(x) + 1);
   // perform update on embedding of graphs
   float lol = prob *negative_weight * learning_rate;
   for (int i = start; i < dimension; i += WARP_SIZE){
     float u = uEmbeddings[i];
     float v = vEmbeddings[i];
     vEmbeddings[i] -= u * lol;
     uEmbeddings[i] -= v * lol;
   }
}
__global__ void Embedding_Kernel_SP(unsigned int d_num_vertices, unsigned long samples_per_pool, unsigned int* d_sample_array, unsigned int* d_fake,  emb_t * d_embeddings, float d_lr, int dimension, int negative_samples, float negative_weight, int WARPS_PER_BLOCK, int WARP_SIZE, int NUM_WARPS){
  //A warp per vertex strategy
  const unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
  const unsigned int warp_num = id/WARP_SIZE;
  const float nce_bias = logf(d_num_vertices);
  const float nce_neg_bias=logf(d_num_vertices/(negative_samples));
  unsigned int source, p_sample, n_sample;
  extern __shared__ emb_t emb_s[];
  emb_t* myemb_s = &emb_s[(warp_num%WARPS_PER_BLOCK)*dimension]; 
  emb_t* myemb_g;
  for (int i = warp_num; i<samples_per_pool; i+=NUM_WARPS){
    source = d_sample_array[i*2];
    p_sample = d_sample_array[i*2+1];
    myemb_g = d_embeddings+(dimension*source);
    for (int j =id%WARP_SIZE; j<dimension; j+=WARP_SIZE)
      myemb_s[j] = myemb_g[j];
    single_sample_update_pos(myemb_s, d_embeddings+dimension*p_sample, d_lr, id, nce_bias, dimension, WARP_SIZE); //update one positive sample
    for(unsigned int k = 0; k < negative_samples; k++){
      n_sample = d_fake[i*negative_samples+k];
      single_sample_update_neg(myemb_s, d_embeddings+dimension*n_sample, d_lr, id, nce_neg_bias, dimension, negative_weight, WARP_SIZE);
    }
    for (int l =id%WARP_SIZE; l<dimension; l+=WARP_SIZE)
      myemb_g[l] = myemb_s[l];
  }
}

