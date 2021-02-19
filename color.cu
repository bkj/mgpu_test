#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

#include <cstdlib>  // EXIT_SUCCESS
#include "omp.h"
#include "nvToolsExt.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

#include <thrust/iterator/counting_iterator.h>
#include "thrust/random.h"

#define MANAGED

int n_rows;
int n_cols;
int n_nnz;

int* h_indptr;
int* h_indices;
float* h_data;

int* g_indptr;
int* g_indices;
float* g_data;

struct gpu_info {
  cudaStream_t stream;
  cudaEvent_t  event;
};

std::vector<gpu_info> infos;

cudaStream_t master_stream;


struct my_timer_t {
  float time;

  my_timer_t() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_);
  }

  ~my_timer_t() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  // Alias of each other, start the timer.
  void begin() { cudaEventRecord(start_); }
  void start() { this->begin(); }

  float end() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&time, start_, stop_);

    return milliseconds();
  }

  float seconds() { return time * 1e-3; }
  float milliseconds() { return time; }

 private:
  cudaEvent_t start_, stop_;
};

template <typename index_t, typename iterator_t>
void uniform_distribution(index_t begin, index_t end, iterator_t input) {
  using type_t = typename std::iterator_traits<iterator_t>::value_type;

  auto generate_random = [] __device__(int i) -> type_t {
    thrust::default_random_engine rng;
    rng.discard(i);
    return rng();
  };
  
  thrust::transform(thrust::make_counting_iterator(begin), thrust::make_counting_iterator(end), input, generate_random);
}

int get_num_gpus() {
  int num_gpus = -1;
  cudaGetDeviceCount(&num_gpus);
  return num_gpus;
}

void enable_peer_access() {
  int num_gpus = get_num_gpus();
  
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    for(int j = 0; j < num_gpus; j++) {
      if(i == j) 
        continue;
      cudaDeviceEnablePeerAccess(j, 0);
    }
  }
  
  cudaSetDevice(0);
}

void create_contexts() {
  int num_gpus = get_num_gpus();
  
  cudaSetDevice(0);
  cudaStreamCreateWithFlags(&master_stream, cudaStreamNonBlocking);
  
  for(int i = 0 ; i < num_gpus ; i++) {
    gpu_info info;
    cudaSetDevice(i);
    cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking);
    cudaEventCreate(&info.event);
    infos.push_back(info);
  }
  
  cudaSetDevice(0);
}

void read_binary(std::string filename) {
  FILE* file = fopen(filename.c_str(), "rb");
  
  auto err = fread(&n_rows, sizeof(int), 1, file);
  err = fread(&n_cols, sizeof(int), 1, file);
  err = fread(&n_nnz,  sizeof(int), 1, file);

  std::cout << "n_rows: " << n_rows << std::endl;
  std::cout << "n_cols: " << n_cols << std::endl;
  std::cout << "n_nnz: " << n_nnz << std::endl;

  cudaMallocManaged(&g_indptr,  (n_rows + 1) * sizeof(int));
  cudaMallocManaged(&g_indices, n_nnz        * sizeof(int));
  cudaMallocManaged(&g_data,    n_nnz        * sizeof(float));

  err = fread(g_indptr,  sizeof(int),   n_rows + 1, file);
  err = fread(g_indices, sizeof(int),   n_nnz,      file);
  err = fread(g_data,    sizeof(float), n_nnz,      file);

#ifdef MANAGED
  cudaMemAdvise(g_indptr,  (n_rows + 1) * sizeof(int),   cudaMemAdviseSetReadMostly, 0);
  cudaMemAdvise(g_indices, n_nnz        * sizeof(int),   cudaMemAdviseSetReadMostly, 0);
  cudaMemAdvise(g_data,    n_nnz        * sizeof(float), cudaMemAdviseSetReadMostly, 0);
#endif  
}

void do_test() {
  srand(123123123);
  
  int num_gpus = get_num_gpus();

  // --
  // initialize frontier
  
  thrust::host_vector<int> h_input(n_rows);
  thrust::host_vector<int> h_output(n_rows);
  for(int i = 0; i < n_rows; i++) h_input[i] = i;
  for(int i = 0; i < n_rows; i++) h_output[i] = -1;

  thrust::device_vector<int> input   = h_input;
  thrust::device_vector<int> output  = h_output;
  
  // --
  // initialize data structures
  
  thrust::device_vector<int> d_colors;
  d_colors.resize(n_rows);
  thrust::fill(thrust::device, d_colors.begin(), d_colors.end(), -1);

  int* h_randoms = (int*)malloc(n_rows * sizeof(int));
  for(int i = 0; i < n_rows; i++) h_randoms[i] = rand();
  
  int* randoms;
  cudaMallocManaged(&randoms, n_rows * sizeof(int));
  cudaMemcpy(randoms, h_randoms, n_rows * sizeof(int), cudaMemcpyHostToDevice);
#ifdef MANAGED
  cudaMemAdvise(randoms, n_rows * sizeof(int), cudaMemAdviseSetReadMostly, 0);
#endif

  int* colors  = d_colors.data().get();
  
  // --
  // Run
  
  cudaSetDevice(0);  
  cudaDeviceSynchronize();
  my_timer_t t;
  t.begin();

  int new_sizes[num_gpus];
  
  int* indptr  = g_indptr;
  int* indices = g_indices;
  float* data  = g_data;
  
  nvtxRangePushA("thrust_work");
  
  int iteration = 0;
  while(input.size() > 4) {
    
    int chunk_size  = (input.size() + num_gpus - 1) / num_gpus;
    
    #pragma omp parallel for num_threads(num_gpus)
    for(int i = 0 ; i < num_gpus ; i++) {
      
      cudaSetDevice(i);

      auto fn = [indptr, indices, data, colors, randoms, iteration] __host__ __device__(int const& vertex) -> bool {
        int start  = indptr[vertex];
        int end    = indptr[vertex + 1];
        int degree = end - start;

        bool colormax = true;
        bool colormin = true;
        int color     = iteration * 2;

        for (int i = 0; i < degree; i++) {
          int u = indices[start + i];

          if (colors[u] != -1 && (colors[u] != color + 1) && (colors[u] != color + 2) || (vertex == u))
            continue;
          if (randoms[vertex] <= randoms[u])
            colormax = false;
          if (randoms[vertex] >= randoms[u])
            colormin = false;
        }

        if (colormax) {
          colors[vertex] = color + 1;
          return false;
        } else if (colormin) {
          colors[vertex] = color + 2;
          return false;
        } else {
          return true;
        }
      };
      
      auto input_begin  = input.begin() + chunk_size * i;
      auto input_end    = input.begin() + chunk_size * (i + 1);
      auto output_begin = output.begin() + chunk_size * i;
      if(i == num_gpus - 1) input_end = input.end();
      
      auto new_output_end = thrust::copy_if(
        thrust::cuda::par.on(infos[i].stream),
        input_begin,
        input_end,
        output_begin,
        fn
      );
      new_sizes[i] = (int)thrust::distance(output_begin, new_output_end);
      cudaEventRecord(infos[i].event, infos[i].stream);
    }
    
    for(int i = 0; i < num_gpus; i++)
      cudaStreamWaitEvent(master_stream, infos[i].event, 0);
    cudaStreamSynchronize(master_stream);
    
    int total_length = 0;
    int offsets[num_gpus];
    offsets[0] = 0;
    for(int i = 1 ; i < num_gpus ; i++) offsets[i] = new_sizes[i - 1] + offsets[i - 1];
    for(int i = 0 ; i < num_gpus ; i++) total_length += new_sizes[i];

    // Reduce
    cudaSetDevice(0);
    for(int i = 0; i < num_gpus; i++) {
      auto output_begin = output.begin() + chunk_size * i;
      thrust::copy_n(
        thrust::cuda::par.on(infos[0].stream),
        output_begin, 
        new_sizes[i], 
        input.begin() + offsets[i]
      );
    }
    
    cudaEventRecord(infos[0].event, infos[0].stream);
    cudaStreamWaitEvent(master_stream, infos[0].event, 0);
    cudaStreamSynchronize(master_stream);
    
    input.resize(total_length);
    output.resize(total_length);
      
    iteration++;
    // t.end(); std::cout << "elapsed: " << t.milliseconds() << std::endl;
  }
  nvtxRangePop();
  
  // Log
  thrust::host_vector<int> out = d_colors;
  thrust::copy(out.begin(), out.begin() + 32, std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
  
  cudaSetDevice(0);
  t.end();  
  std::cout << "total_elapsed: " << t.milliseconds() << std::endl;
}

int main(int argc, char** argv) {
  std::string inpath = argv[1];
  
  enable_peer_access();
  create_contexts();
  read_binary(inpath);

  int num_gpus = get_num_gpus();
  std::cout << "color | num_gpus: " << num_gpus << std::endl;

  int num_iters = 4;
  for(int i = 0; i < num_iters; i++)
    do_test();
  
  std::cout << "-----" << std::endl;
  return EXIT_SUCCESS;
}