#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

#include <cstdlib>  // EXIT_SUCCESS
#include "omp.h"
#include "nvToolsExt.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

#include <thrust/iterator/counting_iterator.h>
#include "thrust/random.h"

int n_rows;
int n_cols;
int n_nnz;

int* h_indptr;
int* h_indices;

int* g_indptr;
int* g_indices;

int** all_indptrs;
int** all_indices;
int** all_randoms;
int** all_inputs;
int** all_colors;

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
  
  cudaStreamCreateWithFlags(&master_stream, cudaStreamNonBlocking);
  
  for(int i = 0 ; i < num_gpus ; i++) {
    gpu_info info;
    cudaSetDevice(i);
    cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&info.event, cudaEventDisableTiming);
    infos.push_back(info);
  }
  
  cudaSetDevice(0);
}

void read_binary(std::string filename) {
  FILE* file = fopen(filename.c_str(), "rb");
  
  auto err = fread(&n_rows, sizeof(int), 1, file);
  err = fread(&n_cols, sizeof(int), 1, file);
  err = fread(&n_nnz,  sizeof(int), 1, file);

  h_indptr  = (int*  )malloc((n_rows + 1) * sizeof(int));
  h_indices = (int*  )malloc(n_nnz        * sizeof(int));

  err = fread(h_indptr,  sizeof(int),   n_rows + 1, file);
  err = fread(h_indices, sizeof(int),   n_nnz,      file);
  
  int num_gpus = get_num_gpus();
  
  all_indptrs = (int**)malloc(num_gpus * sizeof(int*));
  all_indices = (int**)malloc(num_gpus * sizeof(int*));
  
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    
    int* l_indptrs;
    int* l_indices;
    cudaMalloc(&l_indptrs,  (n_rows + 1) * sizeof(int));
    cudaMalloc(&l_indices,  (n_nnz     ) * sizeof(int));
    
    cudaMemcpy(l_indptrs, h_indptr, (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(l_indices, h_indices, n_nnz * sizeof(int), cudaMemcpyHostToDevice);
    
    all_indptrs[i] = l_indptrs;
    all_indices[i] = l_indices;
  }
  cudaSetDevice(0);

}

void do_test() {
  srand(345345345);
  
  int num_gpus = get_num_gpus();

  // --
  // initialize frontier
  
  int chunk_size  = (n_rows + num_gpus - 1) / num_gpus;
  
  
  // Inputs, chunked across devices
  all_inputs = (int**)malloc(num_gpus * sizeof(int*));
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    
    int* h_input = (int*)malloc(chunk_size * sizeof(int));
    
    int begin    = chunk_size * i;
    for(int ii = 0; ii < chunk_size ; ii++) {
      if(begin + ii < n_rows)
        h_input[ii] = begin + ii;
      else
        h_input[ii] = -1;
    }
    
    int* l_inputs;
    cudaMalloc(&l_inputs, chunk_size * sizeof(int));
    cudaMemcpy(l_inputs, h_input, chunk_size * sizeof(int), cudaMemcpyHostToDevice);  
    all_inputs[i] = l_inputs;
  }
  cudaSetDevice(0);
  
  
  // randoms, copied across devices
  int* h_randoms = (int*)malloc(n_rows * sizeof(int));
  for(int i = 0; i < n_rows; i++) h_randoms[i] = (int)rand();
  
  all_randoms = (int**)malloc(num_gpus * sizeof(int*));
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    
    int* l_randoms;
    cudaMalloc(&l_randoms, n_rows * sizeof(int));
    cudaMemcpy(l_randoms, h_randoms, n_rows * sizeof(int), cudaMemcpyHostToDevice);  
    all_randoms[i] = l_randoms;
  }
  cudaSetDevice(0);


  // colors, chunked across devices  
  all_colors = (int**)malloc(num_gpus * sizeof(int*));

  int* h_color = (int*)malloc(chunk_size * sizeof(int));
  for(int ii = 0; ii < chunk_size ; ii++) h_color[ii] = -1;
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    
    int* l_colors;
    cudaMalloc(&l_colors, chunk_size * sizeof(int));
    cudaMemcpy(l_colors, h_color, chunk_size * sizeof(int), cudaMemcpyHostToDevice);  
    all_colors[i] = l_colors;
  }
  cudaSetDevice(0);
  
  // --
  // Run
  
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);  
    cudaDeviceSynchronize();
  }

  cudaSetDevice(0);

  my_timer_t t;
  std::vector<float> per_iteration_times;

  int* colors0 = all_colors[0];
  int* colors1 = all_colors[1];
  int* colors2 = all_colors[2];
  int* colors3 = all_colors[3];
  
  nvtxRangePushA("thrust_work");

  int iteration = 0;
  while(iteration < 16) {
    t.begin();
    
    #pragma omp parallel for 
    for(int i = num_gpus - 1 ; i >= 0; i--) {
      
      cudaSetDevice(i);

      int* indptr  = all_indptrs[i];
      int* indices = all_indices[i];
      int* randoms = all_randoms[i];
      int* inputs  = all_inputs[i];
      
      int offset   = i * chunk_size;
      int* wcolors = all_colors[i];
      
      auto fn = [indptr, indices, randoms, wcolors, offset, iteration, colors0, colors1, colors2, colors3, chunk_size] __host__ __device__(int const& vertex) {
        if(vertex == -1) return -1;
        
        int start  = indptr[vertex];
        int end    = indptr[vertex + 1];
        int degree = end - start;

        bool colormax = true;
        bool colormin = true;
        int color     = iteration * 2;

        int rv = randoms[vertex];
        
        for (int i = 0; i < degree; i++) {
          int u = indices[start + i];

          int ncolor = -1;
          if(u < chunk_size) {
            ncolor = colors0[u - 0 * chunk_size];
          } else if(u < 2 * chunk_size) {
            ncolor = colors1[u - 1 * chunk_size];
          } else if(u < 3 * chunk_size) {
            ncolor = colors2[u - 2 * chunk_size];
          } else {
            ncolor = colors3[u - 3 * chunk_size];
          }
          
          if (ncolor != -1 && (ncolor != color + 1) && (ncolor != color + 2) || (vertex == u)) continue;
          
          int ru = randoms[u];
          if(colormax) {if (rv <= ru) colormax = false;}
          if(colormin) {if (rv >= ru) colormin = false;}
          
          // if(!colormax && !colormin) break; // optimization
        }

        if (colormax) {
          wcolors[vertex - offset] = color + 1;
          return -1;
        } else if (colormin) {
          wcolors[vertex - offset] = color + 2;
          return -1;
        } else {
          return vertex;
        }
      };

      thrust::transform(
        thrust::cuda::par.on(infos[i].stream),
        inputs,
        inputs + chunk_size,
        inputs,
        fn
      );

      cudaEventRecord(infos[i].event, infos[i].stream);
    }
    
    cudaSetDevice(0);
    for(int i = 0; i < num_gpus; i++)
      cudaStreamWaitEvent(master_stream, infos[i].event, 0);

    cudaStreamSynchronize(master_stream);
      
    iteration++;
    t.end();
    per_iteration_times.push_back(t.milliseconds());
    std::cout << t.milliseconds() << std::endl;
  }
  nvtxRangePop();
  
  cudaSetDevice(0);

  float total_elapsed = 0;
  for (auto& n : per_iteration_times)
    total_elapsed += n;

  std::cout << "total_elapsed: " << total_elapsed << std::endl;
}

int main(int argc, char** argv) {
  cudaSetDevice(0);
  
  std::string inpath = argv[1];
  
  enable_peer_access();
  create_contexts();
  read_binary(inpath);

  int num_gpus = get_num_gpus();
  std::cout << "color | num_gpus: " << num_gpus << " vertices: " << n_rows << std::endl;

  int num_iters = 2;
  for(int i = 0; i < num_iters; i++)
    do_test();
  
  std::cout << "-----" << std::endl;
  return EXIT_SUCCESS;
}
