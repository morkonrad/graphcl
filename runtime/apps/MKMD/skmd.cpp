#include "mkmd.h"


#include <random>
#include <execution>
#include <stdexcept>
#include <iomanip>
#include <thread>
#include <chrono>


static std::string tasks = "";

void kernels::init_kernels_skmd()
{
    if (tasks.empty())
    {
        tasks.append("#define T float");
        tasks.append("\n");
        tasks.append(kernels::task_ADD_SUB);
        tasks.append("\n");
        tasks.append(kernels::task_ADD);
        tasks.append("\n");
        tasks.append(kernels::task_MERGE);
        tasks.append("\n");
        tasks.append(kernels::task_MM);
        tasks.append("\n");
        tasks.append(kernels::task_MT);
        //std::cout << tasks << std::endl;
    }
}

#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))
inline void transpose4x4_SSE(float* A, float* B, const int lda, const int ldb) {
    __m128 row1 = _mm_load_ps(&A[0 * lda]);
    __m128 row2 = _mm_load_ps(&A[1 * lda]);
    __m128 row3 = _mm_load_ps(&A[2 * lda]);
    __m128 row4 = _mm_load_ps(&A[3 * lda]);
    _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
    _mm_store_ps(&B[0 * ldb], row1);
    _mm_store_ps(&B[1 * ldb], row2);
    _mm_store_ps(&B[2 * ldb], row3);
    _mm_store_ps(&B[3 * ldb], row4);
}

inline void transpose_block_SSE4x4(float* A, float* B, const int mat_height, const int mat_width, const int lda, const int ldb, const int block_size) {

#pragma omp parallel for
    for (int i = 0; i < mat_height; i += block_size) {
        for (int j = 0; j < mat_width; j += block_size) {
            int max_i2 = i + block_size < mat_height ? i + block_size : mat_height;
            int max_j2 = j + block_size < mat_width ? j + block_size : mat_width;
            for (int i2 = i; i2 < max_i2; i2 += 4) {
                for (int j2 = j; j2 < max_j2; j2 += 4) {
                    transpose4x4_SSE(&A[i2 * lda + j2], &B[j2 * ldb + i2], lda, ldb);
                }
            }
        }
    }
}

constexpr size_t BLOCK_DIM_global = 16;

template<typename T>
static void generate_rand_real(std::vector<T>& container, const T start, const T end)
{
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(static_cast<T>(start), static_cast<T>(end));

    std::for_each(std::execution::par_unseq, container.begin(), container.end(), [&gen, &dis](T& val) {
        val = static_cast<float>(dis(gen));
    });
}

static int Benchmark_platform(mkmd_input& matrix_data,
                              const std::vector<offload_info>& proc_setups,
                              const std::vector<std::tuple<map_device_info, map_device_info, std::string>>& copy_setups,
                              const bool benchmark_transfers = true)
{
    auto& device = *matrix_data.device();

    std::cout << "ALLOCATE MEMORY ..." << std::endl;

    const auto items_in = matrix_data.items();
    const auto items_out = matrix_data.items_out();

    std::vector<float> random_values_a(items_in);
    std::vector<float> random_values_b(items_in);
    std::vector<float> random_values_c(items_in);
    std::vector<float> random_values_d(items_in);

    generate_rand_real(random_values_a, 0.01f, 1.0f);
    generate_rand_real(random_values_b, 0.1f, 2.0f);
    generate_rand_real(random_values_c, 1.1f, 2.0f);
    generate_rand_real(random_values_d, 2.1f, 2.5f);

    //allocate memory
    auto matrix_a = device.alloc<float>(random_values_a, true);//read_only
    if (!matrix_a)return COOPCL_BAD_ALLOC;

    auto matrix_x = device.alloc<float>(random_values_b, true);//read_only
    if (!matrix_x)return COOPCL_BAD_ALLOC;

    auto matrix_g = device.alloc<float>(random_values_c, true);//read_only
    if (!matrix_g)return COOPCL_BAD_ALLOC;

    auto matrix_e = device.alloc<float>(random_values_d, true);//read_only
    if (!matrix_e)return COOPCL_BAD_ALLOC;

    //------------------------------------------ READ_WRITE

    auto matrix_at = device.alloc<float>(items_out);
    if (!matrix_at)return COOPCL_BAD_ALLOC;

    auto matrix_xg = device.alloc<float>(items_out);
    if (!matrix_xg)return COOPCL_BAD_ALLOC;

    auto matrix_etxa = device.alloc<float>(items_out);
    if (!matrix_etxa)return COOPCL_BAD_ALLOC;

    auto matrix_etxg = device.alloc<float>(items_out);
    if (!matrix_etxg)return COOPCL_BAD_ALLOC;

    auto matrix_atxe = device.alloc<float>(items_out);
    if (!matrix_atxe)return COOPCL_BAD_ALLOC;

    auto matrix_etxgxe = device.alloc<float>(items_out);
    if (!matrix_etxgxe)return COOPCL_BAD_ALLOC;

    auto matrix_results = device.alloc<float>(items_out);
    if (!matrix_results)return COOPCL_BAD_ALLOC;

    std::cout << "BUILD KERNELS..." << std::endl;

    //Create tasks
    std::string jit_flags = " -cl-unsafe-math-optimizations -cl-fast-relaxed-math";


    auto task_mat_add_sub = device.create_task(tasks, "mat_add_sub", jit_flags);
    if (!task_mat_add_sub)throw std::runtime_error("Error JTI, FIMXE!!!");

    auto task_mat_merge = device.create_task(tasks, "mat_merge", jit_flags);
    if (!task_mat_merge)throw std::runtime_error("Error JTI, FIMXE!!!");

    std::array<size_t, 3> global_sizes = { static_cast<size_t>(matrix_data._matrix_width), static_cast<size_t>(matrix_data._matrix_height),1 };
    std::array<size_t, 3> local_sizes = { BLOCK_DIM_global,BLOCK_DIM_global,1 };

    //const auto local_mem = (BLOCK_DIM_global + 1) * BLOCK_DIM_global * sizeof(float);
    //const int offset = 0;
    //cl::LocalSpaceArg lmem = cl::Local(local_mem);


    const auto ms_scale_factor = 1e-6f;

    auto call_benchmark = [&device,
                           &ms_scale_factor, &global_sizes, &local_sizes, &task_mat_merge, &task_mat_add_sub,
                           &matrix_data, &matrix_a, &matrix_results, &matrix_atxe, &matrix_etxa, &matrix_etxgxe]
        (const std::vector<offload_info>& proc,
         const std::vector<std::tuple<map_device_info, map_device_info, std::string>>& copy_source_destination,
         const size_t iterations = 10, const bool benchmark_copy = true)->int
    {
        int err = 0;

        const size_t iter = iterations < 1 ? 1 : iterations;

        if (benchmark_copy)
        {
            // Check copy H2D and D2H and D2D
            for (auto&[src, dst, msg] : copy_source_destination)
            {


                float copy_duration_ms_avg = 0;
                float copy_duration_ms_avg_host = 0;
                std::cout << "------"<<std::endl;
                for (size_t i = 0; i < iter; i++)
                {

                    std::cout << "Iteration:\t[" << i << "/" << matrix_data.iterations() << "]\r" << std::flush;
                    const auto begin = std::chrono::system_clock::now();

                    clAppEvent copy_wait;
                    //matrix_a->benchmark_memory();
                    err = matrix_a->copy_async({}, copy_wait, 0, matrix_a->size(), src, dst);
                    if (err != CL_SUCCESS)return err;

                    err = copy_wait.wait();
                    if (err != CL_SUCCESS)return err;

                    auto copy_duration_ms = copy_wait.duration(ms_scale_factor);
                    copy_duration_ms_avg += copy_duration_ms;

                    const auto end = std::chrono::system_clock::now();
                    const std::chrono::duration<double> diff = end - begin;
                    copy_duration_ms_avg_host+=(diff.count()*1e3f);
                }

                copy_duration_ms_avg = copy_duration_ms_avg / static_cast<float>(iter);
                std::cout << msg << ":\t " << std::setw(4) << copy_duration_ms_avg << " ms" << std::endl;
                std::cout << "Host duration:\t" << copy_duration_ms_avg_host / static_cast<float>(iter) << " ms" << std::endl;

            }
            std::cout << "-----------------------------------------------" << std::endl;
        }


        task_mat_add_sub->clear_dependences();

        auto proc_id = 0;
        for (auto& off_proc : proc)
        {

            float duration_ms_task_add_avg = 0;
            float duration_ms_task_merge_avg = 0;

            std::cout << "PAUSE 1 sec ..." << std::endl;
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(1000ms);

            std::cout << "Processing ..." << std::endl;
            for (size_t i = 0; i < iter; i++)
            {
                std::cout << "Iteration:\t[" << i << "/" << matrix_data.iterations() << "]\r" << std::flush;
                err = device.execute_async(task_mat_add_sub, off_proc, global_sizes, local_sizes,
                                           matrix_results, matrix_atxe, matrix_etxa, matrix_etxgxe, matrix_data._matrix_width, matrix_data._matrix_height);
                if (err != CL_SUCCESS)return err;

                err = device.execute_async(task_mat_merge, off_proc, global_sizes, local_sizes,
                                           matrix_results, matrix_atxe, matrix_data._matrix_width, matrix_data._matrix_height);
                if (err != CL_SUCCESS)return err;

                auto duration_ms_task_add = task_mat_add_sub->duration(ms_scale_factor);
                duration_ms_task_add_avg += duration_ms_task_add;

                auto duration_ms_task_merge = task_mat_merge->duration(ms_scale_factor);
                duration_ms_task_merge_avg += duration_ms_task_merge;

                task_mat_add_sub->wait_clear_events();
                task_mat_merge->wait_clear_events();
            }

            duration_ms_task_add_avg = duration_ms_task_add_avg / static_cast<float>(iter);
            duration_ms_task_merge_avg = duration_ms_task_merge_avg / static_cast<float>(iter);

            std::string dev_name = "";
            auto devs = device.sub_devices();
            float sum_off = 0;

            for (auto&[ctx, dev] : devs)
            {
                for (auto& dev_off : off_proc)
                {
                    const auto&[of, dt_did] = dev_off;
                    const auto&[dt, did] = dt_did;
                    if (dev.device_id() == did && dev.device_type() == dt)
                    {
                        std::cout << "Offload:\t" << of << ", device:\t" << dev.name() << std::endl;

                        sum_off += of;
                        dev_name.append(dev.name());
                        if (1.0f - sum_off < 0.001f)break;
                        dev_name.append("+");
                    }
                }
            }

            std::cout << "PROC [" << dev_name << "] " << "Task_ADD:\t" << std::setw(4) << duration_ms_task_add_avg << " ms" << std::endl;
            std::cout << "PROC [" << dev_name << "] " << "Task_MERGE:\t" << std::setw(4) << duration_ms_task_merge_avg << " ms" << std::endl;


            std::cout << "-----------------------------------------------" << std::endl;
            proc_id++;
        }
        return err;
    };


    return call_benchmark(proc_setups, copy_setups, matrix_data.iterations(), benchmark_transfers);

}

static int benchmark_merge_host(const size_t size, const size_t iter)
{
    int err = 0;
    std::vector<std::uint8_t> a(size, 1);
    std::vector<std::uint8_t> b(size, 1);

    const auto cnt_iter = iter < 1 ? 1 : iter;
    float avg_time = 0;

    for (int i = 0; i < cnt_iter; i++)
    {
        auto start = std::chrono::system_clock::now();
        std::transform(std::execution::par_unseq,
                       a.begin(), a.end(),
                       b.begin(), b.begin(), [](auto a, auto b) { return (a + b); });

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = end - start;
        avg_time += diff.count()*1e3f;
    }

    avg_time = avg_time / cnt_iter;
    std::cout << "CPU_Merge:\t" << std::setw(4) << avg_time << " ms" << std::endl;
    return err;
}

static void benchmark_cpu_mat_transponse(const int mat_width, const int mat_height, const size_t cnt_iter = 10)
{
    constexpr auto block_size = 16;
    int lda = ROUND_UP(mat_height, block_size);
    int ldb = ROUND_UP(mat_width, block_size);

    float* A = (float*)_mm_malloc(sizeof(float) * lda * ldb, 64);
    float* B = (float*)_mm_malloc(sizeof(float) * lda * ldb, 64);

    float avg_time = 0;

    for (int i = 0; i < cnt_iter; i++) {

        auto start = std::chrono::system_clock::now();

        transpose_block_SSE4x4(A, B, mat_height, mat_width, lda, ldb, block_size);

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = end - start;
        avg_time += diff.count() * 1e3f;
    }

    if (cnt_iter > 0)
        avg_time = avg_time / cnt_iter;

    std::cout << "CPU_MAT_TR:\t" << std::setw(4) << avg_time << " ms" << std::endl;

    _mm_free(A);
    _mm_free(B);
}

    int skmd::Benchmark_obj061(mkmd_input matrix_data)
{
    const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
    const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
    const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
    const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);

        offload_info gpu0_50 = { {0.5f,gpu0 } };
    offload_info gpu1_50 = { {0.5f,gpu1 } };
    offload_info gpu2_50 = { {0.5f,gpu2 } };

        offload_info gpu0_25 = { {0.25f,gpu0 } };
    offload_info gpu1_25 = { {0.25f,gpu1 } };
    offload_info gpu2_25 = { {0.25f,gpu2 } };

        offload_info gpu0_44 = { {0.44f,gpu0 } };
    offload_info gpu1_38 = { {0.38f,gpu1 } };
    offload_info gpu2_18 = { {0.18f,gpu2 } };

        offload_info gpu0_42 = { {0.42f,gpu0 } };
    offload_info gpu1_41 = { {0.41f,gpu1 } };
    offload_info gpu2_17 = { {0.17f,gpu2 } };

        offload_info gpu0_35 = { {0.35f,gpu0 } };
    offload_info gpu1_43 = { {0.43f,gpu1 } };

        offload_info gpu0_37 = { {0.37f,gpu0 } };
    offload_info gpu2_22 = { {0.22f,gpu2 } };

        offload_info gpu0_gpu1 = {{0.55f,gpu0 },{0.45f,gpu1 }};
	offload_info gpu0_gpu1_gpu2_a = {{0.4f,gpu0 }, {0.4f,gpu1 }, {0.2f,gpu2 }};		
    offload_info gpu0_gpu1_gpu2_c = { {0.33f,gpu0 }, {0.33f,gpu1 }, {0.34f,gpu2 } };
    offload_info gpu0_gpu1_gpu2_d = { {0.25f,gpu0 }, {0.25f,gpu1 }, {0.5f,gpu2 } };// this setup exposes a some strange BUG!
    offload_info gpu0_gpu1_gpu2_e = { {0.42f,gpu0 }, {0.41f,gpu1 }, {0.17f,gpu2 } };
    offload_info gpu0_gpu1_gpu2_f = { {0.41f,gpu0 }, {0.42f,gpu1 }, {0.17f,gpu2 } };
    offload_info gpu0_gpu1_gpu2_g = { {0.38f,gpu0 }, {0.44f,gpu1 }, {0.18f,gpu2 } };
    offload_info gpu0_gpu1_gpu2_h = { {0.35f,gpu0 }, {0.43f,gpu1 }, {0.22f,gpu2 } };

        std::vector<offload_info> obj_061_proc = { { {1.0f,cpu} }, {{1.0f,gpu0} }, {{1.0f,gpu1} } };


        //   std::vector<offload_info> obj_061_proc ={
    //	gpu0,gpu1,gpu2,
        //	//gpu0_gpu1,gpu0_50,gpu1_50,gpu2_50,gpu0_25,gpu1_25,gpu2_25,
        //	gpu0_gpu1_gpu2_a,gpu0_gpu1_gpu2_c, gpu0_gpu1_gpu2_d,gpu0_gpu1_gpu2_e,gpu0_gpu1_gpu2_d,gpu0_gpu1_gpu2_f,gpu0_gpu1_gpu2_d,gpu0_gpu1_gpu2_g,gpu0_gpu1_gpu2_h,
        //	//gpu0_44,gpu1_38,gpu2_18,
        //	//gpu0_42,gpu1_41,gpu2_17,
        //	//gpu0_37,gpu1_41,gpu2_22,
        //	//gpu0_35,gpu1_43,gpu2_22,
        //};

        std::vector<std::tuple<map_device_info,map_device_info,std::string>> obj_061_copy_setups=
            {
                { {cpu }, {gpu0 },"CPU->GPU0"},
                { {cpu }, {gpu1 },"CPU->GPU1"},
                { {cpu }, {gpu2 },"CPU->GPU2"},

                { {gpu0 }, {cpu },"GPU0->CPU"},
                { {gpu1 }, {cpu },"GPU1->CPU"},
                { {gpu2 }, {cpu },"GPU2->CPU"},

                { {gpu0 }, {gpu1 },"GPU0->GPU1"},
                { {gpu0 }, {gpu2 },"GPU0->GPU2"},
                { {gpu1 }, {gpu2 },"GPU1->GPU2"},
                };

    const auto size = matrix_data.items() * sizeof(float);
    auto err = benchmark_merge_host(size, matrix_data.iterations());
    if (err != 0)return err;

    benchmark_cpu_mat_transponse(matrix_data._matrix_width, matrix_data._matrix_height, matrix_data.iterations());

    return Benchmark_platform(matrix_data,obj_061_proc,obj_061_copy_setups,1);
}

    int skmd::Benchmark_obj119(mkmd_input matrix_data)
{
    const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
    const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
    const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
    const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);


        std::vector<offload_info> offloads = { { {1.0f,cpu} }, {{1.0f,gpu0} }, {{1.0f,gpu1} } };

        std::vector<std::tuple<map_device_info, map_device_info, std::string>> obj_copy_setups =
            {
                { {cpu }, {gpu0 },"CPU->GPU0"},
                { {gpu0 }, {cpu },"GPU0->CPU"},
                };

    const auto size = matrix_data.items() * sizeof(float);
    auto err = benchmark_merge_host(size, matrix_data.iterations());
    if (err != 0)return err;

    benchmark_cpu_mat_transponse(matrix_data._matrix_width, matrix_data._matrix_height, matrix_data.iterations());

    return Benchmark_platform(matrix_data, offloads, obj_copy_setups, 1);
}

    int skmd::Benchmark_obj129(mkmd_input matrix_data)
{
    const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
    const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
    const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
    const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);


        offload_info gpu0_gpu1a = {{0.5f,gpu0 },{0.5f,gpu1 }};

        std::vector<offload_info> obj_129_proc = { { {1.0f,cpu} }, {{1.0f,gpu0} }, {{1.0f,gpu1} }, gpu0_gpu1a };
    //std::vector<offload_info> obj_129_proc = { gpu0,gpu1,gpu0_gpu1a };

        std::vector<std::tuple<map_device_info,map_device_info,std::string>> obj_129_copy_setups=
            {
                { cpu , gpu0 ,"CPU->GPU0"},
                { cpu , gpu1 ,"CPU->GPU1"},

                { gpu0 , cpu ,"GPU0->CPU"},
                { gpu1 , cpu ,"GPU1->CPU"},

                { gpu0 , gpu1 ,"GPU0->GPU1"},

                };

    const auto size = matrix_data.items() * sizeof(float);
    auto err = benchmark_merge_host(size, matrix_data.iterations());
    if (err != 0)return err;

    benchmark_cpu_mat_transponse(matrix_data._matrix_width, matrix_data._matrix_height, matrix_data.iterations());

    return Benchmark_platform(matrix_data,obj_129_proc,obj_129_copy_setups,1);
}

int skmd::Benchmark_MM(mkmd_input matrix_data)
{
    bool is_obj_061 = matrix_data.is_obj_061();

    const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
    const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
    const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
    const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);

    std::vector<offload_info> offloads;
    offload_info gpu_v0 = { {1.0f,{CL_DEVICE_TYPE_GPU,matrix_data.gpu_id()} } }; // single gpu

    if (is_obj_061)
    {

        //CPU+GPU				2048x2048 matrix 50 it.
        offload_info gpu_v01 = { {1.0f,cpu } };
        offload_info gpu_v02 = { {1.0f,gpu0 } };
        offload_info gpu_v1 = { {0.1f,cpu },{0.9f,gpu0 } };
        offloads = { gpu_v01,gpu_v02,gpu_v1 };
    }
    else if(matrix_data.is_obj_129())
    {
        //2 GPUS				2048x2048 matrix 50 it.
        offload_info gpu_v01 = { {1.0f,cpu } }; // 172.7ms 1/x=0.0058 //115W ->19.78 J
        offload_info gpu_v02 = { {1.0f,gpu0 } }; // 43.3ms 1/x=0.0232 //90W ->0.387 J
        offload_info gpu_v03 = { {1.0f,gpu1 } }; // 46.9ms 1/x=0.0217 //120W -> 7.52 J

        // sum_all = 0.0507 (11,46,43)%
        // sum_gpu0_gpu1 = 0.0449 (0,52,48)%
        // sum_cpu_gpu0 = 0.029 (20,80,0)%
        // sum_cpu_gpu1 = 0.0275 (23,0,77)%

        offload_info gpu_v1 = { {0.11f,cpu },{0.46f,gpu0 },{0.43f,gpu1 } }; // 17ms (105,91,85) 281W 4.77 J
        offload_info gpu_v2 = { {0.52f,gpu0 },{0.48f,gpu1 } }; // 22ms (12,94,88)  194W -> 4.2 J
        offload_info gpu_v3 = { {0.06f,cpu },{0.94f,gpu0 } }; // 40ms (105,78,12) 195W -> 7.8 J
        offload_info gpu_v4 = { {0.06f,cpu },{0.94f,gpu1 } }; // 43ms (110,9,152) 271W -> 11.6 J

        offloads = { gpu_v01,gpu_v02,gpu_v03,
                    gpu_v1,gpu_v2,gpu_v3,gpu_v4 };
    }
    else if (matrix_data.is_obj_119())
    {
        //3 GPUS
        offload_info gpu_v1 = { {1.0f,cpu } };
        offload_info gpu_v2 = { {1.0f,gpu0 } };
        offload_info gpu_v3 = { {1.0f,gpu1 } };
        offload_info gpu_v4 = { {1.0f,gpu2 } };

        offload_info gpu_v5 = { {0.02f,cpu },{0.5f,gpu0 },
                               {0.3f,gpu1 },{0.18f,gpu2 } }; // 4.4ms

        offloads = { gpu_v1,gpu_v2,gpu_v3,gpu_v4, gpu_v5 };

    }

    std::string task_MAT_MUL = "";
    task_MAT_MUL.append(kernels::task_MM);


    auto& device = *matrix_data.device();

    std::cout << "ALLOCATE MEMORY ..." << std::endl;

    const auto items_in = matrix_data.items();
    std::vector<float> random_values_a(items_in);
    generate_rand_real(random_values_a, 0.01f, 1.0f);

    //allocate memory
    auto matrix_inA = device.alloc<float>(random_values_a, true);//read_only
    if (!matrix_inA)return COOPCL_BAD_ALLOC;

    auto matrix_inB = device.alloc<float>(random_values_a, true);//read_only
    if (!matrix_inB)return COOPCL_BAD_ALLOC;


    //------------------------------------------ READ_WRITE

    auto matrix_out = device.alloc<float>(items_in);
    if (!matrix_out)return COOPCL_BAD_ALLOC;

    const auto size_input_MB = (matrix_inA->size() + matrix_inB->size()) * 1e-6;
    const auto size_out_MB = (matrix_out->size() ) * 1e-6;

    std::cout << "Input memory size:\t" << std::setw(3) << size_input_MB << " MB" << std::endl;
    std::cout << "Output memory size:\t" << std::setw(3) << size_out_MB << " MB" << std::endl;
    std::cout << "--------------------\n";

    std::cout << "BUILD KERNELS..." << std::endl;

    const auto BLOCK_SIZE = 16;

    //Create tasks
    std::string jit_flags = " -cl-unsafe-math-optimizations -cl-fast-relaxed-math -DT=float -DBLOCK_SIZE=";
    jit_flags.append(std::to_string(BLOCK_SIZE));

    const std::string kname = "matrixMul";

    auto task_MM = device.create_task(task_MAT_MUL, kname, jit_flags);
    if (!task_MM)throw std::runtime_error("Error JTI, FIMXE!!!");

    std::array<size_t, 3> global_sizes = { static_cast<size_t>(matrix_data.width()), static_cast<size_t>(matrix_data.height()),1 };
    std::array<size_t, 3> local_sizes = { BLOCK_SIZE,BLOCK_SIZE,1 };


    const auto ms_scale_factor = 1e-6f;
    int err = 0;

    const size_t iter = matrix_data.iterations() < 1 ? 1 : matrix_data.iterations();
    task_MM->clear_dependences();

    auto proc_id = 0;
    for (auto& off_proc : offloads)
    {
        float duration_ms_task = 0;

#ifdef _PAUSE_
        std::cout << "PAUSE 10 sec ..." << std::endl;
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(10000ms);
#endif
        std::cout << "Processing ..." << std::endl;
        for (size_t i = 0; i < iter; i++)
        {
            std::cout << "Iteration:\t[" << i << "/" << matrix_data.iterations() << "]\r" << std::flush;
            task_MM->wait_clear_events();
            const int mat_w = matrix_data.width();
            const int mat_h = matrix_data.height();

            if (kname == "mat_mul_no_opt")
            {
                //kernel void mat_mul(global T * restrict C, const global T * restrict A, const global T * restrict B, const int mat_width, const int mat_height)
                err = device.execute_async(task_MM, off_proc, global_sizes, local_sizes,
                                           matrix_out, matrix_inA, matrix_inB, mat_w, mat_h);
            }
            else if (kname=="matrixMul")
            {

                //NOTE: with this init. the code expects a square matrices A,B,C !
                //const int uiWA = matrix_data.width();
                //const int uiWB = uiWA;
                //const int trueLocalSize1 = matrix_data.height();
                const auto local_mem = BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
				cl::LocalSpaceArg lmem = cl::Local(local_mem);

                //kernel void matrixMul( global float* C, global float* A, global float* B, local float* As, local float* Bs, const int uiWA, const int uiWB, const int trueLocalSize1)
                err = device.execute_async(task_MM, off_proc, global_sizes, local_sizes,
                                           matrix_out, matrix_inA, matrix_inB, lmem, lmem, mat_w, mat_w, mat_h);
            }
            if (err != CL_SUCCESS)return err;

            const auto duration_ms = task_MM->duration(ms_scale_factor);
            duration_ms_task += duration_ms;
        }

        duration_ms_task = duration_ms_task / static_cast<float>(iter);

        std::string dev_name = "";
        auto devs = device.sub_devices();
        float sum_off = 0;

        for (auto& [ctx, dev] : devs)
        {
            for (auto& dev_off : off_proc)
            {
                const auto& [of, dt_did] = dev_off;
                const auto& [dt, did] = dt_did;
                if (dev.device_id() == did && dev.device_type() == dt)
                {
                    std::cout << "Offload:\t" << of << ", device:\t" << dev.name() << std::endl;

                    sum_off += of;
                    dev_name.append(dev.name());
                    if (1.0f - sum_off < 0.001f)break;
                    dev_name.append("+");
                }
            }
        }

        std::cout << "PROC [" << dev_name << "] " << "Task_MM:\t" << std::setw(4) << duration_ms_task << " ms" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
        proc_id++;
    }
    return err;

    //return Benchmark_platform(matrix_data, offloads , {}, 0, 1, 0);
    //return 0;
}

int skmd::Benchmark_MA(mkmd_input matrix_data) 
{
    bool is_obj_061 = matrix_data.is_obj_061();

    const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
    const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
    const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
    const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);

    std::vector<offload_info> offloads;
    offload_info gpu_v0 = { {1.0f,{CL_DEVICE_TYPE_GPU,matrix_data.gpu_id()} } }; // single gpu

    if (is_obj_061)
    {
        //3 GPUS
        offload_info gpu_v1 = { {1.0f,cpu } };
        offload_info gpu_v2 = { {1.0f,gpu0 } };
        offload_info gpu_v3 = { {1.0f,gpu1 } };
        offload_info gpu_v4 = { {1.0f,gpu2 } };
        offloads = { gpu_v1,gpu_v2,gpu_v3,gpu_v4 };
    }
    else
    {
        //2 GPUS
        offload_info gpu_v1 = { {1.0f,cpu } };
        offload_info gpu_v2 = { {1.0f,gpu0 } };
        offload_info gpu_v3 = { {1.0f,gpu1 } };
        offloads = { gpu_v1,gpu_v2,gpu_v3 };
    }
    return 0;//TODO: Benchmark_platform(matrix_data, offloads, {}, 0, 0, 1);
}

int skmd::Benchmark_MV(mkmd_input matrix_data, bool apply_vector_vector)
{
    const static std::string task_MV =
        R"(
	// #define BLOCK_SIZE 2048 // vector size, items
	
	#define T float

  __kernel void MV_2(
	global T* restrict resultVector,
	const global T* restrict matrixA,
	const global T* restrict vectorB,
	const int width_A)
	{
		const int tx = get_global_id(0);
		__local float vectB[BLOCK_SIZE];

		event_t copy_event = async_work_group_copy(vectB, vectorB, BLOCK_SIZE, 0);
		wait_group_events(1, &copy_event);

		float value = 0;
		#pragma unroll 128
		for (unsigned int k = 0; k < BLOCK_SIZE; ++k) {
			value += matrixA[tx * width_A + k] * vectB[k];
		}

		resultVector[tx] = value;
	}

	__kernel void MV(const global T* restrict dA, const global T * restrict dx, global T * restrict dy, const int nRows, const int nCols)
	{
		const int tid = get_global_id(0);
		const  int lidx = get_local_id(0);
		
		local T x_shared[BLOCK_SIZE];

		T y_val = 0.0;

		#pragma unroll 128
		for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
		{
			if ((m * BLOCK_SIZE + lidx) < nCols) 
				x_shared[lidx] = dx[lidx + m * BLOCK_SIZE];
			else
				x_shared[lidx] = 0.f;

			barrier(CLK_LOCAL_MEM_FENCE);

			#pragma unroll 128
			for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
				// --- Column-major ordering - faster
				y_val += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
				// --- Row-major ordering - slower
				//y_val += dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
			}

			barrier(CLK_LOCAL_MEM_FENCE);
		}

		if (tid < nRows) dy[tid] = y_val;
	}

)";

    bool is_obj_061 = matrix_data.is_obj_061();

    const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
    const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
    const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
    const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);

    std::vector<offload_info> offloads;
    offload_info gpu_v0 = { {1.0f,{CL_DEVICE_TYPE_GPU,matrix_data.gpu_id()} } }; // single gpu

    if (is_obj_061)
    {
        //3 GPUS
        offload_info gpu_v1 = { {1.0f,cpu } };
        offload_info gpu_v2 = { {1.0f,gpu0 } };
        offload_info gpu_v3 = { {1.0f,gpu1 } };
        offload_info gpu_v4 = { {1.0f,gpu2 } };
        offloads = { gpu_v1,gpu_v2,gpu_v3,gpu_v4 };
    }
    else
    {
        //2 GPUS				2Kx2K matrix 2K vec
        offload_info gpu_v01 = { {1.0f,cpu } }; // 0.66ms
        offload_info gpu_v02 = { {1.0f,gpu0 } }; // 0.84ms
        offload_info gpu_v03 = { {1.0f,gpu1 } }; // 0.78ms

        offloads = { gpu_v01,gpu_v02,gpu_v03};
    }

    auto& device = *matrix_data.device();

    std::cout << "ALLOCATE MEMORY ..." << std::endl;

    const auto items_in = apply_vector_vector==true?matrix_data.width():matrix_data.items();
    const auto vector_items = matrix_data.width();
    std::vector<float> random_values_a(items_in);
    generate_rand_real(random_values_a, 0.01f, 1.0f);

    std::vector<float> random_values_b(vector_items);
    generate_rand_real(random_values_b, 0.01f, 1.0f);

    //allocate memory
    auto matrix_in = device.alloc<float>(random_values_a, true);//read_only
    if (!matrix_in)return COOPCL_BAD_ALLOC;

    auto vector_in = device.alloc<float>(random_values_b, true);//read_only
    if (!matrix_in)return COOPCL_BAD_ALLOC;

    //------------------------------------------ READ_WRITE

    auto matrix_out = device.alloc<float>(vector_items);
    if (!matrix_out)return COOPCL_BAD_ALLOC;


    const auto size_input_MB = (matrix_in->size() + vector_in->size()) * 1e-6;
    const auto size_out_MB = (matrix_out->size() ) * 1e-6;

    std::cout << "Input memory size:\t" << std::setw(3) << size_input_MB << " MB" << std::endl;
    std::cout << "Output memory size:\t" << std::setw(3) << size_out_MB << " MB" << std::endl;
    std::cout << "--------------------\n";

    std::cout << "BUILD KERNELS..." << std::endl;

    //Create tasks
    std::string jit_flags = " -cl-unsafe-math-optimizations -cl-fast-relaxed-math -DBLOCK_SIZE=";
    jit_flags.append(std::to_string(vector_items));

    const auto kname = "MV";
	auto task_MV_ = device.create_task(task_MV, kname, jit_flags);	
	if (!task_MV_)throw std::runtime_error("Error JTI, FIMXE!!!");

    std::array<size_t, 3> global_sizes = { static_cast<size_t>(apply_vector_vector==true?matrix_data.width():matrix_data.height()), 1,1 };
    std::array<size_t, 3> local_sizes = { 256,1,1 };


    const auto ms_scale_factor = 1e-6f;
    int err = 0;

    const size_t iter = matrix_data.iterations() < 1 ? 1 : matrix_data.iterations();
    task_MV_->clear_dependences();

    auto proc_id = 0;
    for (auto& off_proc : offloads)
    {
        float duration_ms_task = 0;

#ifdef _PAUSE_
        std::cout << "PAUSE 10 sec ..." << std::endl;
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(10000ms);
#endif
        std::cout << "Processing ..." << std::endl;
        for (size_t i = 0; i < iter; i++)
        {
            std::cout << "Iteration:\t[" << i << "/" << matrix_data.iterations() << "]\r" << std::flush;
            task_MV_->wait_clear_events();
            const int mat_w = matrix_data.width();
            const int mat_h = apply_vector_vector==true?1:matrix_data.height();

            if (kname == "MV_2")
            {
                //__kernel void MV_2(global float* resultVector,global float* matrixA,global float* vectorB,const int width_A)
                err = device.execute_async(task_MV_, off_proc, global_sizes, local_sizes,
                                           matrix_out, matrix_in, vector_in, mat_w);			}
            else
            {
                //__kernel void MV(const global T* restrict dA, const global T * restrict dx, global T * restrict dy, const int nRows, const int nCols)
                err = device.execute_async(task_MV_, off_proc, global_sizes, local_sizes,
                                           matrix_in, vector_in, matrix_out, mat_h, mat_w);
            }
            if (err != CL_SUCCESS)return err;

            const auto duration_ms = task_MV_->duration(ms_scale_factor);
            duration_ms_task += duration_ms;
        }

        duration_ms_task = duration_ms_task / static_cast<float>(iter);

        std::string dev_name = "";
        auto devs = device.sub_devices();
        float sum_off = 0;

        for (auto& [ctx, dev] : devs)
        {
            for (auto& dev_off : off_proc)
            {
                const auto& [of, dt_did] = dev_off;
                const auto& [dt, did] = dt_did;
                if (dev.device_id() == did && dev.device_type() == dt)
                {
                    std::cout << "Offload:\t" << of << ", device:\t" << dev.name() << std::endl;

                    sum_off += of;
                    dev_name.append(dev.name());
                    if (1.0f - sum_off < 0.001f)break;
                    dev_name.append("+");
                }
            }
        }

        std::cout << "PROC [" << dev_name << "] " << "Task_MV:\t" << std::setw(4) << duration_ms_task << " ms" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
        proc_id++;
    }
    return err;
}

int skmd::Benchmark_BL(mkmd_input matrix_data)
{
    const static std::string task_BLUR =
        R"(

#define T float

kernel
void gaussian_blur(const global T* restrict inputChannel,
				   global T* restrict outputChannel,
				   const int numRows, const int numCols,
				   const global T* const filter, const int filterWidth)
{
  //naive not optimized Gauss blur filter

  const int px = get_global_id(0);
  const int py = get_global_id(1);
  if (px >= numCols || py >= numRows) {
	  return;
  }

  float c = 0.0f;

  for (int fx = 0; fx < filterWidth; fx++) {
	for (int fy = 0; fy < filterWidth; fy++) {
	  int imagex = px + fx - filterWidth / 2;
	  int imagey = py + fy - filterWidth / 2;
	  imagex = min(max(imagex,0),numCols-1);
	  imagey = min(max(imagey,0),numRows-1);
	  c += (filter[fy*filterWidth+fx] * inputChannel[imagey*numCols+imagex]);
	}
  }
  outputChannel[py*numCols+px] = c;
}
)";
    bool is_obj_061 = matrix_data.is_obj_061();

    const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
    const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
    const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
    const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);

    std::vector<offload_info> offloads;
    offload_info gpu_v0 = { {1.0f,{CL_DEVICE_TYPE_GPU,matrix_data.gpu_id()} } }; // single gpu

    if (is_obj_061)
    {
        //3 GPUS
        offload_info gpu_v1 = { {1.0f,cpu } };
        offload_info gpu_v2 = { {1.0f,gpu0 } };
        offload_info gpu_v3 = { {1.0f,gpu1 } };
        offload_info gpu_v4 = { {1.0f,gpu2 } };
        offloads = { gpu_v1,gpu_v2,gpu_v3,gpu_v4 };
    }
    else
    {
        //2 GPUS				2Kx2K matrix
        offload_info gpu_v01 = { {1.0f,cpu } }; // 10.3ms 1/x=0.0970
        offload_info gpu_v02 = { {1.0f,gpu0 } }; // 4.1ms 1/x=0.243
        offload_info gpu_v03 = { {1.0f,gpu1 } }; // 3.5ms 1/x=0.285

        // sum_all = 0.625
        // sum_gpu0_gpu1 = 0.528
        // sum_cpu_gpu0 = 0.34
        // sum_cpu_gpu1 = 0.382
        offload_info gpu_v1 = { {0.15f,cpu },{0.39f,gpu0 },{0.45f,gpu1 } }; //1.7
        offload_info gpu_v2 = { {0.46f,gpu0 },{0.54f,gpu1 } }; //1.9
        offload_info gpu_v3 = { {0.28f,cpu },{0.72f,gpu0 } }; //3.1
        offload_info gpu_v4 = { {0.25f,cpu },{0.75f,gpu1 } }; //2.7

        offloads = { gpu_v01,gpu_v02,gpu_v03,
                    gpu_v1,gpu_v2,gpu_v3,gpu_v4 };
    }

    auto& device = *matrix_data.device();

    std::cout << "ALLOCATE MEMORY ..." << std::endl;

    const auto items_in = matrix_data.items();

    std::vector<float> random_values_a(items_in);
    const int f_radius = 5;
    const int fw = 2 * f_radius + 1;
    std::vector<float> h_filter(fw, 1.1f);
    generate_rand_real(random_values_a, 0.01f, 1.0f);

    //allocate memory
    auto matrix_in = device.alloc<float>(random_values_a, true);//read_only
    if (!matrix_in)return COOPCL_BAD_ALLOC;

    auto matrix_filter = device.alloc<float>(h_filter, true);//read_only
    if (!matrix_filter)return COOPCL_BAD_ALLOC;

    //------------------------------------------ READ_WRITE

    auto matrix_out = device.alloc<float>(items_in);//read_only
    if (!matrix_out)return COOPCL_BAD_ALLOC;

    const auto size_input_MB = (matrix_in->size() + matrix_filter->size()) * 1e-6;
    const auto size_out_MB = (matrix_out->size() ) * 1e-6;

    std::cout << "Input memory size:\t" << std::setw(3) << size_input_MB << " MB" << std::endl;
    std::cout << "Output memory size:\t" << std::setw(3) << size_out_MB << " MB" << std::endl;
    std::cout << "--------------------\n";


    std::cout << "BUILD KERNELS..." << std::endl;

    //Create tasks
    std::string jit_flags = " -cl-unsafe-math-optimizations -cl-fast-relaxed-math";

    auto task_BL = device.create_task(task_BLUR, "gaussian_blur", jit_flags);
    if (!task_BL)throw std::runtime_error("Error JTI, FIMXE!!!");

    std::array<size_t, 3> global_sizes = { static_cast<size_t>(matrix_data._matrix_width), static_cast<size_t>(matrix_data._matrix_height),1 };
    std::array<size_t, 3> local_sizes = { BLOCK_DIM_global,BLOCK_DIM_global,1 };

    const auto ms_scale_factor = 1e-6f;
    int err = 0;

    const size_t iter = matrix_data.iterations() < 1 ? 1 : matrix_data.iterations();
    task_BL->clear_dependences();

    auto proc_id = 0;
    for (auto& off_proc : offloads)
    {
        float duration_ms_task_blur_avg = 0;

#ifdef _PAUSE_
        std::cout << "PAUSE 10 sec ..." << std::endl;
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(10000ms);
#endif
        std::cout << "Processing ..." << std::endl;
        for (size_t i = 0; i < iter; i++)
        {
            std::cout << "Iteration:\t[" << i << "/" << matrix_data.iterations() << "]\r" << std::flush;
            task_BL->wait_clear_events();

            /*void gaussian_blur(const global float* const inputChannel,
				global float* const outputChannel,
				const int numRows, const int numCols,
				const global float* const filter, const int filterWidth)*/

            err = device.execute_async(task_BL, off_proc, global_sizes, local_sizes,
                                       matrix_in, matrix_out, matrix_data._matrix_height, matrix_data._matrix_width,
                                       matrix_filter, fw);
            if (err != CL_SUCCESS)return err;



            auto duration_ms_task_blur = task_BL->duration(ms_scale_factor);
            duration_ms_task_blur_avg += duration_ms_task_blur;
        }

        duration_ms_task_blur_avg = duration_ms_task_blur_avg / static_cast<float>(iter);

        std::string dev_name = "";
        auto devs = device.sub_devices();
        float sum_off = 0;

        for (auto&[ctx, dev] : devs)
        {
            for (auto& dev_off : off_proc)
            {
                const auto&[of, dt_did] = dev_off;
                const auto& [dt, did] = dt_did;
                if (dev.device_id() == did && dev.device_type() == dt)
                {
                    std::cout << "Offload:\t" << of << ", device:\t" << dev.name() << std::endl;

                    sum_off += of;
                    dev_name.append(dev.name());
                    if (1.0f - sum_off < 0.001f)break;
                    dev_name.append("+");
                }
            }
        }

        std::cout << "PROC [" << dev_name << "] " << "Task_GBLUR:\t" << std::setw(4) << duration_ms_task_blur_avg << " ms" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
        proc_id++;
    }
    return err;

}

int skmd::Benchmark_NB(mkmd_input matrix_data)
{
    const static std::string tasks_NB =
        R"(

	#define UNROLL_FACTOR  8
	__kernel void NB(
		const __global float4* restrict pos, 
		const __global float4* restrict vel, 
		const int numBodies, 
		const float deltaTime, 
		const float epsSqr, 
		__global float4* restrict newPosition, 
		__global float4* restrict newVelocity) {

	unsigned int gid = get_global_id(0);
	float4 myPos = pos[gid];
	float4 acc = (float4)0.0f;


	int i = 0;
	for (; (i + UNROLL_FACTOR) < numBodies; ) {
	#pragma unroll UNROLL_FACTOR
		for (int j = 0; j < UNROLL_FACTOR; j++, i++) {
			float4 p = pos[i];
			float4 r;
			r.xyz = p.xyz - myPos.xyz;
			float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;

			float invDist = 1.0f / sqrt(distSqr + epsSqr);
			float invDistCube = invDist * invDist * invDist;
			float s = p.w * invDistCube;

			// accumulate effect of all particles
			acc.xyz += s * r.xyz;
		}
	}
	for (; i < numBodies; i++) {
		float4 p = pos[i];

		float4 r;
		r.xyz = p.xyz - myPos.xyz;
		float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;

		float invDist = 1.0f / sqrt(distSqr + epsSqr);
		float invDistCube = invDist * invDist * invDist;
		float s = p.w * invDistCube;

		// accumulate effect of all particles
		acc.xyz += s * r.xyz;
	}

	float4 oldVel = vel[gid];

	// updated position and velocity
	float4 newPos;
	newPos.xyz = myPos.xyz + oldVel.xyz * deltaTime + acc.xyz * 0.5f * deltaTime * deltaTime;
	newPos.w = myPos.w;

	float4 newVel;
	newVel.xyz = oldVel.xyz + acc.xyz * deltaTime;
	newVel.w = oldVel.w;

	// write to global memory
	newPosition[gid] = newPos;
	newVelocity[gid] = newVel;
})";

    bool is_obj_061 = matrix_data.is_obj_061();

    const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
    const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
    const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
    const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);

    std::vector<offload_info> offloads;
    offload_info gpu_v0 = { {1.0f,{CL_DEVICE_TYPE_GPU,matrix_data.gpu_id()} } }; // single gpu

    if (is_obj_061)
    {
        //3 GPUS
        offload_info gpu_v1 = { {1.0f,cpu } };
        offload_info gpu_v2 = { {1.0f,gpu0 } };
        offload_info gpu_v3 = { {1.0f,gpu1 } };
        offload_info gpu_v4 = { {1.0f,gpu2 } };
        offloads = { gpu_v1,gpu_v2,gpu_v3,gpu_v4 };
    }
    else
    {
        //2 GPUS				256x256 matrix 50 it.
        offload_info gpu_v01 = { {1.0f,cpu } }; // 120.7ms 1/x=0.0082 //105W
        offload_info gpu_v02 = { {1.0f,gpu0 } }; // 31.3ms 1/x=0.0319 //90W
        offload_info gpu_v03 = { {1.0f,gpu1 } }; // 36.9ms 1/x=0.0271 //128W

        // sum_all = 0.0672 (12,48,40)%
        // sum_gpu0_gpu1 = 0.059 (0,54,46)%
        // sum_cpu_gpu0 = 0.0401 (20,80,0)%
        // sum_cpu_gpu1 = 0.0353 (23,0,77)%

        offload_info gpu_v1 = { {0.1f,cpu },{0.5f,gpu0 },{0.4f,gpu1 } }; // 14.7ms , (75,82,117) 274W-> 4.027 J
        offload_info gpu_v2 = { {0.54f,gpu0 },{0.46f,gpu1 } }; //16.5ms, (17,76,98) 191W -> 3.151 J
        offload_info gpu_v3 = { {0.20f,cpu },{0.8f,gpu0 } }; //28ms, (98,45,12) 155W ->4.340 J
        offload_info gpu_v4 = { {0.23f,cpu },{0.77f,gpu1 } }; //32ms, (98,9,110) 217W ->6.944 J

        offloads = { gpu_v01,gpu_v02,gpu_v03,
                    gpu_v1,gpu_v2,gpu_v3,gpu_v4 };
    }

    auto& device = *matrix_data.device();

    std::cout << "ALLOCATE MEMORY ..." << std::endl;


    /*__kernel void NB(
	const __global float4 * pos, 
	const __global float4 * vel, 
	const int numBodies, 
	const float deltaTime, 
	const float epsSqr, 
	__global float4 * newPosition, 
	__global float4 * newVelocity)*/

    const auto items_in = matrix_data.items();
    const int numBodies = items_in;
    std::vector<float> random_values_a(items_in*4);//cl_float4
    generate_rand_real(random_values_a, 0.01f, 1.0f);

    //allocate memory
    auto matrix_in1 = device.alloc<float>(random_values_a, true);//read_only
    if (!matrix_in1)return COOPCL_BAD_ALLOC;

    auto matrix_in2 = device.alloc<float>(random_values_a, true);//read_only
    if (!matrix_in2)return COOPCL_BAD_ALLOC;

    //------------------------------------------ READ_WRITE

    auto matrix_out1 = device.alloc<float>(items_in*4);
    if (!matrix_out1)return COOPCL_BAD_ALLOC;

    auto matrix_out2 = device.alloc<float>(items_in * 4);
    if (!matrix_out2)return COOPCL_BAD_ALLOC;


    const auto size_input_MB = (matrix_in1->size()+ matrix_in2->size()) * 1e-6;
    const auto size_out_MB = (matrix_out1->size() + matrix_out2->size()) * 1e-6;

    std::cout << "Input memory size:\t" << std::setw(3) << size_input_MB << " MB" << std::endl;
    std::cout << "Output memory size:\t" << std::setw(3) << size_out_MB << " MB" << std::endl;
    std::cout << "--------------------\n";

    std::cout << "BUILD KERNELS..." << std::endl;

    //Create tasks
    std::string jit_flags = " -cl-unsafe-math-optimizations -cl-fast-relaxed-math";

    auto task_NB = device.create_task(tasks_NB, "NB", jit_flags);
    if (!task_NB)throw std::runtime_error("Error JTI, FIMXE!!!");

    std::array<size_t, 3> global_sizes = { static_cast<size_t>(numBodies), 1,1 };
    std::array<size_t, 3> local_sizes = { 128,1,1 };

    const auto ms_scale_factor = 1e-6f;
    int err = 0;

    const size_t iter = matrix_data.iterations() < 1 ? 1 : matrix_data.iterations();
    task_NB->clear_dependences();

    const float dt = 0.1f;
    const float eps = 0.001f;

    auto proc_id = 0;
    for (auto& off_proc : offloads)
    {
        float duration_ms_task_blur_avg = 0;

#ifdef _PAUSE_
        std::cout << "PAUSE 10 sec ..." << std::endl;
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(10000ms);
#endif
        std::cout << "Processing ..." << std::endl;
        for (size_t i = 0; i < iter; i++)
        {
            std::cout << "Iteration:\t[" << i << "/" << matrix_data.iterations() << "]\r" << std::flush;
            task_NB->wait_clear_events();

            /*__kernel void NB(
	const __global float4 * pos,
	const __global float4 * vel,
	const int numBodies,
	const float deltaTime,
	const float epsSqr,
	__global float4 * newPosition,
	__global float4 * newVelocity)*/

            err = device.execute_async(task_NB, off_proc, global_sizes, local_sizes,
                                       matrix_in1, matrix_in2,
                numBodies,dt,eps,
                                       matrix_out1, matrix_out2);
            if (err != CL_SUCCESS)return err;

            auto duration_ms_task_blur = task_NB->duration(ms_scale_factor);
            duration_ms_task_blur_avg += duration_ms_task_blur;
        }

        duration_ms_task_blur_avg = duration_ms_task_blur_avg / static_cast<float>(iter);

        std::string dev_name = "";
        auto devs = device.sub_devices();
        float sum_off = 0;

        for (auto& [ctx, dev] : devs)
        {
            for (auto& dev_off : off_proc)
            {
                const auto& [of, dt_did] = dev_off;
                const auto& [dt, did] = dt_did;
                if (dev.device_id() == did && dev.device_type() == dt)
                {
                    std::cout << "Offload:\t" << of << ", device:\t" << dev.name() << std::endl;

                    sum_off += of;
                    dev_name.append(dev.name());
                    if (1.0f - sum_off < 0.001f)break;
                    dev_name.append("+");
                }
            }
        }

        std::cout << "PROC [" << dev_name << "] " << "Task_NB:\t" << std::setw(4) << duration_ms_task_blur_avg << " ms" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
        proc_id++;
    }
    return err;

}

int skmd::Benchmark_BS(mkmd_input matrix_data)
{

    const static std::string tasks_BS =
        R"(
	#define S_LOWER_LIMIT 10.0f
	#define S_UPPER_LIMIT 100.0f
	#define K_LOWER_LIMIT 10.0f
	#define K_UPPER_LIMIT 100.0f
	#define T_LOWER_LIMIT 1.0f
	#define T_UPPER_LIMIT 10.0f
	#define R_LOWER_LIMIT 0.01f
	#define R_UPPER_LIMIT 0.05f
	#define SIGMA_LOWER_LIMIT 0.01f
	#define SIGMA_UPPER_LIMIT 0.10f

	void phi_scalar(float X, float* phi)
	{
		float y;
		float absX;
		float t;
		float result;

		const float c1 = (float)0.319381530f;
		const float c2 = (float)-0.356563782f;
		const float c3 = (float)1.781477937f;
		const float c4 = (float)-1.821255978f;
		const float c5 = (float)1.330274429f;

		const float zero = (float)0.0f;
		const float one = (float)1.0f;
		const float two = (float)2.0f;
		const float temp4 = (float)0.2316419f;

		const float oneBySqrt2pi = (float)0.398942280f;

		absX = fabs(X);
		t = one / (one + temp4 * absX);

		y = one - oneBySqrt2pi * exp(-X * X / two) * t
			* (c1 + t
				* (c2 + t
					* (c3 + t
						* (c4 + t * c5))));

		result = (X < zero) ? (one - y) : y;

		*phi = result;
	}

	__kernel void BS(
		const __global float* restrict randArray,
		const int width,
		__global float* restrict call,
		__global float* restrict put)
	{
		float d1, d2;
		float phiD1, phiD2;
		float sigmaSqrtT;
		float KexpMinusRT;

		size_t xPos = get_global_id(0);
		size_t yPos = get_global_id(1);
		float two = (float)2.0f;
		float inRand = randArray[yPos * width + xPos];
		float S = S_LOWER_LIMIT * inRand + S_UPPER_LIMIT * (1.0f - inRand);
		float K = K_LOWER_LIMIT * inRand + K_UPPER_LIMIT * (1.0f - inRand);
		float T = T_LOWER_LIMIT * inRand + T_UPPER_LIMIT * (1.0f - inRand);
		float R = R_LOWER_LIMIT * inRand + R_UPPER_LIMIT * (1.0f - inRand);
		float sigmaVal = SIGMA_LOWER_LIMIT * inRand + SIGMA_UPPER_LIMIT * (1.0f - inRand);


		sigmaSqrtT = sigmaVal * sqrt(T);

		d1 = (log(S / K) + (R + sigmaVal * sigmaVal / two) * T) / sigmaSqrtT;
		d2 = d1 - sigmaSqrtT;

		KexpMinusRT = K * exp(-R * T);
		phi_scalar(d1, &phiD1);
		phi_scalar(d2, &phiD2);
		call[yPos * width + xPos] = S * phiD1 - KexpMinusRT * phiD2;
		phi_scalar(-d1, &phiD1);
		phi_scalar(-d2, &phiD2);
		put[yPos * width + xPos] = KexpMinusRT * phiD2 - S * phiD1;


	})";

    bool is_obj_061 = matrix_data.is_obj_061();

    const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
    const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
    const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
    const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);

    std::vector<offload_info> offloads;
    offload_info gpu_v0 = { {1.0f,{CL_DEVICE_TYPE_GPU,matrix_data.gpu_id()} } }; // single gpu

    if (is_obj_061)
    {
        //3 GPUS
        offload_info gpu_v1 = { {1.0f,cpu } };
        offload_info gpu_v2 = { {1.0f,gpu0 } };
        offload_info gpu_v3 = { {1.0f,gpu1 } };
        offload_info gpu_v4 = { {1.0f,gpu2 } };
        offloads = { gpu_v1,gpu_v2,gpu_v3,gpu_v4 };
    }
    else
    {
        //2 GPUS				2Kx2K matrix 50  it.
        offload_info gpu_v01 = { {1.0f,cpu } }; // 12.5ms 1/x=0.08	66W -> 825J
        offload_info gpu_v02 = { {1.0f,gpu0 } }; // 14.6ms 1/x=0.068 69W -> 1007 J
        offload_info gpu_v03 = { {1.0f,gpu1 } }; // 19.3ms 1/x=0.051 106W -> 2045 J

        // sum_all = 0.1998 (40,34,26)
        // sum_gpu0_gpu1 = 0.119 (0,57,43)
        // sum_cpu_gpu0 = 0.148 (54,46,0)
        // sum_cpu_gpu1 = 0.131 (61,0,39)
        offload_info gpu_v1 = { {0.4f,cpu },{0.34f,gpu0 },{0.26f,gpu1 } }; //5.8ms (57,38,44) 139W -> 0.806 J
        offload_info gpu_v2 = { {0.57f,gpu0 },{0.43f,gpu1 } }; //7.9ms (12,29,44) 85W ->0.671 J
        offload_info gpu_v3 = { {0.54f,cpu },{0.46f,gpu0 } }; //7.8
        offload_info gpu_v4 = { {0.61f,cpu },{0.39f,gpu1 } }; //7.7

        offloads = { gpu_v01,gpu_v02,gpu_v03,
                    gpu_v1,gpu_v2,gpu_v3,gpu_v4 };
    }

    auto& device = *matrix_data.device();

    std::cout << "ALLOCATE MEMORY ..." << std::endl;

    auto samples = matrix_data.items();
    const auto GROUP_SIZE = 256;

    // Calculate width and height from samples
    samples = samples / 4;
    samples = (samples / GROUP_SIZE) ? (samples / GROUP_SIZE) * GROUP_SIZE :
                                     GROUP_SIZE;

    unsigned int tempVar1 = (unsigned int)sqrt((double)samples);
    tempVar1 = (tempVar1 / GROUP_SIZE) ? (tempVar1 / GROUP_SIZE) * GROUP_SIZE :
                                       GROUP_SIZE;
    samples = tempVar1 * tempVar1;

    int width = tempVar1;
    int height = width;


    const auto items_in = width * height;

    std::vector<float> random_values_a(items_in*4);//cl_float4
    generate_rand_real(random_values_a, 0.01f, 1.0f);


    /*__kernel void BS(
		const __global float* randArray,
		const int width,
		__global float* call,
		__global float* put)*/

    //allocate memory

    auto matrix_in = device.alloc<float>(random_values_a, true);//read_only
    if (!matrix_in)return COOPCL_BAD_ALLOC;

    //------------------------------------------ READ_WRITE
    auto matrix_out1 = device.alloc<float>(items_in * 4);
    if (!matrix_out1)return COOPCL_BAD_ALLOC;

    auto matrix_out2 = device.alloc<float>(items_in * 4);
    if (!matrix_out2)return COOPCL_BAD_ALLOC;



    const auto size_input_MB = (matrix_in->size()) * 1e-6;
    const auto size_out_MB = (matrix_out1->size() + matrix_out2->size()) * 1e-6;

    std::cout << "Input memory size:\t" << std::setw(3) << size_input_MB << " MB" << std::endl;
    std::cout << "Output memory size:\t" << std::setw(3) << size_out_MB << " MB" << std::endl;
    std::cout << "--------------------\n";

    std::cout << "BUILD KERNELS..." << std::endl;

    //Create tasks
    std::string jit_flags = " -cl-unsafe-math-optimizations -cl-fast-relaxed-math";

    auto task_BS = device.create_task(tasks_BS, "BS", jit_flags);
    if (!task_BS)throw std::runtime_error("Error JTI, FIMXE!!!");

    const int stride = width * 4;

    std::array<size_t, 3> global_sizes = { static_cast<size_t>(stride), static_cast<size_t>(height),1 };
    std::array<size_t, 3> local_sizes = { 1,1,1 };

    const auto ms_scale_factor = 1e-6f;
    int err = 0;

    const size_t iter = matrix_data.iterations() < 1 ? 1 : matrix_data.iterations();
    task_BS->clear_dependences();

    auto proc_id = 0;
    for (auto& off_proc : offloads)
    {
        float duration_ms_task_blur_avg = 0;

#ifdef _PAUSE_
        std::cout << "PAUSE 10 sec ..." << std::endl;
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(10000ms);
#endif
        std::cout << "Processing ..." << std::endl;
        for (size_t i = 0; i < iter; i++)
        {
            std::cout << "Iteration:\t[" << i << "/" << matrix_data.iterations() << "]\r" << std::flush;
            task_BS->wait_clear_events();


            err = device.execute_async(task_BS, off_proc, global_sizes, local_sizes,
                                       matrix_in, stride, matrix_out1, matrix_out2);
            if (err != CL_SUCCESS)return err;



            auto duration_ms_task_blur = task_BS->duration(ms_scale_factor);
            duration_ms_task_blur_avg += duration_ms_task_blur;
        }

        duration_ms_task_blur_avg = duration_ms_task_blur_avg / static_cast<float>(iter);

        std::string dev_name = "";
        auto devs = device.sub_devices();
        float sum_off = 0;

        for (auto& [ctx, dev] : devs)
        {
            for (auto& dev_off : off_proc)
            {
                const auto& [of, dt_did] = dev_off;
                const auto& [dt, did] = dt_did;
                if (dev.device_id() == did && dev.device_type() == dt)
                {
                    std::cout << "Offload:\t" << of << ", device:\t" << dev.name() << std::endl;

                    sum_off += of;
                    dev_name.append(dev.name());
                    if (1.0f - sum_off < 0.001f)break;
                    dev_name.append("+");
                }
            }
        }

        std::cout << "PROC [" << dev_name << "] " << "Task_BS:\t" << std::setw(4) << duration_ms_task_blur_avg << " ms" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
        proc_id++;
    }
    return err;

}

int skmd::Benchmark_MT(mkmd_input matrix_data)
{
    bool is_obj_061 = matrix_data.is_obj_061();

    const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
    const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
    const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
    const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);

    std::vector<offload_info> offloads;
    offload_info gpu_v0 = { {1.0f,{CL_DEVICE_TYPE_GPU,matrix_data.gpu_id()} } }; // single gpu

    if (is_obj_061)
    {
        //3 GPUS
        offload_info gpu_v1 = { {1.0f,cpu } };
        offload_info gpu_v2 = { {1.0f,gpu0 } };
        offload_info gpu_v3 = { {1.0f,gpu1 } };
        offload_info gpu_v4 = { {1.0f,gpu2 } };
        offloads = { gpu_v1,gpu_v2,gpu_v3,gpu_v4 };
    }
    else if(matrix_data.is_obj_129())
    {
        //2 GPUS				2048x2048 matrix 50 it.
        offload_info gpu_v01 = { {1.0f,cpu } }; // 1.4ms 1/x=0.714 //60W ->19.78 J
        offload_info gpu_v02 = { {1.0f,gpu0 } }; // 2.4ms 1/x=0.416 //30W ->0.387 J
        offload_info gpu_v03 = { {1.0f,gpu1 } }; // 1.4ms 1/x=0.714 //51W -> 7.52 J

        // sum_all = 1.844 (39,22,39)%
        // sum_gpu0_gpu1 = 1.428 (0,37,63)%
        // sum_cpu_gpu0 = 1.428 (63,37,0)%
        // sum_cpu_gpu1 = 1.13 (50,0,50)%

        offload_info gpu_v1 = { {0.39f,cpu },{0.22f,gpu0 },{0.39f,gpu1 } }; // 0.63 ms (35,30,45) 110W 69.3 J
        offload_info gpu_v2 = { {0.37f,gpu0 },{0.63f,gpu1 } }; // 0.89 ms (12,31,55)  98W -> 87.2 J
        offload_info gpu_v3 = { {0.63f,cpu },{0.37f,gpu0 } }; // 0.92 ms (39,31,12) 82W -> 75.4 J
        offload_info gpu_v4 = { {0.5f,cpu },{0.5f,gpu1 } }; // 0.73 ms (38,9,51) 98W -> 71.54 J

        offloads = { gpu_v01,gpu_v02,gpu_v03,
                    gpu_v1,gpu_v2,gpu_v3,gpu_v4 };
    }
    else if (matrix_data.is_obj_119())
    {
        //CPU+GPU				2048x2048 matrix 50 it.
        offload_info off_v01 = { {1.0f,cpu } };
        offload_info off_v02 = { {1.0f,gpu0 } };
        //offload_info gpu_v1 = { {0.1f,cpu },{0.9f,gpu0 } };
        offloads = { off_v01,off_v02};
    }

    std::string task_MAT_TRANSP = "";
    task_MAT_TRANSP.append(kernels::task_MT);


    auto& device = *matrix_data.device();

    std::cout << "ALLOCATE MEMORY ..." << std::endl;

    const auto items_in = matrix_data.items();
    std::vector<float> random_values_a(items_in);
    generate_rand_real(random_values_a, 0.01f, 1.0f);

    //allocate memory
    auto matrix_in = device.alloc<float>( random_values_a, true);//read_only
    if (!matrix_in)return COOPCL_BAD_ALLOC;


    //------------------------------------------ READ_WRITE

    auto matrix_out = device.alloc<float>(items_in);
    if (!matrix_out)return COOPCL_BAD_ALLOC;

    const auto size_input_MB = (matrix_in->size()) * 1e-6;
    const auto size_out_MB = (matrix_out->size()) * 1e-6;

    std::cout << "Input memory size:\t" << std::setw(3) << size_input_MB << " MB" << std::endl;
    std::cout << "Output memory size:\t" << std::setw(3) << size_out_MB << " MB" << std::endl;
    std::cout << "--------------------\n";

    std::cout << "BUILD KERNELS..." << std::endl;

    const auto BLOCK_DIM = 16;

    //Create tasks
    std::string jit_flags = " -cl-unsafe-math-optimizations -cl-fast-relaxed-math -DT=float -DBLOCK_DIM=";
    jit_flags.append(std::to_string(BLOCK_DIM));

    const std::string kname = "mat_transpose";

    auto task_MT = device.create_task(task_MAT_TRANSP, kname, jit_flags);
    if (!task_MT)throw std::runtime_error("Error JTI, FIMXE!!!");

    std::array<size_t, 3> global_sizes = { static_cast<size_t>(matrix_data.width()), static_cast<size_t>(matrix_data.height()),1 };
    std::array<size_t, 3> local_sizes = { BLOCK_DIM,BLOCK_DIM,1 };


    const auto ms_scale_factor = 1e-6f;
    int err = 0;

    const int mat_w = matrix_data.width();
    const int mat_h = matrix_data.height();
    const int offset = 0;
    const auto local_mem = (BLOCK_DIM+1) * BLOCK_DIM * sizeof(float);

    cl::LocalSpaceArg lmem = cl::Local(local_mem);

    const size_t iter = matrix_data.iterations() < 1 ? 1 : matrix_data.iterations();
    task_MT->clear_dependences();

    auto proc_id = 0;
    for (auto& off_proc : offloads)
    {
        float duration_ms_task = 0;

#ifdef _PAUSE_
        std::cout << "PAUSE 10 sec ..." << std::endl;
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(10000ms);
#endif
        std::cout << "Processing ..." << std::endl;
        for (size_t i = 0; i < iter; i++)
        {
            std::cout << "Iteration:\t[" << i << "/" << matrix_data.iterations() << "]\r" << std::flush;
            task_MT->wait_clear_events();

            //kernel void mat_transpose(global T* restrict odata, const global T* restrict  idata, const int offset, const int width, const int height, __local T* block)
            err = device.execute_async(task_MT, off_proc, global_sizes, local_sizes,
                                       matrix_out, matrix_in, offset, mat_w, mat_w, lmem);

            if (err != CL_SUCCESS)return err;

            const auto duration_ms = task_MT->duration(ms_scale_factor);
            duration_ms_task += duration_ms;
        }

        duration_ms_task = duration_ms_task / static_cast<float>(iter);

        std::string dev_name = "";
        auto devs = device.sub_devices();
        float sum_off = 0;

        for (auto& [ctx, dev] : devs)
        {
            for (auto& dev_off : off_proc)
            {
                const auto& [of, dt_did] = dev_off;
                const auto& [dt, did] = dt_did;
                if (dev.device_id() == did && dev.device_type() == dt)
                {
                    std::cout << "Offload:\t" << of << ", device:\t" << dev.name() << std::endl;

                    sum_off += of;
                    dev_name.append(dev.name());
                    if (1.0f - sum_off < 0.001f)break;
                    dev_name.append("+");
                }
            }
        }

        std::cout << "PROC [" << dev_name << "] " << "Task_MT:\t" << std::setw(4) << duration_ms_task << " ms" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
        proc_id++;
    }
    return err;

    //return Benchmark_platform(matrix_data, offloads, {}, 0, 0, 1);

}

int skmd::Benchmark_D2D(mkmd_input matrix_data)
{
    const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
    const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
    const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
    const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);

    std::vector<std::tuple<map_device_info, map_device_info, std::string>> copy_setups;

    if(matrix_data.device()->cnt_gpus()==1)
    {
        copy_setups = {
            { {cpu }, {gpu0 },"CPU->GPU_H2D"},
            { {gpu0 }, {cpu },"GPU->CPU_D2H"}
        };
        }
    else if(matrix_data.device()->cnt_gpus()==2)
    {
        copy_setups = {
            { {cpu }, {gpu0 },"CPU->GPU0"},
            { {gpu0 }, {cpu },"GPU0->CPU"},
            { {cpu }, {gpu1 },"CPU->GPU1"},
            { {gpu1 }, {cpu },"GPU1->CPU"},
            { {gpu0 }, {gpu1 },"GPU0->GPU1"},
            };
    }
    else if(matrix_data.device()->cnt_gpus()==3)
    {
        copy_setups = {
            { {cpu }, {gpu0 },"CPU->GPU0"},
            { {gpu0 }, {cpu },"GPU0->CPU"},

            { {cpu }, {gpu1 },"CPU->GPU1"},
            { {gpu1 }, {cpu },"GPU1->CPU"},

            { {cpu }, {gpu2 },"CPU->GPU2"},
            { {gpu2 }, {cpu },"GPU2->CPU"},

            { {gpu0 }, {gpu1 },"GPU0->GPU1"},
            { {gpu0 }, {gpu1 },"GPU0->GPU2"},
            { {gpu1 }, {gpu2 },"GPU1->GPU2"},
            };
    }
    const bool check_transfers = true;
    return Benchmark_platform(matrix_data, { }, copy_setups, check_transfers);

}

int skmd::Benchmark_VV(mkmd_input matrix_data)
{
    return Benchmark_MV(matrix_data,true);
}
