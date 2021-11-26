
#include "mkmd.h"
#include <iostream>
#include <functional>
#include <algorithm>

namespace proto {

	/*
class Node
{
	Node(const std::vector<std:::unique_ptr<clMemory>>& data_dependencies_in,
		const std::vector<std:::unique_ptr<clMemory >>& data_dependencies_outs,
		const std::vector<Node>& predecessors )
	{

	}
};
class graph
{
	graph() {};




};

auto create_graph() {
	return graph();
}


int main()
{
	std::string status;
	virtual_device device(status);
	const auto ndr = { 4096,4096,1 };
	const auto wgs = { 16,16,1 };
	const auto items = ndr[0]*ndr[1];

	auto matE = device.alloc<float>(items);
	auto matU = device.alloc<float>(items);
	auto matV = device.alloc<float>(items);
	auto matUE = device.alloc<float>(items);
	auto matVt = device.alloc<float>(items);
	auto matUEVt = device.alloc<float>(items);

	auto taskUE = device.create_task("", "");
	auto taskVT = device.create_task("", "");
	auto taskUEVT = device.create_task("", "");

	auto graph = create_graph();
	auto node0 = Node({}, { matU,matE,matV }, {});
	auto node1 = Node({ matU,matE},{ matUE }, {node0});
	auto node2 = Node({matV}, { matVt }, {node0});
	auto node3 = Node({ matUE,matVt }, { matUEVt }, { node1,node2 });
	auto node4 = Node({ matUEVt }, {}, { node3 });

	graph.add_nodes(node0, node1, node2, node3, node4);

	bool profile = true;
	if (profile) {
		node1.profile(taskUE, ndr, wgs, matU, matE, matUE);
		node2.profile(taskVT, ndr, wgs, matV, matVt);
		node3.profile(taskUEVT, ndr, wgs, matUE, matVt, matUEVt);
	}

	auto policy = graphcl::schedule::policy::PERF;
	auto schedule = graph.calculate_schedule(policy)
	auto err = graph.launch(schedule);
	return err;
}*/

	struct Task {
		Task(const std::string name) : _name(name) {}
		void call() const { std::cout << _name << "\n"; }
		std::string _name;
	};

	int test_variable_schedule_ordering()
	{
		const Task t1("MM1");
		const Task t2("MT");
		const Task t3("MM2");

		//task_id and callable function
		std::vector<std::pair<int, std::function<void()>>> schedule;

		//store a call to a member function and object
		std::function<void()> f_task_call1 = std::bind(&Task::call, t1);
		std::function<void()> f_task_call2 = std::bind(&Task::call, t2);
		std::function<void()> f_task_call3 = std::bind(&Task::call, t3);

		schedule.push_back({ 1,f_task_call1 });
		schedule.push_back({ 2,f_task_call2 });
		schedule.push_back({ 3,f_task_call3 });

		for (auto& t : schedule)
			t.second();

		std::cout << "Change invoke order ..." << std::endl;

		std::sort(schedule.begin(), schedule.end(), [](const auto a, const auto b)
			{
				return a.first > b.first;
			});

		for (auto& t : schedule)
			t.second();

		return 0;
	}

	namespace HEFT
	{
		/*sched_CLYAP_C{ 
		
		0: [
		ScheduleEvent(task = 0, start = 0, end = 0.0, proc = 0), 
		ScheduleEvent(task = 2, start = 0, end = 0.14, proc = 0), 
		ScheduleEvent(task = 1, start = 0.14, end = 51.74, proc = 0)] , 

		1: [ScheduleEvent(task = 3, start = 4.14, end = 55.64, proc = 1), 
			  ScheduleEvent(task = 4, start = 55.74, end = 56.0, proc = 1)] , 

		2: [] }
		*/

		struct node_matrix {
			int nodes{ 1 };
			node_matrix() = default;
			node_matrix(int item)
			{
				nodes = item;
			}
		};
		
		struct edge_matrix {
			int edges{ 1 };
			edge_matrix() = default;
			edge_matrix(int item)
			{
				edges = item;
			}
		};

		struct schedule_item {
			int start_time{ 0 };
			int end_time{ 0 };
			int proc_id{ 0 };
		};

		struct schedule {
			std::vector<int> list;
			std::vector<schedule_item> sch_list;

			schedule() = default;
			schedule(int item)
			{
				list.push_back(item);
			}
		};

		struct idle_processors {
			std::vector<int> list;
			idle_processors(int i)
			{
				list.push_back(i);
			}
		};
		
		struct offload_nodes {
			std::vector<int> list;
		};

		auto select_nodes_on_critical_path = [](const schedule& L)->offload_nodes
		{
			return offload_nodes();
		};

		auto find_idle_processors = [](const schedule& L)->idle_processors
		{
			return idle_processors(1);
		};

		auto calculate_HEFT = [](const node_matrix& V, const edge_matrix& E)->schedule
		{
			return schedule(V.nodes + E.edges);
		};

		auto insert_sub_kernel_nodes = [](node_matrix V, edge_matrix E, idle_processors IDL, offload_nodes CPN)->std::pair< node_matrix, edge_matrix>
		{
			auto v = node_matrix(2);
			auto e = edge_matrix(2);
			return { v, e };
		};

		schedule calculate_schedule_graphcl(const node_matrix& V, const edge_matrix& E, schedule& SCHL)
		{
			if (SCHL.list.empty())
			{
				SCHL = calculate_HEFT(V, E);
				return calculate_schedule_graphcl(V, E, SCHL);
			}
			else
			{
				auto CPNodes = select_nodes_on_critical_path(SCHL);
				auto IDL = find_idle_processors(SCHL);
				if (!IDL.list.empty())
				{
					auto [nV, nE] = insert_sub_kernel_nodes(V, E, IDL, CPNodes);
					if (nV.nodes != V.nodes && nE.edges != E.edges)
					{
						SCHL = calculate_HEFT(nV, nE);
						return calculate_schedule_graphcl(nV, nE, SCHL);
					}
					else
						return SCHL;
				}
			}
			return SCHL;
		};
	};

	void calculate_schedule_graphcl_test()
	{
		HEFT::node_matrix V;
		HEFT::edge_matrix E;
		HEFT::schedule SCHL;

		auto hsch = HEFT::calculate_schedule_graphcl(V, E, SCHL);
	}
};

int main(int argc, char** argv)
{
	//proto::test_variable_schedule_ordering();
	//proto::calculate_schedule_graphcl_test();

	kernels::init_kernels_mkmd();
	kernels::init_kernels_skmd();

	int  err = 0; 
	
	enum class app_ids { 
		APPS=0, BENCH_COMM=1, 
		ABE=2, GABE=3, MEQ=4, CLYAP=5, SVD=6, 
		MM=7, MT=8, PCIE=9, BL=10, BS=11, NB=12, MV=13, VV=14
	};

	//DEFAULT values
	size_t  matrix_width = 256;
	size_t  matrix_height = 256;
	std::uint8_t gpu_id = 0;
	auto schedule = SCHEDULE_DEFAULT;
	size_t  iterations = 10;
	app_ids app_id = app_ids::APPS;

	auto dump_help=[](std::ostream& ost)
	{
		ost << "Usage:\t" << " ./app matrix_width<int> matrix_height<int> gpu_id<int> schedule<int>(0,1,2) iterations<int> app_id<int>(0:6)\n";
		ost << "Example:\t" << " ./app 2048 2048 0 0 15 0\n" ;
	};

	if (argc == 1)
	{
		//USE DEFAULT
	}
	else if (argc == 2)
	{
		dump_help(std::cout);
		return 0;
	}
	else if (argc == 7)
	{
		//Parse
		matrix_width = std::atoi(argv[1]);
		matrix_height = std::atoi(argv[2]);
		gpu_id = std::atoi(argv[3]);
		schedule = std::atoi(argv[4]);
		iterations = std::atoi(argv[5]);	
		app_id = static_cast<app_ids>( std::atoi(argv[6]) );
	}
	else
	{
		dump_help(std::cerr);
		return -1;
	}

	std::cout << "Execute:\t" << argv[0] << " "		
		<< matrix_width <<" "
		<< matrix_height <<" "
		<< (int)gpu_id <<" "
		<< schedule <<" "		
		<< iterations <<" "
		<<(int)app_id << std::endl;

	std::cout << "-------------------------------------------" << std::endl;

	std::string dev_ok;
	virtual_device device(dev_ok);
	if (!dev_ok.empty())return COOPCL_BAD_DEVICE_INIT;

	const auto matrix_sizes = mkmd_input(device, matrix_width, matrix_height, gpu_id, schedule,iterations);	

	switch (app_id)
	{
		case app_ids::APPS: 
		{
			std::cout << "##################################" << std::endl;
			std::cout << "EXECUTE all MKMD apps" << std::endl;
			std::cout << "##################################" << std::endl;
			

			/// Algebraic_Bernoulli_ABE
			err = mkmd::Algebraic_Bernoulli_ABE(matrix_sizes);
			if (err != 0)return err;

			/// Generalized_Algebraic_Bernoulli_GABE
			err = mkmd::Generalized_Algebraic_Bernoulli_GABE(matrix_sizes);
			if (err != 0)return err;

			/// MEQ
			err = mkmd::Matrix_equation(matrix_sizes);
			if (err != 0)return err;

			/// Lyapunov
			err = mkmd::Continuous_Lyapunov(matrix_sizes);
			if (err != 0)return err;

			/// SVD
			err = mkmd::SVD(matrix_sizes);
			if (err != 0)return err;

		}
		break;

		case app_ids::BENCH_COMM:
		{
			std::cout << "##################################" << std::endl;
			std::cout << "EXECUTE MKMD benchmark communication" << std::endl;
			std::cout << "##################################" << std::endl;
			
			if (matrix_sizes.is_obj_061()) {
				err = skmd::Benchmark_obj061(matrix_sizes);
				if (err != 0)return err;
			}
			else if (matrix_sizes.is_obj_129())
			{
				err = skmd::Benchmark_obj129(matrix_sizes);
				if (err != 0)return err;
			}
			else if (matrix_sizes.is_obj_119())
			{
				err = skmd::Benchmark_obj119(matrix_sizes);
				if (err != 0)return err;
			}
		}
		break;

		case app_ids::ABE:
		{	
			std::cout << "##################################" << std::endl;
			std::cout << "EXECUTE Algebraic_Bernoulli_ABE" << std::endl;
			std::cout << "##################################" << std::endl;
			err = mkmd::Algebraic_Bernoulli_ABE(matrix_sizes);
			if (err != 0)return err;
		}
		break;

		case app_ids::GABE:
		{			
			std::cout << "##################################" << std::endl;
			std::cout << "EXECUTE Generalized_Algebraic_Bernoulli_GABE" << std::endl;
			std::cout << "##################################" << std::endl;
			err = mkmd::Generalized_Algebraic_Bernoulli_GABE(matrix_sizes);
			if (err != 0)return err;
		}
		break;

		case app_ids::CLYAP:
		{		
			std::cout << "##################################" << std::endl;
			std::cout << "EXECUTE Continuous_Lyapunov" << std::endl;
			std::cout << "##################################" << std::endl;
			err = mkmd::Continuous_Lyapunov(matrix_sizes);
			if (err != 0)return err;
		}
		break;

		case app_ids::MEQ:
		{
			std::cout << "##################################" << std::endl;
			std::cout << "EXECUTE Matrix_equation" << std::endl;
			std::cout << "##################################" << std::endl;
			err = mkmd::Matrix_equation(matrix_sizes);
			if (err != 0)return err;
		}
		break;

		case app_ids::SVD:
		{
			std::cout << "##################################" << std::endl;
			std::cout << "EXECUTE SVD" << std::endl;
			std::cout << "##################################" << std::endl;
			err = mkmd::SVD(matrix_sizes);
			if (err != 0)return err;
		}
		break;

		case app_ids::MM:
		{
			std::cout << "##################################" << std::endl;
			std::cout << "EXECUTE MAT_MULTIPLY" << std::endl;
			std::cout << "##################################" << std::endl;
			err = skmd::Benchmark_MM(matrix_sizes);
			if (err != 0)return err;
		}
		break;

		case app_ids::MT:
		{
			std::cout << "##################################" << std::endl;
			std::cout << "EXECUTE MAT_TRANSPOSE" << std::endl;
			std::cout << "##################################" << std::endl;
			err = skmd::Benchmark_MT(matrix_sizes);
			if (err != 0)return err;
		}
		break;

		case app_ids::PCIE:
		{
			std::cout << "##################################" << std::endl;
			std::cout << "EXECUTE PCI-EXP COMM. D2D" << std::endl;
			std::cout << "##################################" << std::endl;
			err = skmd::Benchmark_D2D(matrix_sizes);
			if (err != 0)return err;
		}
		break;
		
		case app_ids::BL:
		{
			std::cout << "##################################" << std::endl;
			std::cout << "EXECUTE IMG_BLUR" << std::endl;
			std::cout << "##################################" << std::endl;
			err = skmd::Benchmark_BL(matrix_sizes);
			if (err != 0)return err;
		}
		break;

		case app_ids::BS:
		{
			std::cout << "##################################" << std::endl;
			std::cout << "EXECUTE BS" << std::endl;
			std::cout << "##################################" << std::endl;
			err = skmd::Benchmark_BS(matrix_sizes);
			if (err != 0)return err;
		}
		break;

		case app_ids::NB:
		{
			std::cout << "##################################" << std::endl;
			std::cout << "EXECUTE NB" << std::endl;
			std::cout << "##################################" << std::endl;
			err = skmd::Benchmark_NB(matrix_sizes);
			if (err != 0)return err;
		}
		break;

		case app_ids::MV:
		{
			std::cout << "##################################" << std::endl;
			std::cout << "EXECUTE MV" << std::endl;
			std::cout << "##################################" << std::endl;
			err = skmd::Benchmark_MV(matrix_sizes);
			if (err != 0)return err;
		}
		break;

		case app_ids::VV:
		{
			std::cout << "##################################" << std::endl;
			std::cout << "EXECUTE VV" << std::endl;
			std::cout << "##################################" << std::endl;
			err = skmd::Benchmark_VV(matrix_sizes);
			if (err != 0)return err;
		}
		break;

	default:
		break;
	}    
	
	if(err!=0) std::cerr << "MKMD, FAILED ! FIXME !!!" << std::endl;

	return err;
}
