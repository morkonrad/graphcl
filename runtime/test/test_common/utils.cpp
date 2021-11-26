
#include "utils.h"

#include <sstream>
#include <fstream>
#include <iostream>

#include <algorithm>
#include <random>
#include <execution>

namespace utils
{
	template<typename T>
	static void generate_rand_real(std::vector<T> & container, const T start, const T end)
	{
		std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<> dis(static_cast<T>(start), static_cast<T>(end));

		std::for_each(std::execution::par_unseq, container.begin(), container.end(), [&gen, &dis](T& val) {
			val = static_cast<float>(dis(gen));
			});
	}
	
	template<typename T>
	static void generate_rand_int(std::vector<T> & container, const T start, const T end)
	{
		std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()

		std::uniform_int_distribution<> dis(static_cast<int>(start), static_cast<int>(end));
		std::for_each(std::execution::par_unseq, container.begin(), container.end(), [&gen, &dis](T& val) {
			val = static_cast<T>(dis(gen));
			});
	}
	
	void generate_rand(std::vector<float>& container, const float start, const float end)
	{
		generate_rand_real(container, start, end);
	}

	void generate_rand(std::vector<double>& container, const double start, const double end)
	{
		generate_rand_real(container, start, end);
	}

	void generate_rand(std::vector<int8_t>& container, const int8_t start, const int8_t end)
	{
		generate_rand_int(container, start, end);
	}

	void generate_rand(std::vector<int16_t>& container, const int16_t start, const int16_t end)
	{
		generate_rand_int(container, start, end);
	}

	void generate_rand(std::vector<int32_t>& container, const int32_t start, const int32_t end)
	{
		generate_rand_int(container, start, end);
	}

	void generate_rand(std::vector<int64_t>& container, const int64_t start, const int64_t end)
	{
		generate_rand_int(container, start, end);
	}

	void generate_rand(std::vector<uint8_t>& container, const uint8_t start, const uint8_t end)
	{
		generate_rand_int(container, start, end);
	}

	void generate_rand(std::vector<uint16_t>& container, const uint16_t start, const uint16_t end)
	{
		generate_rand_int(container, start, end);
	}

	void generate_rand(std::vector<uint32_t>& container, const uint32_t start, const uint32_t end)
	{
		generate_rand_int(container, start, end);
	}

	void generate_rand(std::vector<uint64_t>& container, const uint64_t start, const uint64_t end)
	{
		generate_rand_int(container, start, end);
	}

	float generate_rand(const float start, const float end)
	{
		std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<> dis(static_cast<float>(start), static_cast<float>(end));
		return static_cast<float>(dis(gen));
	}

	int on_coopcl_error(const std::string err_txt)
	{
		if (!err_txt.empty())
		{
			std::cerr << err_txt << std::endl;
			std::exit(-100);
		}
		return 0;
	}

	int compare_values(const std::vector<int> & src)
	{
		std::cout << "Compare results ..." << std::endl;
		int id = 0;
		for (auto& v : src)
		{
			if (v != id)
			{
				std::cerr << "Some error at position: " << id - 1 << ": " << v << "!=" << id << " fixme!!!" << std::endl;
				return -2;
			}

			id++;
		}
		std::cout << "Compare results <OK>" << std::endl;
		std::cout << "--------------------" << std::endl;
		return 0;
	}

	int compare_values(const std::vector<int> & src, const int expect_val)
	{
		std::cout << "Compare results ..." << std::endl;
		int id = 1;
		for (auto& v : src)
		{
			if (v != expect_val)
			{
				std::cerr << "Some error at position: " << id - 1 << ": " << v << "!=" << expect_val << " fixme!!!" << std::endl;
				return -2;
			}
			id++;
		}
		std::cout << "Compare results <OK>" << std::endl;
		std::cout << "--------------------" << std::endl;
		return 0;
	}

	
	std::tuple <std::vector<size_t>, std::vector<float>, size_t, bool, std::string> 
		parse_test_args(const int argc, char** argv)
	{
		const auto path_conf_file = argv[1];
		std::cout << "Try parse input values from:\t" <<  path_conf_file << std::endl;

		std::vector<float> data_items;
		std::vector<float> data_split_ratios;
		size_t iterations = 0;
		bool check_values = false;
		std::string path_csv_file = "";

		std::ifstream ifs(path_conf_file);
		if (!ifs.is_open())
		{
			std::cerr << "Couldn't open:\t" << path_conf_file << " ... FIXME, EXIT !!" << std::endl;
			return{};
		}
		else
		{
			/*
			* Assumed simple file format
			* 1) line is csv_line with sizes 
			* 2) line is also csv_line with float data_split_ratios
			* 3) line has only one value with count iterations
			* 4) line has only one value with 0 or 1 flag that means check results or not
			* 5) line includes path to write the file with benchmark profiles
			* 
			* Example below: 
			512,1024,2048
			0.0f,0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,1.0f
			10
			1
			/tmp/bench_name.csv
			*/

			std::stringstream ss_config;
			ss_config << ifs.rdbuf();

			if (ss_config.str().empty())
			{
				std::cerr << "File: " << path_conf_file << "is empty ,fixme!" << std::endl;
				return {};
			}


			auto read_sign_seprated_line =[](
				std::vector<float>& parsed_values,
				const std::string & line,
				const char& separator)->int
			{
				if (line.empty())return -1;
				
				std::string token;
				std::istringstream iss(line);
				
				while (getline(iss, token, separator))
				{
					auto val = std::atof(token.c_str());
					parsed_values.push_back(val);					
				}
				return 0;
			};
			
			const char separator = ',';
			std::string line;
			std::istringstream ss(ss_config.str());

			size_t id = 0;
			while (getline(ss, line))
			{				
				switch (id++)
				{
					case 0:
					{
						const auto read_ok = read_sign_seprated_line(data_items, line, separator);
						if (read_ok != 0)return {};
					}break;
					case 1:
					{
						const auto read_ok = read_sign_seprated_line(data_split_ratios, line, separator);
						if (read_ok != 0)return {};
					}break;
					case 2:
					{
						iterations = std::atoi(line.c_str());
					}break;
					case 3:
					{
						check_values = std::atoi(line.c_str());
					}break;
					case 4:
					{
						path_csv_file = line;
					}break;
				}
			}
		}
		
		std::vector<size_t> data_items_int;
		data_items_int.reserve(data_items.size());
		for (auto v : data_items)
			data_items_int.emplace_back(static_cast<size_t>(v));

		return{ data_items_int,data_split_ratios,iterations,check_values,path_csv_file };
	}

	std::tuple <std::vector<size_t>, std::vector<float>, size_t, bool, std::string> 
		get_default_test_setup()
	{
		std::cout << "Use default values to start unit-test ..." << std::endl;

		const std::vector<size_t> data_items{ 512,1024,2048 };
		const std::vector<float> offloads = { 0.05f, 0.10f, 0.25f , 0.3f , 0.5f, 0.6f, 0.75f };
        const size_t iterations = 1;
        const bool check_results = true;
		return { data_items,offloads,iterations,check_results,"" };
	}

	/**
	 * @brief Compare values and sizes of containers: in and inRef
	 * @param in container
	 * @param inRef container
	 * @param delta threshold
	 * @return -1 if sizes unequal or if abs_diff(values)>=delta
	*/
	template<typename T>
	static int compare_values_real(const std::vector<T> & in, const std::vector<T> & inRef, const T delta)
	{
		std::cout << "--------------------" << std::endl;
		std::cout << "Compare results ..." << std::endl;

		if (in.size() != inRef.size()) {
			std::cerr << "Unequal size container sizes ... FIXME!" << std::endl;
			std::cout << "--------------------" << std::endl;
			return -1;
		}

		const auto& mispair = std::mismatch(std::execution::par_unseq, in.begin(), in.end(), inRef.begin(), inRef.end(), [&delta](const T a, const T b)->bool
			{
				const auto diff = std::fabs(a - b);
				if (diff >= delta)return false;
				return true;
			});
		if (mispair.first != in.end() || mispair.second != inRef.end())
		{
			size_t id = 0;
			for (auto b = in.begin(); b != in.end(); b++, id++)
			{
				if (b == mispair.first)
					break;
			}

			std::cerr << "Some wrong values: " << *mispair.first << " != " << *mispair.second << " at[" << id << "] ... FIXME!" << std::endl;
			std::cout << "--------------------" << std::endl;
			return -1;
		}

		std::cout << "Compare results <OK>" << std::endl;
		std::cout << "--------------------" << std::endl;
		return 0;
	}


	int compare_values(const std::vector<double>& in, const std::vector<double>& inRef, const double delta /*= 1e-3f*/)
	{
		return compare_values_real(in, inRef, delta);
	}

	int compare_values(const std::vector<float>& in, const std::vector<float>& inRef, const float delta /*= 1e-3f*/)
	{
		return compare_values_real(in, inRef, delta);
	}
	/**
	 * @brief Compare values and sizes of containers: in and inRef
	 * @param in container
	 * @param inRef container
	 * @return -1 if sizes unequal or if any value in in[:]!=inRef[:]
	*/
	template<typename T>
	static int gen_compare_values_int(const std::vector<T>& in, const std::vector<T>& inRef)
	{
		std::cout << "--------------------" << std::endl;
		std::cout << "Compare results ..." << std::endl;

		if (in.size() != inRef.size()) {
			std::cerr << "Unequal size container sizes ... FIXME!" << std::endl;
			std::cout << "--------------------" << std::endl;
			return -1;
		}

		const auto& mispair = std::mismatch(std::execution::par_unseq, in.begin(), in.end(), inRef.begin(), inRef.end(), [](const auto a, const auto b)->bool
			{
				if (a == b)return true;
				return false;
			});

		if (mispair.first != in.end() || mispair.second != inRef.end())
		{
			size_t id = 0;
			for (auto b = in.begin(); b != in.end(); b++, id++)
			{
				if (b == mispair.first)
					break;
			}

			std::cerr << "Some wrong values: " << (int)*mispair.first << " != " << (int)*mispair.second << " at[" << id << "] ... FIXME!" << std::endl;
			std::cout << "--------------------" << std::endl;
			return -1;
		}

		std::cout << "Compare results <OK>" << std::endl;
		std::cout << "--------------------" << std::endl;
		return 0;
	}

	int compare_values(const std::vector<std::int8_t>& in, const std::vector<std::int8_t>& inRef)
	{
		return gen_compare_values_int(in, inRef);
	}

	int compare_values(const std::vector<std::int16_t>& in, const std::vector<std::int16_t>& inRef)
	{
		return gen_compare_values_int(in, inRef);
	}

	int compare_values(const std::vector<std::int32_t>& in, const std::vector<std::int32_t>& inRef)
	{
		return gen_compare_values_int(in, inRef);
	}

	int compare_values(const std::vector<std::int64_t>& in, const std::vector<std::int64_t>& inRef)
	{
		return gen_compare_values_int(in, inRef);
	}

	int compare_values(const std::vector<std::uint8_t>& in, const std::vector<std::uint8_t>& inRef)
	{
		return gen_compare_values_int(in, inRef);
	}

	int compare_values(const std::vector<std::uint16_t>& in, const std::vector<std::uint16_t>& inRef)
	{
		return gen_compare_values_int(in, inRef);
	}

	int compare_values(const std::vector<std::uint32_t>& in, const std::vector<std::uint32_t>& inRef)
	{
		return gen_compare_values_int(in, inRef);
	}

	int compare_values(const std::vector<std::uint64_t>& in, const std::vector<std::uint64_t>& inRef)
	{
		return gen_compare_values_int(in, inRef);
	}



}
