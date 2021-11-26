#pragma once

#include <vector>
#include <string>
#include <fstream>

enum class verbose_t { write_to_cout = -1, write_to_file = 1};

namespace utils
{
	float generate_rand(const float start, const float end);

    void generate_rand(std::vector<float>& container, const float start, const float end);
    void generate_rand(std::vector<double>& container, const double start, const double end);

	void generate_rand(std::vector<int8_t>& container, const int8_t start, const int8_t end);
	void generate_rand(std::vector<int16_t>& container, const int16_t start, const int16_t end);
	void generate_rand(std::vector<int32_t>& container, const int32_t start, const int32_t end);
	void generate_rand(std::vector<int64_t>& container, const int64_t start, const int64_t end);
		
	void generate_rand(std::vector<uint8_t>& container, const uint8_t start, const uint8_t end);
	void generate_rand(std::vector<uint16_t>& container, const uint16_t start, const uint16_t end);
	void generate_rand(std::vector<uint32_t>& container, const uint32_t start, const uint32_t end);
	void generate_rand(std::vector<uint64_t>& container, const uint64_t start, const uint64_t end);


	int compare_values(const std::vector<int>& src);

	int compare_values(const std::vector<int>& src, const int expect_val);

	int on_coopcl_error(const std::string err_txt);

	/**
	 * @brief Compare values and sizes of containers: in and inRef
	 * @param in container
	 * @param inRef container
	 * @param delta threshold
	 * @return -1 if sizes unequal or if abs_diff(values)>=delta
	*/
	int compare_values(const std::vector<float>& in, const std::vector<float>& inRef, const float delta = 1e-3f);
	int compare_values(const std::vector<double>& in, const std::vector<double>& inRef, const double delta = 1e-3f);

	/**
	 * @brief Compare values and sizes of containers: in and inRef
	 * @param in container
	 * @param inRef container
	 * @return -1 if sizes unequal or if any value in in[:]!=inRef[:]
	*/
	int compare_values(const std::vector<std::int8_t>& in, const std::vector<std::int8_t>& inRef);
	int compare_values(const std::vector<std::int16_t>& in, const std::vector<std::int16_t>& inRef);
	int compare_values(const std::vector<std::int32_t>& in, const std::vector<std::int32_t>& inRef);
	int compare_values(const std::vector<std::int64_t>& in, const std::vector<std::int64_t>& inRef);
	
	int compare_values(const std::vector<std::uint8_t>& in, const std::vector<std::uint8_t>& inRef);
	int compare_values(const std::vector<std::uint16_t>& in, const std::vector<std::uint16_t>& inRef);
	int compare_values(const std::vector<std::uint32_t>& in, const std::vector<std::uint32_t>& inRef);
	int compare_values(const std::vector<std::uint64_t>& in, const std::vector<std::uint64_t>& inRef);


	std::tuple <std::vector<size_t>, std::vector<float>, size_t, bool, std::string>
		parse_test_args(const int argc,char** argv);
	
	std::tuple <std::vector<size_t>, std::vector<float>, size_t, bool, std::string>
		get_default_test_setup();


}
