#include "math_utils_test.hpp"
TEST(MathUtilsTest, TestChoice) {
    auto& re = RandomEngine<float, 0>::getInstance();
    const int size = 10;
    auto probs = re.uniform(size);
    for ( int i=0; i < size; ++i ) { probs(i) =1e-2; }
    //probs = probs / probs.sum();

    Eigen::ArrayXf cumsum(size);
    Eigen::ArrayXi indices(size);
    choice(re, probs, cumsum, indices);
    std::cout << probs << std::endl;
    std::cout << "____" << std::endl; 
    std::cout << indices << std::endl;
}

TEST(MathUtilsTest, TestMinMaxNorm) {
	auto tensor = torch::ones({5,5});
	auto a = tensor.accessor<float,2>();
	tensor[1][2] = 3;
	tensor[4][4] = -6;
	std::cout << tensor << std::endl;
	min_max_norm(tensor.index({1}), -1.f, false);
	std::cout << tensor << std::endl;
}

TEST(MathUtilsTest, TestForward) {
	int actions = 4;
	int atoms = 3;
	auto input = torch::rand({2,4,3});
	
	std::cout << "original input:" << std::endl;
	std::cout << input << std::endl;

	auto mean = input.mean(1, true);
	std::cout << "mean: " << std::endl;
	std::cout << mean << std::endl;

	input = input - mean;
	std::cout << "input minus mean:" << std::endl;
	std::cout << input << std::endl;

	auto output = torch::nn::functional::softmax(input.view({-1, 3}), torch::nn::functional::SoftmaxFuncOptions(1));

	std::cout << "softmax" << std::endl;	
	std::cout << output << std::endl;
	
	std::cout << "forward output" << std::endl;
	output = output.view({-1, actions, atoms});
	std::cout << output << std::endl;

	
	auto support = torch::linspace(-10,10,3);
	auto next_dist = output * support;


	std::cout << "output * support" << std::endl;
	std::cout << next_dist << std::endl;

	auto next_action = std::get<1>(next_dist.sum(2).max(1));
	std::cout << "next action:" << std::endl;
	std::cout << next_action << std::endl;


	next_action = next_action.unsqueeze(1).unsqueeze(1).expand({next_dist.size(0), 1, next_dist.size(2)});
	std::cout << "next action transformed:" << std::endl;
	std::cout << next_action << std::endl;

	next_dist = next_dist.gather(1, next_action).squeeze(1);
	std::cout << "next dist" << std::endl;
	std::cout << next_dist << std::endl;

}
