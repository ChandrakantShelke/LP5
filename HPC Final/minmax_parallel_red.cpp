#include <iostream>
#include <omp.h>
#include <chrono> //for measuring execution time.
#include <limits> //for getting min/max values (like INT_MAX).
#include <vector>
#include <numeric>

using namespace std;
using namespace std::chrono;

int sequentialMax(const vector<int> &data)
{
	//If array is empty, return the minimum possible int.
    if (data.empty())
        return numeric_limits<int>::min();
    int max_val = data[0];
    for (int value : data)
        if (value > max_val)
            max_val = value;
    return max_val;
}

int sequentialMin(const vector<int> &data)
{
    if (data.empty())
        return numeric_limits<int>::max();
    int min_val = data[0];
    for (int value : data)
        if (value < min_val)
            min_val = value;
    return min_val;
}

int sequentialSum(const vector<int> &data)
{
    int sum = 0;
    for (int value : data)
        sum += value;
    return sum;
}

double sequentialAverage(const vector<int> &data)
{
    if (data.empty())
        return 0.0;
    return static_cast<double>(sequentialSum(data)) / data.size();
}

int parallelMin(const vector<int> &data)
{
    if (data.empty())
        return numeric_limits<int>::max();
    int min_val = data[0];
    #pragma omp parallel for reduction(min : min_val)
    //Each thread finds its local minimum, OpenMP reduces them to global minimum
    for (size_t i = 1; i < data.size(); ++i)
        if (data[i] < min_val)
            min_val = data[i];
    return min_val;
}

int parallelMax(const vector<int> &data)
{
    if (data.empty())
        return numeric_limits<int>::min();
    int max_val = data[0];
    #pragma omp parallel for reduction(max : max_val)
    for (size_t i = 1; i < data.size(); ++i)
        if (data[i] > max_val)
            max_val = data[i];
    return max_val;
}

int parallelSum(const vector<int> &data)
{
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < data.size(); ++i)
        sum += data[i];
    return sum;
}

double parallelAverage(const vector<int> &data)
{
    if (data.empty())
        return 0.0;
    return static_cast<double>(parallelSum(data)) / data.size();
}

int main()
{
    vector<int> data = {1, 4, 8, 6, 9, 5, 3, 11, 22, 33, 44, 55, 66};
    for (int i = 0; i < 100000; i++)
        data.push_back(rand() % 1000);

    auto start_seq_min = high_resolution_clock::now();
    int seq_min = sequentialMin(data);
    auto stop_seq_min = high_resolution_clock::now();
    auto duration_seq_min = duration_cast<microseconds>(stop_seq_min - start_seq_min);

    auto start_seq_max = high_resolution_clock::now();
    int seq_max = sequentialMax(data);
    auto stop_seq_max = high_resolution_clock::now();
    auto duration_seq_max = duration_cast<microseconds>(stop_seq_max - start_seq_max);

    auto start_seq_sum = high_resolution_clock::now();
    int seq_sum = sequentialSum(data);
    auto stop_seq_sum = high_resolution_clock::now();
    auto duration_seq_sum = duration_cast<microseconds>(stop_seq_sum - start_seq_sum);

    auto start_seq_avg = high_resolution_clock::now();
    double seq_avg = sequentialAverage(data);
    auto stop_seq_avg = high_resolution_clock::now();
    auto duration_seq_avg = duration_cast<microseconds>(stop_seq_avg - start_seq_avg);

    auto start_par_min = high_resolution_clock::now();
    int par_min = parallelMin(data);
    auto stop_par_min = high_resolution_clock::now();
    auto duration_par_min = duration_cast<microseconds>(stop_par_min - start_par_min);

    auto start_par_sum = high_resolution_clock::now();
    int par_sum = parallelSum(data);
    auto stop_par_sum = high_resolution_clock::now();
    auto duration_par_sum = duration_cast<microseconds>(stop_par_sum - start_par_sum);

    auto start_par_max = high_resolution_clock::now();
    int par_max = parallelMax(data);
    auto stop_par_max = high_resolution_clock::now();
    auto duration_par_max = duration_cast<microseconds>(stop_par_max - start_par_max);

    auto start_par_avg = high_resolution_clock::now();
    double par_avg = parallelAverage(data);
    auto stop_par_avg = high_resolution_clock::now();
    auto duration_par_avg = duration_cast<microseconds>(stop_par_avg - start_par_avg);

    cout << "\nMinimum:\n";
    cout << "Sequential Time: " << duration_seq_min.count() << " µs\n";
    cout << "Parallel Time: " << duration_par_min.count() << " µs\n";
    cout << "Speedup Factor: " << static_cast<double>(duration_seq_min.count()) / duration_par_min.count() << "\n";

    cout << "\nMaximum:\n";
    cout << "Sequential Time: " << duration_seq_max.count() << " µs\n";
    cout << "Parallel Time: " << duration_par_max.count() << " µs\n";
    cout << "Speedup Factor: " << static_cast<double>(duration_seq_max.count()) / duration_par_max.count() << "\n";

    cout << "\nSum:\n";
    cout << "Sequential Time: " << duration_seq_sum.count() << " µs\n";
    cout << "Parallel Time: " << duration_par_sum.count() << " µs\n";
    cout << "Speedup Factor: " << static_cast<double>(duration_seq_sum.count()) / duration_par_sum.count() << "\n";

    cout << "\nAverage:\n";
    cout << "Sequential Time: " << duration_seq_avg.count() << " µs\n";
    cout << "Parallel Time: " << duration_par_avg.count() << " µs\n";
    cout << "Speedup Factor: " << static_cast<double>(duration_seq_avg.count()) / duration_par_avg.count() << "\n";

    return 0;
}
////g++ -fopenmp filename.cpp -o filename
//./a.out