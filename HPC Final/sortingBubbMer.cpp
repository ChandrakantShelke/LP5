#include <iostream>
#include <vector>
#include <cstdlib> // For rand() and srand()
#include <ctime>
#include <omp.h>

using namespace std;

#define NUM_THREADS 4  // Number of threads for parallel execution
#define THRESHOLD 100  // Threshold to switch from parallel to sequential

//Function to Generate Random Array
void generateRandomArray(vector<int> &arr, int size) {
    srand(time(0));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 1000;
    }
}

void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

void bubbleSortSequential(vector<int> &arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void bubbleSortParallel(vector<int> &arr) {
    int n = arr.size();
    //Parallelizes the outer loop with OpenMP. Each iteration (pass) is done in a separate thread
    #pragma omp parallel for
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

void merge(vector<int> &arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    vector<int> L(n1), R(n2);
    
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

   //Merges the temp vectors back into the original array in sorted order.
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }
    //Adds any remaining elements from the temporary arrays
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void mergeSortSequential(vector<int> &arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortSequential(arr, left, mid);
        mergeSortSequential(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void mergeSortParallel(vector<int> &arr, int left, int right) {
    if (right - left <= THRESHOLD) {
        mergeSortSequential(arr, left, right);  // Switch to sequential when small enough
    } else {
        if (left < right) {
            int mid = left + (right - left) / 2;

            #pragma omp parallel sections
            {
                #pragma omp section
                mergeSortParallel(arr, left, mid);

                #pragma omp section
                mergeSortParallel(arr, mid + 1, right);
            }

            merge(arr, left, mid, right);
        }
    }
}

int main() {
    int size = 0; // Array size

    cout << "Enter number of elements in array: ";
    cin >> size;
    
    vector<int> arr(size), arr_copy;

    // Generate random array
    generateRandomArray(arr, size);
    arr_copy = arr;  // Copy for comparison

    // Bubble Sort - Sequential Execution
    double bubbleSeqStart = omp_get_wtime();
    bubbleSortSequential(arr);
    double bubbleSeqEnd = omp_get_wtime();
    double bubbleSeqTime = bubbleSeqEnd - bubbleSeqStart;
    cout << "Bubble Sort Sequential Execution Time: " << bubbleSeqTime << " sec\n";

    // Bubble Sort - Parallel Execution
    double bubbleParStart = omp_get_wtime();
    bubbleSortParallel(arr);
    double bubbleParEnd = omp_get_wtime();
    double bubbleParTime = bubbleParEnd - bubbleParStart;
    cout << "Bubble Sort Parallel Execution Time: " << bubbleParTime << " sec\n";
    cout << "Bubble Sort Speedup: " << bubbleSeqTime / bubbleParTime << "\n\n";

    // Merge Sort - Sequential Execution
    arr = arr_copy; // Reset array
    double mergeSeqStart = omp_get_wtime();
    mergeSortSequential(arr, 0, size - 1);
    double mergeSeqEnd = omp_get_wtime();
    double mergeSeqTime = mergeSeqEnd - mergeSeqStart;
    cout << "Merge Sort Sequential Execution Time: " << mergeSeqTime << " sec\n";

    // Merge Sort - Parallel Execution
    arr = arr_copy; // Reset array
    double mergeParStart = omp_get_wtime();
    mergeSortParallel(arr, 0, size - 1);
    double mergeParEnd = omp_get_wtime();
    double mergeParTime = mergeParEnd - mergeParStart;
    cout << "Merge Sort Parallel Execution Time: " << mergeParTime << " sec\n";
    cout << "Merge Sort Speedup: " << mergeSeqTime / mergeParTime << "\n";

    return 0;
}
//g++ -fopenmp filename.cpp -o filename
//./a.out
