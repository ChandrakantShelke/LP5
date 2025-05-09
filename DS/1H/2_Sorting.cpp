#include<iostream>
#include<vector>
#include<omp.h>
#include<ctime>

using namespace std;
#define NUM_THREADS 4  // Number of threads for parallel execution


void generateArray(vector<int> &arr, int size){
    srand(time(0));
    for(int i=0; i<size; i++){
        arr[i] = rand() % 1000;
    }
}

void seqBubbleSort(vector<int> &arr){
    int n = arr.size();
    
    for(int i=0; i<n-1; i++){
        for(int j=0; j<n-i-1; j++){
            if(arr[j] > arr[j+1]){
                swap(arr[j], arr[j+1]);
            }
        }
    }
}

void parBubbleSort(vector<int> &arr){
    int n = arr.size();
    bool sorted = false;

    while(! sorted){
        sorted = true;

        #pragma omp parallel for shared(arr, sorted) //even
        for(int i=0; i<n-1; i+=2){
            if(arr[i] > arr[i+1]){
                swap(arr[i], arr[i+1]);
                sorted = false;
            }
        }

        #pragma omp parallel for shared(arr, sorted)
        for(int i=1; i<n-1; i+=2){
            if(arr[i] > arr[i+1]){
                swap(arr[i], arr[i+1]);
                sorted = false;
            }
        }
    }
}

void merge(vector<int> &arr, int st, int mid, int en){
    vector<int> temp;
    int i = st, j = mid+1;
    
    while(i<=mid && j<=en){
        if(arr[i] <= arr[j]){
            temp.push_back(arr[i]);
            i++;
        }
        else{
            temp.push_back(arr[j]);
            j++;
        }
    }
    
    while(i<=mid){
        temp.push_back(arr[i]);
        i++;
    }
    
    while(j<=en){
        temp.push_back(arr[j]);
        j++;
    }
    
    for(int idx=0; idx<temp.size(); idx++){
        arr[st+idx] = temp[idx];
    }
}

void seqMergeSort(vector<int> &arr, int st, int en){
    
    if(st < en){
        int mid = st + (en-st)/2;
        
        seqMergeSort(arr, st, mid);//left
        
        seqMergeSort(arr, mid+1, en);//Right
        merge(arr, st, mid, en);   
    }
}

void parMergeSort(vector<int> &arr, int st, int en){
    
    if(st < en){
        int mid = st + (en-st)/2;
        
        #pragma  omp parallel sections
        {
            #pragma omp section
            parMergeSort(arr, st, mid);//left
            #pragma omp section
            parMergeSort(arr, mid+1, en);//Right 
        }
        
        merge(arr, st, mid, en);
    }
}

int main(){
    int size;
    cout<<"Enter Number of Inputs : ";
    cin>>size;
    
    vector<int> arr(size), arr_copy;
    generateArray(arr, size);
    arr_copy = arr;
    
    double seqBubStart, seqBubEnd, seqMerStart, seqMerEnd, seqBubTime, seqMerTime;
    double parBubStart, parBubEnd, parMerStart, parMerEnd, parBubTime, parMerTime;
    
    seqBubStart = omp_get_wtime();
    seqBubbleSort(arr);
    seqBubEnd = omp_get_wtime();
    seqBubTime = (seqBubEnd - seqBubStart);
    cout<<"Sequential Bubble Time : "<<seqBubTime<<endl;
    
    parBubStart = omp_get_wtime();
    parBubbleSort(arr);
    parBubEnd = omp_get_wtime();
    parBubTime = (parBubEnd - parBubStart);
    cout<<"Parallel Bubble Time : "<<parBubTime<<endl;
    cout<<"SpeedUp : "<<(seqBubTime/parBubTime)<<endl;
    
    // arr = arr_copy;
    arr = arr_copy;
    seqMerStart = omp_get_wtime();
    seqMergeSort(arr,0, arr.size()-1);
    seqMerEnd = omp_get_wtime();
    seqMerTime = (seqMerEnd - seqMerStart);
    cout<<"Sequential Merge Time : "<<seqMerTime<<endl;
    
    arr = arr_copy;
    parMerStart = omp_get_wtime();
    parMergeSort(arr, 0, arr.size()-1);
    parMerEnd = omp_get_wtime();
    parMerTime = (parMerEnd - parMerStart);
    cout<<"Parallel Merge Time : "<<parMerTime<<endl;
    cout<<"SpeedUp : "<<(seqMerTime/parMerTime)<<endl;
}