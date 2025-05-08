#include<iostream>
#include<vector>
#include<stack>
#include<queue>
#include<omp.h>

using namespace std;

void seqDFS(vector<vector<int>> &adj, int n, int start) {
    vector<bool> visited(n, false);
    stack<int> st;
    st.push(start);

    while(!st.empty()) {
        int node = st.top();
        st.pop();

        for(auto i : adj[node]) {
            if(!visited[i]) {
                visited[i] = true;
                st.push(i);
            }
        }
    }
}

void parDFS(vector<vector<int>> &adj, int n, int start) {
    vector<bool> visited(n, false);
    stack<int> st;
    st.push(start);

    while(!st.empty()) {
        int node;

        #pragma omp critical
        {
        if(!st.empty()) {
            node = st.top();
            st.pop();
        }
    }

        for(auto i : adj[node]) {

            #pragma omp critical
            {
                if(!visited[i]) {
                visited[i] = true;
                st.push(i);
                }
            }   
        }
    }
}

void seqBFS(vector<vector<int>> &adj, int n, int start) {
    vector<bool> visited(n, false);
    queue<int> que;
    que.push(start);

    while(!que.empty()) {
        int node = que.front();
        que.pop();

        for(auto i : adj[node]) {
            if(!visited[i]) {
                visited[i] = true;
                que.push(i);
            }
        }
    }
}

void parBFS(vector<vector<int>> &adj, int n, int start) {
    vector<bool> visited(n, false);
    queue<int> que;
    que.push(start);

    while(!que.empty()) {
        int node;

        #pragma omp critical
        {
            if(!que.empty()) {
                node = que.front();
                que.pop();
            }
        }

        for(auto i : adj[node]) {

            #pragma omp critical
            {
                if(!visited[i]) {
                    visited[i] = true;
                    que.push(i);
                }
            }
        }
    }
}

void generateRandom(vector<vector<int>> &adj, int u, int v) {
    adj[u].push_back(v);
    adj[v].push_back(u);
}

int main() {
    int size;
    cout<<"Enter Size : ";
    cin>>size;
    vector<vector<int>> adj(size);

    for(int i=0; i<size; i++) {
        generateRandom(adj, rand()%size, rand()%size);
    }

    double seqBFsStart, seqBFsEnd, parBFsStart, parBFsEnd, seqBFsTime, parBFsTime;
    double seqDFsStart, seqDFsEnd, parDFsStart, parDFsEnd, seqDFsTime, parDFsTime;

    
    seqBFsStart = omp_get_wtime();
    seqBFS(adj, size, 0);
    seqBFsEnd = omp_get_wtime();
    seqBFsTime = (seqBFsEnd - seqBFsStart);
    cout<<"Sequential BFS Time : "<<seqBFsTime<<endl<<endl;

    parBFsStart = omp_get_wtime();
    parBFS(adj, size, 0);
    parBFsEnd = omp_get_wtime();
    parBFsTime = (parBFsEnd - parBFsStart);
    cout<<"Parallel BFS Time : "<<parBFsTime<<endl<<endl;

    cout<<"BFS Speedup : "<<(seqBFsTime/parBFsTime)<<endl<<endl;



    seqDFsStart = omp_get_wtime();
    seqDFS(adj, size, 0);
    seqDFsEnd = omp_get_wtime();
    seqDFsTime = (seqDFsEnd - seqDFsStart);
    cout<<"Sequential DFS Time : "<<seqDFsTime<<endl<<endl;

    parDFsStart = omp_get_wtime();
    parDFS(adj, size, 0);
    parDFsEnd = omp_get_wtime();
    parDFsTime = (parDFsEnd - parDFsStart);
    cout<<"Parallel DFS Time : "<<parDFsTime<<endl<<endl;

    cout<<"DFS Speedup : "<<(seqDFsTime/parDFsTime)<<endl<<endl;
}
////g++ -fopenmp filename.cpp -o filename
//./a.out
//200 input 
//Shared resources are variables, data structures,
// or memory locations that multiple threads can access
// or modify at the same time during parallel execution.