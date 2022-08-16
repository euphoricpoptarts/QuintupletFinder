#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <bit>
#include <omp.h>
#include <chrono>
#include <list>
#include <algorithm>

using namespace std;

void readWords(const string fname, vector<string>& output){
    ifstream f(fname);
    if(!f.is_open()) return;
    string line;
    while(getline(f, line)){
        output.push_back(line);
    }
}

uint32_t cook(const string& word){
    uint32_t x = 0;
    for(auto& c : word){
        x |= (1 << (c - 'a'));
    }
    return x;
}

vector<uint32_t> cookVector(const vector<string>& words){
    vector<uint32_t> cooked;
    unordered_set<uint32_t> uniq;
    for(auto& word : words){
        uint32_t x = cook(word);
        if(std::popcount(x) == 5){
            if(uniq.find(x) == uniq.end()){
                cooked.push_back(x);
                uniq.insert(x);
            }
        }
    }
    return cooked;
}

vector<vector<int>> adjList(const vector<uint32_t>& cooked){
    int n = cooked.size();
    vector<vector<int>> adj(n, vector<int>());
#pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            if((cooked[i] & cooked[j]) == 0){
                adj[i].push_back(j);
            }
        }
    }
    return adj;
}

struct quint {
    int i, j, k, l, m;
};

vector<quint> add(vector<quint>& a, const vector<quint>& b){
    for(auto& x : b){
        a.push_back(x);
    }
    return a;
}

unordered_set<uint32_t> pairs(const vector<uint32_t>& cooked){
    int n = cooked.size();
    unordered_set<uint32_t> p;
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            if((cooked[i] & cooked[j]) == 0){
                p.insert(cooked[i] | cooked[j]);
            }
        }
    }
    return p;
}

int getQuints(const vector<uint32_t>& cooked, const vector<vector<int>>& adj, const unordered_set<uint32_t>& p){
#pragma omp declare reduction (merge:vector<quint>:omp_out=add(omp_out,omp_in))
    int n = cooked.size();
    vector<quint> result;
    int count = 0;
    char missing[3] = {'j','q','x'};
#pragma omp parallel for schedule(dynamic, 1) reduction(+: count)
    for(int i = 0; i < n; i++){
        uint32_t x1 = cooked[i];
        for(auto j : adj[i]){
            uint32_t x2 = x1 | cooked[j];
            for(auto k : adj[j]){
                if((x2 & cooked[k]) != 0) continue;
                uint32_t x3 = x2 | cooked[k];
                x3 = (~x3) & 0b11111111111111111111111111;
                for(int s = 0; s < 3; s++){
                    uint32_t q = 1 << (missing[s] - 'a');
                    q = q ^ x3;
                    if(popcount(q) == 10 && p.find(q) != p.end()) count++;
                }
                //for(auto l : adj[k]){
                //    if((x3 & cooked[l]) != 0) continue;
                //    uint32_t x4 = x3 | cooked[l];
                //    for(auto m : adj[l]){
                //        if((x4 & cooked[m]) == 0){
                //            quint y{i, j, k, l, m};
                //            result.push_back(y);
                //        }
                //    }
                //}
            }
        }
    }
    return count;
}

int main(){
    using tp = typename chrono::high_resolution_clock::time_point;
    vector<string> words;
    readWords("wordle-nyt-allowed-guesses.txt", words);
    readWords("wordle-nyt-answers-alphabetical.txt", words);
    vector<uint32_t> cooked = cookVector(words);
    sort(cooked.begin(), cooked.end());
    reverse(cooked.begin(), cooked.end());
    vector<vector<int>> adj = adjList(cooked);
    unordered_set<uint32_t> p = pairs(cooked);
    tp t1 = chrono::high_resolution_clock::now();
    int q = getQuints(cooked, adj, p);
    cout << q << endl;
    tp t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> d = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << d.count() << endl;
    return 0;
}
