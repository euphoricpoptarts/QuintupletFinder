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
#include <numeric>

using namespace std;

void readWords(const string fname, vector<string>& output){
    ifstream f(fname);
    if(!f.is_open()) return;
    string line;
    while(getline(f, line)){
        output.push_back(line);
    }
}

//convert words to bit-string
uint32_t cook(const string& word){
    uint32_t x = 0;
    for(auto& c : word){
        x |= (1 << (c - 'a'));
    }
    return x;
}

//collect all unique bit-strings with exactly five bits set
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

//map words to the idx of their bit-string in cooked
vector<vector<string>> wordMap(const vector<string>& words, const vector<uint32_t>& cooked){
    unordered_map<uint32_t, int> m;
    for(int i = 0; i < cooked.size(); i++){
        m[cooked[i]] = i;
    }
    vector<vector<string>> wM(cooked.size(), vector<string>());
    for(auto& word : words){
        uint32_t x = cook(word);
        if(std::popcount(x) == 5){
            int idx = m[x];
            wM[idx].push_back(word);
        }
    }
    return wM;
}

//sort in increasing order of degree
vector<uint32_t> sortByAdj(const vector<uint32_t>& cooked){
    int n = cooked.size();
    vector<int> adjC(n, 0);
#pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if((cooked[i] & cooked[j]) == 0){
                adjC[i]++;
            }
        }
    }
    vector<int> idx(n, 0);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&adjC] (int a, int b) {return adjC[a] < adjC[b];});
    vector<uint32_t> sorted;
    for(auto& i : idx){
        sorted.push_back(cooked[i]);
    }
    return sorted;
}

//create a sparse adjacency matrix from cooked
vector<vector<int>> adjList(const vector<uint32_t>& cooked){
    int n = cooked.size();
    vector<vector<int>> skip(n, vector<int>(n + 1, 0));
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        skip[i][n] = n;
        uint32_t A = cooked[i];
        for (int j = n - 1; j >= i; --j) {
            uint32_t B = cooked[j];
            skip[i][j] = ((A & B) == 0) ? j : skip[i][j + 1];
        }
    }
    return skip;
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

//find all quintuplets of indices in cooked whose entries contain 25 unique set bits
vector<quint> getQuints(const vector<uint32_t>& cooked, const vector<vector<int>>& adj){
#pragma omp declare reduction (merge:vector<quint>:omp_out=add(omp_out,omp_in))
    int n = cooked.size();
    vector<quint> result;
#pragma omp parallel for schedule(dynamic, 1) reduction(merge: result)
    for(int i = 0; i < n; i++){
        uint32_t x1 = cooked[i];
        for(int j = adj[i][i]; j < n; j++, j = adj[i][j]){
            uint32_t x2 = x1 | cooked[j];
            for(int k = adj[j][j]; k < n; k++, k = adj[i][k], k = adj[j][k]){
                if((x2 & cooked[k]) != 0) continue;
                uint32_t x3 = x2 | cooked[k];
                for(int l = adj[k][k]; l < n; l++, l = adj[i][l], l = adj[j][l], l = adj[k][l]){
                    if((x3 & cooked[l]) != 0) continue;
                    uint32_t x4 = x3 | cooked[l];
                    for(int m = adj[l][l]; m < n;  m++, m = adj[i][m], m = adj[j][m], m = adj[k][m], m = adj[l][m]){
                        if((x4 & cooked[m]) == 0){
                            quint y{i, j, k, l, m};
                            result.push_back(y);
                        }
                    }
                }
            }
        }
    }
    return result;
}

//print the words in the corresponding vector to the idx
int printWord(const vector<vector<string>>& wM, int x){
    for(auto& w : wM[x]){
        cout << w << " ";
    }
    cout << endl;
    return wM[x].size();
}

int main(){
    using tp = typename chrono::high_resolution_clock::time_point;
    vector<string> words;
    readWords("wordle-nyt-allowed-guesses.txt", words);
    readWords("wordle-nyt-answers-alphabetical.txt", words);
    //readWords("fives.txt", words);
    sort(words.begin(), words.end());
    vector<uint32_t> cooked = cookVector(words);
    //sorting cooked by increasing degree helps decrease
    //the total entries in adjList
    //cooked = sortByAdj(cooked);
    vector<vector<string>> wM = wordMap(words, cooked);
    vector<vector<int>> adj = adjList(cooked);
    tp t1 = chrono::high_resolution_clock::now();
    vector<quint> q = getQuints(cooked, adj);
    tp t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> d = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Found quintuplets in " << d.count() << "s" << endl;
    int count = 0;
    for(auto x : q){
        cout << "Quintuplet" << endl;
        int y = 1;
        y *= printWord(wM, x.i);
        y *= printWord(wM, x.j);
        y *= printWord(wM, x.k);
        y *= printWord(wM, x.l);
        y *= printWord(wM, x.m);
        cout << endl;
        count += y;
    }
    cout << count << endl;
    return 0;
}
