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

//more frequent letters should have more significance
void initM(char* m){
    m['a'] = 24;
    m['b'] = 9;
    m['c'] = 16;
    m['d'] = 14;
    m['e'] = 25;
    m['f'] = 8;
    m['g'] = 10;
    m['h'] = 11;
    m['i'] = 22;
    m['j'] = 27;
    m['k'] = 5;
    m['l'] = 17;
    m['m'] = 12;
    m['n'] = 19;
    m['o'] = 21;
    m['p'] = 13;
    m['q'] = 26;
    m['r'] = 23;
    m['s'] = 18;
    m['t'] = 20;
    m['u'] = 15;
    m['v'] = 4;
    m['w'] = 6;
    m['x'] = 3;
    m['y'] = 7;
    m['z'] = 2;
}

//convert words to bit-string
uint32_t cook(const string& word, const char* m){
    uint32_t x = 0;
    for(auto& c : word){
        x |= 1 << m[c];
    }
    return x;
}

//collect all unique bit-strings with exactly five bits set
vector<uint32_t> cookVector(const vector<string>& words){
    vector<uint32_t> cooked;
    unordered_set<uint32_t> uniq;
    char m[256];
    initM(m);
    for(auto& word : words){
        uint32_t x = cook(word, m);
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
    char cm[256];
    initM(cm);
    for(auto& word : words){
        uint32_t x = cook(word, cm);
        if(std::popcount(x) == 5){
            int idx = m[x];
            wM[idx].push_back(word);
        }
    }
    return wM;
}

//create a sparse adjacency matrix from cooked
uint16_t* adjList(const vector<uint32_t>& cooked){
    int n = cooked.size();
    int cols = n+1;
    uint16_t* skip = new uint16_t[n*cols];
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        skip[i*cols + n] = n;
        uint32_t A = cooked[i];
        for (int j = n - 1; j >= i; j--) {
            uint32_t B = cooked[j];
            skip[i*cols + j] = ((A & B) == 0) ? static_cast<uint16_t>(j) : skip[i*cols + j + 1];
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
vector<quint> getQuints(const vector<uint32_t>& cooked, const uint16_t* adj){
#pragma omp declare reduction (merge:vector<quint>:omp_out=add(omp_out,omp_in))
    int n = cooked.size();
    int cols = n+1;
    vector<quint> result;
    vector<int> first(n, 0);
#pragma omp parallel for
    for(int i = 0; i < n; i++){
        first[i] = adj[i*cols+i];
    }
    uint32_t jq = (1 << 26) | (1 << 27);
#pragma omp parallel for schedule(dynamic, 1) reduction(merge: result)
    for(int i = 0; i < n; i++){
        uint32_t x1 = cooked[i];
        if((x1 & jq) == 0) continue;
        for(int j = first[i]; j < n; j++, j = adj[i*cols+j]){
            uint32_t x2 = x1 | cooked[j];
            for(int k = first[j]; k < n; k++, k = adj[i*cols+k], k = adj[j*cols+k]){
                if((x2 & cooked[k]) != 0) continue;
                uint32_t x3 = x2 | cooked[k];
                for(int l = first[k]; l < n; l++, l = adj[i*cols+l], l = adj[j*cols+l], l = adj[k*cols+l]){
                    if((x3 & cooked[l]) != 0) continue;
                    uint32_t x4 = x3 | cooked[l];
                    for(int m = first[l]; m < n;  m++, m = adj[i*cols+m], m = adj[j*cols+m], m = adj[k*cols+m], m = adj[l*cols+m]){
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

struct sortf {
    bool operator()(const uint32_t& x, const uint32_t& y) const {
        uint32_t jq = (1 << 26) | (1 << 27);
        return (x ^ jq) < (y ^ jq);
    }
};

int main(){
    using tp = typename chrono::high_resolution_clock::time_point;
    tp t1 = chrono::high_resolution_clock::now();
    vector<string> words;
    readWords("wordle-nyt-allowed-guesses.txt", words);
    readWords("wordle-nyt-answers-alphabetical.txt", words);
    //readWords("fives.txt", words);
    vector<uint32_t> cooked = cookVector(words);
    //sort cooked by value to create larger gaps in skiptable
    //move j and q containing words to front of sorted array
    sort(cooked.begin(), cooked.end(), sortf());
    vector<vector<string>> wM = wordMap(words, cooked);
    uint16_t* adj = adjList(cooked);
    vector<quint> q = getQuints(cooked, adj);
    delete[] adj;
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
    tp t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> d = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Found quintuplets in " << d.count() << "s" << endl;
    return 0;
}
