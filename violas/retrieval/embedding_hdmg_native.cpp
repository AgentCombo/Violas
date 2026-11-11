#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>
static double dot(const float* a,const float* b,int d) {
    double s=0;
    #pragma omp simd reduction(+:s)
    for(int i=0;i<d;i++) s+=double(a[i])*double(b[i]);
    return s;
}
static float fast_dot(const float* a,const float* b,int d) {
    float s=0;
    #pragma omp simd reduction(+:s)
    for(int i=0;i<d;i++) s+=a[i]*b[i];
    return s;
}
// Packed immutable gallery and the existing HDMG edges; no query-answer cache.
extern "C" int embedding_search(int n,int g,int d,const float* vectors,const double* norms,
 const float* centers,const float* pivots,const int* edges,const int* edge_offsets,
 const int* members,const int* member_offsets,const int* entries,int entry_count,int global_entry,
 const float* query,int group_budget,int navigation_budget,int exhaustive,int* output,double* scores,int* stats) {
    const double qnorm=std::sqrt(dot(query,query,d));
    std::vector<double> priority(g,-1e300);
    std::vector<uint8_t> seen(g,0);
    std::priority_queue<std::pair<double,int>> frontier;
    std::vector<int> discovered;
    auto discover=[&](int node) {
        if(seen[node]) return;
        seen[node]=1;
        double value=fast_dot(centers+node*d,query,d);
        for(int j=0;j<4;j++) value=std::max(value,double(fast_dot(pivots+(node*4+j)*d,query,d)));
        priority[node]=value;
        frontier.push({value,-node});discovered.push_back(node);
    };
    int expanded=0,edge_reads=0;
    if(!exhaustive) {
        std::vector<std::pair<double,int>> entry_scores;
        for(int i=0;i<entry_count;i++) entry_scores.push_back({fast_dot(centers+entries[i]*d,query,d),-entries[i]});
        std::sort(entry_scores.rbegin(),entry_scores.rend());
        for(int i=0;i<std::min(3,entry_count);i++) discover(-entry_scores[i].second);
        if(global_entry>=0) discover(global_entry);
        while(!frontier.empty() && expanded<navigation_budget) {
            int node=-frontier.top().second;frontier.pop();expanded++;
            for(int j=edge_offsets[node];j<edge_offsets[node+1];j++){discover(edges[j]);edge_reads++;}
        }
        auto cmp=[&](int a,int b){return priority[a]!=priority[b]?priority[a]>priority[b]:a<b;};
        int take=std::min(group_budget,int(discovered.size()));
        std::partial_sort(discovered.begin(),discovered.begin()+take,discovered.end(),cmp);
        discovered.resize(take);
    }
    int count=0;
    for(int i=0;i<3;i++){output[i]=-1;scores[i]=-1e300;}
    std::vector<int> candidates;
    if(exhaustive) {candidates.resize(n);for(int i=0;i<n;i++)candidates[i]=i;}
    else {for(int node:discovered)for(int j=member_offsets[node];j<member_offsets[node+1];j++)candidates.push_back(members[j]);}
    std::vector<double> approximate(candidates.size());
    double best[3]={-1e300,-1e300,-1e300};
    for(size_t j=0;j<candidates.size();j++) {
        int i=candidates[j];
        double value=norms[i]>0 && qnorm>0?double(fast_dot(vectors+i*d,query,d))/(norms[i]*qnorm):0.;
        approximate[j]=value;count++;
        for(int p=0;p<3;p++)if(value>best[p]){for(int k=2;k>p;k--)best[k]=best[k-1];best[p]=value;break;}
    }
    // Float32 dot error is bounded by gamma_d * ||x|| ||q||. Recompute
    // every potentially top-three score in float64, including boundary ties.
    const double unit_roundoff=1.1920928955078125e-7;
    const double error=(d*unit_roundoff)/(1.-d*unit_roundoff);
    for(size_t j=0;j<candidates.size();j++) {
        if(approximate[j]<best[2]-2*error)continue;
        int i=candidates[j];
        double value=norms[i]>0 && qnorm>0?dot(vectors+i*d,query,d)/(norms[i]*qnorm):0.;
        for(int p=0;p<3;p++) if(value>scores[p] || (value==scores[p] && (output[p]<0 || i<output[p]))) {
            for(int k=2;k>p;k--){output[k]=output[k-1];scores[k]=scores[k-1];}
            output[p]=i;scores[p]=value;break;
        }
    }
    stats[0]=count;stats[1]=exhaustive?g:int(discovered.size());stats[2]=expanded;stats[3]=edge_reads;
    return std::min(3,count);
}
