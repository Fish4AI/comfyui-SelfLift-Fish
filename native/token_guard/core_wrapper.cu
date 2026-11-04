
#include <cuda_runtime.h>
extern "C" void launch_sol_attn_core(
    void* workspace, void* out, const void* vscale, void* kmean_next, void* vamax_out,
    const void* blen, int tail,
    int batch, int seq_len, int num_heads,
    float tau, float scale, const void* ext_threshold,
    int sink_start, int sink_end, int sink_q_start, int sink_q_end,
    int n_tok, cudaStream_t stream);
extern "C" __declspec(dllexport) int fish_core(
 void* workspace,void* out,const void* vscale,void* kmean_next,void* vamax_out,
 const void* threshold,const void* blen,int T,int H,float tau,float scale,
 int ss,int se,int sqs,int sqe,int tail,int aug,void* stream_ptr) {
 try {
 launch_sol_attn_core(workspace,out,vscale,kmean_next,vamax_out,blen,tail,
 1,T,H,tau,scale,threshold,ss,se,sqs,sqe,aug,(cudaStream_t)stream_ptr);
 } catch(...) { return -1; }
 return (int)cudaGetLastError();
}
