#include <stdio.h>
#include <iostream>
//#include <cooperative_groups.h>
#include <math.h>
#include <string.h>
#include <sstream>
#include <fstream>
//#include <bits/stdc++.h>
//#include <stdlib.h>
//#include <time.h>
using namespace std;
//using namespace cooperative_groups;

/***DEFINING THE DEFINES FOR THE ARRAY INDICES****************************/
//#define N 32
#define C 256
#define H 15
#define W 15
#define R 3
#define S 3
#define M 384
#define E 13
#define F 13
#define U 1


__global__ void red_ch(float* d_r, float* d_o, int num_ch, int num_img, int num_wt)
{
//printf("gpu2 started\n");
float red_sum = 0;
int row = threadIdx.y; int col = threadIdx.x;
for(int i=0; i<num_ch; i++)
{       
        red_sum += d_o[i*(num_wt*num_img*blockDim.x*blockDim.y)+blockIdx.x*num_wt*blockDim.x*blockDim.y+blockIdx.y*blockDim.x*blockDim.y+row*blockDim.x+col] ;
}
d_r[blockIdx.x*num_wt*blockDim.x*blockDim.y+blockIdx.y*blockDim.x*blockDim.y+row*blockDim.x+col] = red_sum;
}
__global__
void ew_gpu_mmul(float* d_o, float* d_i, float* d_w, int width, int height, int stride, int ip_height, int wt_width, int num_wt,int num_img, int num_ch)
{//printf("gpu started\n");
__shared__ float s_w[R*S];
__shared__ float s_i[H*W];
int row = threadIdx.y; int col = threadIdx.x;
if(row*width+col<R*S)
{
s_w[row*width+col] = d_w[blockIdx.y*num_ch*wt_width*wt_width+blockIdx.z*wt_width*wt_width+(row*width+col)];
}
{
int s_i_idx = row*blockDim.x+col;
s_i[s_i_idx] = d_i[blockIdx.x*num_ch*ip_height*ip_height+blockIdx.z*ip_height*ip_height+s_i_idx];
//s_i[s_i_idx+169] = d_i[blockIdx.x*num_ch*ip_height*ip_height+blockIdx.z*ip_height*ip_height+s_i_idx+169];
//s_i[s_i_idx+338] = d_i[blockIdx.x*num_ch*ip_height*ip_height+blockIdx.z*ip_height*ip_height+s_i_idx+338];
//s_i[s_i_idx+507] = d_i[blockIdx.x*num_ch*ip_height*ip_height+blockIdx.z*ip_height*ip_height+s_i_idx+507];
if(s_i_idx+169 < H*W)
s_i[s_i_idx+169] = d_i[blockIdx.x*num_ch*ip_height*ip_height+blockIdx.z*ip_height*ip_height+s_i_idx+169];
}
__syncthreads();

float prod = 0;
if((row<height) && (col<width))//earlier it was num_wt*height & num_img*width
{
for (int i=0; i<wt_width; i++){
	float3 ip = *((float3*)(s_i+(stride*row+i)*ip_height+stride*col));
        float3 wt = *((float3*)(s_w+i*wt_width));
	prod += ip.x*wt.x+ip.y*wt.y+ip.z*wt.z;
        __syncthreads();
}
if(prod>=0)
d_o[0*(num_wt*num_img*blockDim.x*blockDim.y)+blockIdx.x*num_wt*blockDim.x*blockDim.y+blockIdx.y*blockDim.x*blockDim.y+row*blockDim.x+col] += prod;
if(row*width+col<R*S){
     s_w[(row*width+col)] = 0;
__syncthreads();
}
}
}

void element_wise_mmul(float* output, float* input, float* weight, int batch_size)
{
int x,y,i,j,m,n,k;
for(n=0; n<batch_size; n++){
for (m=0 ; m<M; m++){
 for (x=0; x<F; x++){
         for(y=0; y<E; y++){
            //    OP[x][y] = 0; // adding bias to output
                 for (i=0; i<R; i++){
                         for (j=0; j<S; j++){
				for(k=0; k<C; k++){
				float ip = input[n*C*H*W+k*H*W+(U*x+i)*H+(U*y+j)];
				float wt = weight[m*C*R*S+k*R*S+i*S+j];

				float prod = ip*wt;
				if(prod>=0)
				output[n*E*F*M+m*E*F+x*E+y] += prod;
                                 //OP[x][y] += IP[U*x+i][U*y+j]*WT[i][j];
                                                         }}
                                                 }
                                        }
                                         }
 

}
}
}
int main(int argc, char* argv[])
{
int batch_size = atoi(argv[1]);
/*************INITALIZING MATRICES*********************************/
float *IP = (float*) malloc(batch_size*C*H*W*sizeof(float));
//float IP[H][W];
float *OP = (float*) malloc(batch_size*M*F*E*sizeof(float));
//float OP[F][E];
float *OPG = (float*) malloc(batch_size*M*F*E*sizeof(float));
float *WT = (float*) malloc(M*C*R*S*sizeof(float));
//float WT[R][S];
float* d_o;
float* d_i;
float* d_w;
float* d_r;
//clock_t cpu_start, gpu_start, cpu_end, gpu_end;
//int a,b,c,d;
int c,d,m,n,k;
/*INITIALIZING WEIGHT MATRIX*/
for (m=0; m<M; m++){
for(k=0;k<C;k++){
for (c=0; c<R; c++){
	for(d=0; d<S; d++){
		//WT[c][d] = 2.0;
		//WT[m*C*R*S+k*R*S+c*S+d] = (int)k+1;
		WT[m*C*R*S+k*R*S+c*S+d] = (float)rand()/(float)(RAND_MAX+1.0);
}	
}
}
}
/*INITIALIZING OUTPUT MATRIX*/
for (n=0; n<batch_size;n++){
for (m=0; m<M; m++){
for (c=0; c<F; c++){
	for(d=0; d<E; d++){
		//OP[c][d] = 0;
		OP[n*M*F*E+m*F*E+c*E+d] = 0;
}
}	
}
}
/*INITIALIZING INPUT MATRIX*/
for (n=0; n<batch_size; n++){
for(k=0;k<C;k++){
for (c=0; c<H; c++){
	for(d=0; d<W; d++){
	//	IP[c][d] = (a+b+c+d);
	if ((c==0) || (d==0) || (c==14) || (d==14))
        IP[n*C*H*W+k*H*W+c*W+d] = 0;
        else
        IP[n*C*H*W+k*H*W+c*W+d] = (float)rand()/(RAND_MAX+1.0);
	
}
}	
}
}
if(cudaSuccess != cudaMalloc((void**) &d_i,batch_size*C*H*W*sizeof(float)))
{
printf("error in d_i malloc\n");
}
cudaMemcpy(d_i, IP, batch_size*C*H*W*sizeof(float), cudaMemcpyHostToDevice);
if(cudaSuccess != cudaMalloc((void**) &d_w, M*C*R*S*sizeof(float)))
{
printf("error in d_w malloc\n");	
}
cudaMemcpy(d_w, WT, M*C*R*S*sizeof(float), cudaMemcpyHostToDevice);
if(cudaSuccess != cudaMalloc((void**) &d_o,(long int)batch_size*M*E*F*sizeof(float)))
{
printf("error in d_o malloc\n");
}
if(cudaSuccess != cudaMalloc((void**) &d_r,batch_size*M*E*F*sizeof(float)))
{
printf("error in d_r malloc\n");
}

//cpu_start = clock();
//element_wise_mmul(OP, IP, WT, batch_size);
printf("cpu done\n");
//cpu_end = clock();
dim3 dimGrid(batch_size,384,256);
dim3 dimBlock(13,13,1);
//dim3 dimGridRed(batch_size,384,1);
//dim3 dimBlockRed(13,13,1);
//int op_height = 3; int op_width = 3; int stride = 1; int ip_height = 4;int wt_height = 2; int num_wt = 96; int num_img = 1; int num_ch = 384;
//gpu_start = clock();
ew_gpu_mmul<<<dimGrid, dimBlock>>>(d_o,d_i,d_w,13,13,1,15,3,384,batch_size,256);
cudaDeviceSynchronize();
//red_ch<<<dimGridRed, dimBlockRed>>>(d_r,d_o,256,batch_size,384);
//gpu_end = clock();
//void *kernelArgs[] = {(void *)&d_o, (void *)&d_i, (void *)&d_w,(void *)&op_height, (void *)&op_width, (void *)&stride, (void *)&ip_height,(void *)&wt_height, (void *)&num_wt, (void *)&num_img, (void *)&num_ch };
//cudaLaunchCooperativeKernel((void*)ew_gpu_mmul,dimGrid,dimBlock,kernelArgs,0,NULL);
//cudaDeviceSynchronize();
cudaMemcpy(OPG,d_o,batch_size*M*E*F*sizeof(float), cudaMemcpyDeviceToHost);

/**print outputs**/
//int e,f,g,h;
int g,h,s,u;
float max_error = 0;
string filename = "layer_3_"+to_string(batch_size);
ifstream fin(filename.c_str());
string line ;
//for (t=0;t<C;t++){
for (u=0;u<batch_size;u++){
for (s=0;s<M;s++){
for (g=0; g<F; g++){
	for(h=0; h<E; h++){	
	getline(fin,line);
        float error = abs(OPG[u*M*F*E+s*E*F+g*E+h]-atof(line.c_str()));
//	float error = abs(OPG[u*M*F*E+s*E*F+g*E+h]-OP[u*M*F*E+s*E*F+g*E+h]);
	if(error > max_error)
	max_error = error;
 //     printf("the output is %f for index %d, %d,%d,%d.\n",OP[u*M*F*E+s*E*F+g*E+h],u,s,g,h);
   //    printf("diff CPU and GPU is %f for index %d,%d,%d,%d.\n", OPG[u*M*F*E+s*E*F+g*E+h]-OP[u*M*F*E+s*E*F+g*E+h],u,s,g,h);
     //   printf("the output from GPU  is %f for index,%d,%d,%d,%d.\n",OPG[u*M*F*E+s*E*F+g*E+h],u,s,g,h);
}
}
}
}
fin.close();
printf("max error is %f\n", max_error);
//}
//cout<<"time taken by cpu call is "<<((double)(cpu_end-cpu_start))/CLOCKS_PER_SEC<<"secs"<<endl;
//cout<<"time taken by gpu call is "<<((double)(gpu_end-gpu_start))/CLOCKS_PER_SEC<<"secs"<<endl;

cudaFree(d_o);
cudaFree(d_i);
cudaFree(d_w);
cudaFree(d_r);
free(OPG);
free(IP);
free(WT);
free(OP);	
return 0;
}

