#include <stdio.h>
#include <iostream>
#include <cuda_fp16.h>
//#include <cooperative_groups.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <string.h>
//#include <bits/stdc++.h>
//#include <stdlib.h>
//# <time.h>
using namespace std;
//using namespace cooperative_groups;

/***DEFINING THE DEFINES FOR THE ARRAY INDICES****************************/
//#define N 128
#define C 3
#define H 227
#define W 227
#define R 11
#define S 11
#define M 96
#define E 55
#define F 55
#define U 4
__global__  void red_ch(float* d_r, float* d_o, int num_ch, int num_img, int num_wt, int height, int width)
{
int row = threadIdx.y; int col = threadIdx.x;
for (int y=0;y<2;y++){
for (int x=0;x<2;x++){
float red_sum = 0;
for(int i=0; i<num_ch; i++)
{       //if((2*row+y<height)&&(2*col+x<width))
	red_sum += d_o[i*(num_wt*num_img*height*width)+blockIdx.x*num_wt*height*width+blockIdx.y*height*width+(row+y*blockDim.y)*width+(col+x*blockDim.x)] ;	
}
if((row+y*blockDim.y<height)&&(col+x*blockDim.x<width))
d_r[blockIdx.x*num_wt*height*width+blockIdx.y*height*width+(row+y*blockDim.y)*width+(col+x*blockDim.x)] = red_sum;

}
}}
__global__ 
void ew_gpu_mmul(float* d_o, __half* d_i, __half* d_w, int width, int height, int stride, int ip_height, int wt_width, int num_wt,int num_img, int num_ch)
{//float prod=0;
int row = threadIdx.y; int col = threadIdx.x;
__shared__ __half s_w[R*S];
if(row*blockDim.x+col<R*S)
{
s_w[row*blockDim.x+col] = d_w[blockIdx.y*num_ch*wt_width*wt_width+blockIdx.z*wt_width*wt_width+(row*blockDim.x+col)];
}
__syncthreads();
for (int y=0; y<2; y++){
for (int x=0; x<2; x++){ 
float prod = 0;
for (int i=0; i<wt_width; i++){
  for (int j=0; j<wt_width; j++){
  float ip =__half2float(d_i[blockIdx.x*num_ch*ip_height*ip_height+blockIdx.z*ip_height*ip_height+(stride*(row+y*blockDim.y)+i)*ip_height+(stride*(col+x*blockDim.x)+j)]);
 //       float wt = d_w[blockIdx.y*num_ch*wt_width*wt_width+blockIdx.z*wt_width*wt_width+(i*wt_width+j)];
	  prod += ip*__half2float(s_w[(i*wt_width+j)]);
	  __syncthreads();
}
}
 if((row+y*blockDim.y<height)&&(col+x*blockDim.x<width))
{if(prod>=0)
  d_o[blockIdx.z*(num_wt*num_img*height*width)+blockIdx.x*num_wt*height*width+blockIdx.y*height*width+(row+y*blockDim.y)*width+(col+x*blockDim.x)] = prod;
}
}
}
if(row*blockDim.x+col < R*S)
{
  s_w[(row*blockDim.x+col)] = __float2half(0);
}
__syncthreads();
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
__half *IP = (__half*) malloc(batch_size*C*H*W*sizeof(__half));
//float IP[H][W];
float *OP = (float*) malloc(batch_size*M*F*E*sizeof(float));
//float OP[F][E];
float *OPG = (float*) malloc(batch_size*M*F*E*sizeof(float));
__half *WT = (__half*) malloc(M*C*R*S*sizeof(__half));
//float WT[R][S];
float* d_o;
__half* d_i;
__half* d_w;
float* d_r;
//clock_t cpu_start, gpu_start, cpu_end, gpu_end;
//int a,b,c,d;
int c,d,m,n,k;
/* WEIGHT MATRIX*/
for (m=0; m<M; m++){
for(k=0;k<C;k++){
for (c=0; c<R; c++){
	for(d=0; d<S; d++){
		//WT[c][d] = 2.0;
		//WT[m*C*R*S+k*R*S+c*S+d] = (int)k+1;
		WT[m*C*R*S+k*R*S+c*S+d] = __float2half((float)rand()/(float)(RAND_MAX+1.0));
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
	//if ((c<=1) || (d<=1) || (c>=29) || (d>=29))
        //IP[n*C*H*W+k*H*W+c*W+d] = 0;
        //else
        IP[n*C*H*W+k*H*W+c*W+d] = __float2half((float)rand()/(RAND_MAX+1.0));

}
}	
}
}
if(cudaSuccess != cudaMalloc((void**) &d_i,batch_size*C*H*W*sizeof(__half)))
{
printf("error in d_i malloc\n");
}
cudaMemcpy(d_i, IP, batch_size*C*H*W*sizeof(__half), cudaMemcpyHostToDevice);
if(cudaSuccess != cudaMalloc((void**) &d_w, M*C*R*S*sizeof(__half)))
{
printf("error in d_w malloc\n");	
}
cudaMemcpy(d_w, WT, M*C*R*S*sizeof(__half), cudaMemcpyHostToDevice);
if(cudaSuccess != cudaMalloc((void**) &d_o,(long int)C*batch_size*M*E*F*sizeof(float)))
{
printf("error in d_o malloc\n");
}
if(cudaSuccess != cudaMalloc((void**) &d_r,batch_size*M*E*F*sizeof(float)))
{
printf("error in d_r malloc\n");
}

clock_t cpu_start = clock();
//element_wise_mmul(OP, IP, WT, batch_size);
clock_t cpu_end = clock();
dim3 dimGrid(batch_size,96,3);
dim3 dimBlock(28,28,1);
dim3 dimGridRed(batch_size,96,1);
dim3 dimBlockRed(28,28,1);ew_gpu_mmul<<<dimGrid, dimBlock>>>(d_o,d_i,d_w,55,55,4,227,11,96,batch_size,3);
cudaDeviceSynchronize();
red_ch<<<dimGridRed, dimBlockRed>>>(d_r,d_o,3,batch_size,96,55,55);
cudaMemcpy(OPG,d_r,(long int)batch_size*M*E*F*sizeof(float), cudaMemcpyDeviceToHost);

/**print outputs**/
//int e,f,g,h;
int g,h,s,u,t;
float max_error = 0;
string filename = "layer_1_"+to_string(batch_size);
ifstream fin(filename.c_str());
string line ;
//for (t=0;t<C;t++){
for (u=0;u<batch_size;u++){
for (s=0;s<M;s++){
for (g=0; g<F; g++){
	for(h=0; h<E; h++){	
	getline(fin,line);
        float error = abs(OPG[u*M*F*E+s*E*F+g*E+h]-atof(line.c_str()));
	//float error = abs(OPG[u*M*F*E+s*E*F+g*E+h]-OP[u*M*F*E+s*E*F+g*E+h]);
	if(error > max_error)
	max_error = error;
//      printf("the output is %f for index %d, %d,%d,%d.\n",OP[u*M*F*E+s*E*F+g*E+h],u,s,g,h);
//       printf("diff CPU and GPU is %f for index %d,%d,%d,%d.\n", OPG[u*M*F*E+s*E*F+g*E+h]-OP[u*M*F*E+s*E*F+g*E+h],u,s,g,h);
//        printf("the output from GPU  is %f for index,%d,%d,%d,%d.\n",OPG[u*M*F*E+s*E*F+g*E+h],u,s,g,h);
}
}
}
}
fin.close();
printf("max error is %f\n", max_error);
//}
cout<<"time taken by cpu call is "<<((double)(cpu_end-cpu_start))/CLOCKS_PER_SEC<<"secs"<<endl;
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

