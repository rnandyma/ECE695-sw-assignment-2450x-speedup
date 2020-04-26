#include <stdio.h>
#include <iostream>
#include <math.h>
#include <bits/stdc++.h>
#include <cuda_fp16.h>
#include <sstream>
#include <fstream>
#include <string.h>
//#include <stdlib.h>
//#include <time.h>
using namespace std;
/***DEFINING THE DEFINES FOR THE ARRAY INDICES****************************/
//#define N 128
#define C 96
#define H 31
#define W 31
#define R 5
#define S 5
#define M 256
#define E 27
#define F 27
#define U 1
__global__ 
void ew_gpu_mmul(float* d_o, __half* d_i, __half* d_w, int width, int height, int stride, int ip_height, int wt_width, int num_wt,int num_img, int num_ch)
{
int row = threadIdx.y; int col = threadIdx.x;
if((row<height) && (col<width))
{
for (int i=0; i<wt_width; i++){
  for (int j=0; j<wt_width; j++){
   for(int k=0; k<num_ch; k++){
        d_o[blockIdx.x*num_wt*blockDim.x*blockDim.y+blockIdx.y*blockDim.x*blockDim.y+row*blockDim.x+col] += __half2float(d_i[blockIdx.x*num_ch*ip_height*ip_height+k*ip_height*ip_height+(stride*(row)+i)*ip_height+(stride*(col)+j)])*__half2float(d_w[blockIdx.y*num_ch*wt_width*wt_width+k*wt_width*wt_width+(i*wt_width+j)]);

}
}
}
} 
}
void element_wise_mmul(float* output, __half* input, __half* weight, int batch_size)
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
				__half ip = input[n*C*H*W+k*H*W+(U*x+i)*H+(U*y+j)];
				__half wt = weight[m*C*R*S+k*R*S+i*S+j];
				float prod = __half2float(ip)*__half2float(wt);
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

//clock_t cpu_start, gpu_start, cpu_end, gpu_end;
//int a,b,c,d;
int c,d,m,n,k;
/*INITIALIZING WEIGHT MATRIX*/
for (m=0; m<M; m++){
for(k=0;k<C;k++){
for (c=0; c<R; c++){
	for(d=0; d<S; d++){
		//WT[c][d] = 2.0;
		WT[m*C*R*S+k*R*S+c*S+d] = __float2half((float)rand()/(RAND_MAX+1.0));
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
	if ((c<=1) || (d<=1) || (c>=29) || (d>=29))
	IP[n*C*H*W+k*H*W+c*W+d] = __float2half(0);
	else
	IP[n*C*H*W+k*H*W+c*W+d] = __float2half((float)rand()/(RAND_MAX+1.0));

}
}	
}
}
cudaMalloc((void**) &d_i,batch_size*C*H*W*sizeof(__half));
cudaMemcpy(d_i, IP, batch_size*C*H*W*sizeof(__half), cudaMemcpyHostToDevice);
cudaMalloc((void**) &d_w, M*C*R*S*sizeof(__half));
cudaMemcpy(d_w, WT, M*C*R*S*sizeof(__half), cudaMemcpyHostToDevice);
cudaMalloc((void**) &d_o, batch_size*M*E*F*sizeof(float));
//cpu_start = clock();
//clock_t start, end;
//start = clock();
//element_wise_mmul(OP, IP, WT, batch_size);
//end = clock();
//printf("cpu time is %f secs\n", (float)(end-start)/CLOCKS_PER_SEC);
//cpu_end = clock();
dim3 dimGrid(batch_size,256,1);
dim3 dimBlock(27,27,1);
//gpu_start = clock();
ew_gpu_mmul<<<dimGrid, dimBlock>>>(d_o,d_i,d_w,27,27,1,31,5,256,batch_size,96);
//gpu_end = clock();
cudaMemcpy(OPG,d_o, batch_size*M*E*F*sizeof(float), cudaMemcpyDeviceToHost);
float max_error = 0;
int g,h,s,u;
string filename = "layer_2_"+to_string(batch_size);
ifstream fin(filename.c_str());
string line ;
for (u=0;u<batch_size;u++){
for (s=0;s<M;s++){
for (g=0; g<F; g++){
	for(h=0; h<E; h++){
        getline(fin,line);
        float error = abs(OPG[u*M*F*E+s*E*F+g*E+h]-atof(line.c_str()));
	//float error = abs(OPG[u*M*F*E+s*E*F+g*E+h]-OP[u*M*F*E+s*E*F+g*E+h]);
	if (error > max_error)
		max_error = error;
		
 //      printf("the output is %f for index %d, %d,%d,%d.\n",OP[u*M*F*E+s*E*F+g*E+h],u,s,g,h);
  //     printf("diff CPU and GPU is %f for index %d,%d,%d,%d.\n", OPG[u*M*F*E+s*E*F+g*E+h]-OP[u*M*F*E+s*E*F+g*E+h],u,s,g,h);
   //     printf("the output from GPU  is %f for index %d,%d,%d,%d.\n",OPG[u*M*F*E+s*E*F+g*E+h],u,s,g,h);
}
}
}
}
fin.close();
printf("max error = %f\n", max_error);
//cout<<"time taken by cpu call is "<<((double)(cpu_end-cpu_start))/CLOCKS_PER_SEC<<"secs"<<endl;
//cout<<"time taken by gpu call is "<<((double)(gpu_end-gpu_start))/CLOCKS_PER_SEC<<"secs"<<endl;

cudaFree(d_o);
cudaFree(d_i);
cudaFree(d_w);
free(OPG);
free(IP);
free(WT);
free(OP);	
return 0;
}

