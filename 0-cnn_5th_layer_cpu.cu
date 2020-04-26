#include <stdio.h>
#include <iostream>
#include <string.h>

/***DEFINING THE DEFINES FOR THE ARRAY INDICES****************************/
//#define N 1
#define C 384
#define H 15
#define W 15
#define R 3
#define S 3
#define M 256
#define E 13
#define F 13
#define U 1
using namespace std;
int main(int argc, char* argv[])
{
int batch_size = atoi(argv[1]);
/*************INITALIZING MATRICES*********************************/
float IP[batch_size][C][H][W];
float OP[batch_size][M][F][E];
float WT[M][C][R][S];
int a,b,c,d;
/*INITIALIZING WEIGHT MATRIX*/
for(a=0; a<M; a++){
	for (b=0; b<C; b++){
		for (c=0; c<R; c++){
			for(d=0; d<S; d++){
				WT[a][b][c][d] = (float)rand()/(float)(RAND_MAX+1.0);
}
}	
}
}
//printf("hello after weight\n");
/*INITIALIZING OUTPUT MATRIX*/
for(a=0; a<batch_size; a++){
	for (b=0; b<M; b++){
		for (c=0; c<F; c++){
			for(d=0; d<E; d++){
				OP[a][b][c][d] = 0;
}
}	
}
}
//printf("hello after op init\n");
/*INITIALIZING INPUT MATRIX*/
for(a=0; a<batch_size; a++){
	for (b=0; b<C; b++){
		for (c=0; c<H; c++){
			for(d=0; d<W; d++){
				if((c==0)||(d==0)||(c==14)||(d==14))
				IP[a][b][c][d] = 0;//zero padding to eliminate spurious values along the edges to align stride and filter boundary properly with image boundary
				else
				IP[a][b][c][d] = (float)rand()/(float)(RAND_MAX+1.0);;
}
}	
}
}
printf("hello after ininting input mat\n");
/***NAIVE 7 LAYER LOOP IMPLEMENTATION***/
int n,m,x,y,i,j,k;
for(n=0; n<batch_size; n++){
	for (m=0; m<M; m++){
		for (x=0; x<F; x++){
			for(y=0; y<E; y++){
				OP[n][m][x][y] = 0; // adding bias to output
				for (i=0; i<R; i++){
					for (j=0; j<S; j++){
						for(k=0; k<C; k++){
							OP[n][m][x][y] += IP[n][k][U*x+i][U*y+j]*WT[m][k][i][j];
							}
						}
					}
                 /*****ACTIVATION FUNCTION-RELU*******/
if(OP[n][m][x][y] < 0) 
	OP[n][m][x][y] = 0;

					}
				}	
			}
		}
/**print outputs**/
string filename ="layer_5_"+to_string(batch_size);
FILE *fp=fopen(filename.c_str(),"w+");

int e,f,g,h;
for(e=0; e<batch_size; e++){
	for (f=0; f<M; f++){
		for (g=0; g<F; g++){
			for(h=0; h<E; h++){
				//printf("the output is %f for index %d,%d,%d,%d.\n",OP[e][f][g][h], e,f,g,h);
				 //fprintf(fp,"%f,%d,%d,%d,%d.\n",OP[e][f][g][h], e,f,g,h);
				fprintf(fp,"%f\n",OP[e][f][g][h]);


}
}	
}
}
fclose(fp);
return 0;
}


