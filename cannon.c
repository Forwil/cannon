#include <stdio.h>
#include <sys/sysinfo.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <mpi.h>
#include <math.h>

#define UP 0
#define RIGHT 1
#define DOWN 2
#define LEFT 3

#define MIN(a,b) (((a)>(b))? (b):(a))

struct threadArg {
	int tid,n1,n2,n3,numthreads;
	double *B,*A,*C;
};

void my_read(FILE *f,double *p,int time,int size,int offset,int idi,int idj)
{
	int start = sizeof(int) * 2 + (idi * offset * size + idj * size)* sizeof(double);
///	printf("%d ,%d %d\n",start,idi,idj);
	for(int i=0;i<time;i++)
	{
		fseek(f,start,SEEK_SET);
		fread(p,sizeof(double),size,f);
		start += offset * sizeof(double);
		p += size;
	}
}

void my_write(FILE *f,double *p,int time,int size,int offset,int idi,int idj)
{
	int start = sizeof(int) * 2 + (idi * offset * size + idj * size)* sizeof(double);
//	printf("%d C\n",start);
	for(int i=0;i<time;i++)
	{
		fseek(f,start,SEEK_SET);
		fwrite(p,sizeof(double),size,f);
		start += offset * sizeof(double);
		p += size;
	}
}

void output(double *p,int n,int m)
{
	for(int i= 0;i<n;i++)
	{
		for(int j= 0;j<m;j++)
			printf("%f\t",*(p + i*m + j));
		printf("\n");
	}
}

int get(int id,int k,int pos)
{
	int x,y;
	x = id / k;
	y = id % k;
	switch (pos) 
	{
		case UP:
				x -= 1;
				break;
		case RIGHT:
				y += 1;
				break;
		case DOWN:
				x += 1;
				break;
		case LEFT:
				y -= 1;
				break;	
	}
	x = (x + k) % k;
	y = (y + k) % k;
	return x * k + y;
}

void* worker(void * arg)
{
	struct threadArg* myarg = (struct threadArg *)arg;

	for(int i= myarg->tid;i < myarg->n1; i+=myarg->numthreads) 
	for(int k= 0;k < myarg->n2;k++)
	for(int j= 0;j < myarg->n3;j++)
	*(myarg->C + i*myarg->n3 + j) += *(myarg->A + i*myarg->n2 + k) * *(myarg->B + k * myarg->n3 + j);

	return NULL;
}

void mult(double *A,double *B,double *C,int n1k,int n2k,int n3k)
{
	int numthreads;
	pthread_t *tids;
	struct threadArg * targs;

/*  
	for(int i=0;i<n1k;i++)
	for(int j=0;j<n3k;j++)
	for(int k=0;k<n2k;k++)
		*(C + i * n3k + j) += *(A + i * n2k + k) * *(B + k * n3k + j);		
*/
	numthreads = MIN(get_nprocs(),n1k);
	tids = (pthread_t*)malloc(numthreads * sizeof(pthread_t));

	targs = (struct threadArg *)malloc(numthreads*sizeof(struct threadArg));
	for(int i =0;i < numthreads;i++)
	{
		targs[i].tid	=i;
		targs[i].n1		= n1k;
		targs[i].n2		= n2k;
		targs[i].n3		= n3k;
		targs[i].A		=A;
		targs[i].B		=B;
		targs[i].C		=C;
		targs[i].numthreads=numthreads;
	}

	for(int i =0;i<numthreads;i++)// 创建线程
	{
		pthread_create(&tids[i],NULL,worker,&targs[i]);
	}
	for(int i =0;i<numthreads;i++) // 回收线程
	{
		pthread_join(tids[i],NULL);
	}
}

inline void swap(double * &a,double * &b)
{
	double *t;
	t = a;
	a = b;
	b = t;
}

int main(int argc,char* argv[])
{
	int id,numprocs;
	int temp,k,n1,n2,n3,idi,idj,n1k,n2k,n3k,start;
	double *A,*B,*C,*p,time,*tempA,*tempB;
  	FILE *fA,*fB,*fC;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&id);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

	time = MPI_Wtime();	

	k = (int) sqrt(numprocs);
	if(k*k != numprocs)	
	{
		printf("numprocs = %d, is not a square number!\n",numprocs);
    	MPI_Finalize();
		return 0;
	}

	if(!(fA = fopen("A_data", "r"))) {
		printf("Can't open A_data file\n");
    	MPI_Finalize();
		return 0;
 	}

	if(!(fB = fopen("B_data", "r"))) {
		printf("Can't open B_data file\n");
    	MPI_Finalize();
		return 0;
 	}

	if(!(fC = fopen("C_data", "w"))) {
		printf("Can't open C_data file\n");
    	MPI_Finalize();
		return 0;
 	}


	fread(&n1,sizeof(int),1,fA);
	fread(&n2,sizeof(int),1,fA);	
	fread(&temp,sizeof(int),1,fB);
	fread(&n3,sizeof(int),1,fB);
	if(n2 != temp)
	{
		printf("error input size\n");
		MPI_Finalize();
	}

	if(n1 % k || n2 % k || n3 % k)
	{
		printf("size must be div by k\n");
		MPI_Finalize();	
	}	
	n1k = n1 / k;
	n2k = n2 / k;
	n3k = n3 / k;
	if(id == 0)
	{
		printf("size = %d, %d, %d\n",n1,n2,n3);
		printf("core = %d*%d\n",k,k);
		printf("threads = %d\n\n",get_nprocs());
	}

	A = (double *) malloc(n1k * n2k * sizeof(double));
	tempA = (double *) malloc(n1k * n2k * sizeof(double));
	B = (double *) malloc(n2k * n3k * sizeof(double));
	tempB = (double *) malloc(n2k * n3k * sizeof(double));
	C = (double *) malloc(n1k * n3k * sizeof(double));

	idi = id / k;
	idj = id % k;
	my_read(fA,A,n1k,n2k,n2,idi,(idj + idi) % k);
	my_read(fB,B,n2k,n3k,n3,(idi + idj) % k,idj);

//	printf("id = %d A=%f\n",id,*A);

	if(!id)
	{
		printf("read time = %lf\n",MPI_Wtime()-time);	
		time = MPI_Wtime();
	}

	memset(C,0,n1k * n3k * sizeof(double));
	mult(A,B,C,n1k,n2k,n3k);
	MPI_Barrier(MPI_COMM_WORLD);
//	printf("id %d first mult done\n",id);
	for(int i=1;i<k;i++)
	{
	//	printf("id %d round %d\n",id,i);
		if(idj % 2)
		{
			MPI_Send(A,n1k*n2k,MPI_DOUBLE,get(id,k,LEFT),1,MPI_COMM_WORLD); 
			MPI_Recv(tempA,n1k*n2k,MPI_DOUBLE,get(id,k,RIGHT),1,MPI_COMM_WORLD,NULL);
		}
		else
		{
			MPI_Recv(tempA,n1k*n2k,MPI_DOUBLE,get(id,k,RIGHT),1,MPI_COMM_WORLD,NULL);
			MPI_Send(A,n1k*n2k,MPI_DOUBLE,get(id,k,LEFT),1,MPI_COMM_WORLD); 
		}

		if(idi % 2) 
		{
			MPI_Send(B,n2k*n3k,MPI_DOUBLE,get(id,k,UP),2,MPI_COMM_WORLD); 
			MPI_Recv(tempB,n2k*n3k,MPI_DOUBLE,get(id,k,DOWN),2,MPI_COMM_WORLD,NULL);
		}
		else
		{
			MPI_Recv(tempB,n2k*n3k,MPI_DOUBLE,get(id,k,DOWN),2,MPI_COMM_WORLD,NULL);
			MPI_Send(B,n2k*n3k,MPI_DOUBLE,get(id,k,UP),2,MPI_COMM_WORLD); 
		}
		swap(A,tempA);
		swap(B,tempB);
		mult(A,B,C,n1k,n2k,n3k);
	}
	if(!id)
	{
		printf("calc time = %lf\n",MPI_Wtime()-time);	
		time = MPI_Wtime();
	}
//	printf("id %d,n1=%d,n2=%d,n3=%d,k=%d,numprocs =%d\n",id,n1,n2,n3,k,numprocs);
	if(id==0)
	{
		fseek(fC,0,SEEK_SET);
		fwrite(&n1,sizeof(int),1,fC);
		fwrite(&n3,sizeof(int),1,fC);
		my_write(fC,C,n1k,n3k,n3,idi,idj);
		for(int i=1;i<numprocs;i++)
		{
		//	printf("wait %d\n",i);
			MPI_Recv(C,n1k*n3k,MPI_DOUBLE,i,3,MPI_COMM_WORLD,NULL);
		//	printf("recv from %d\n",i);
			my_write(fC,C,n1k,n3k,n3,i / k,i % k);
		}
		fclose(fC);
	}
	else
	{
//		printf("%d to \n",id);
		MPI_Send(C,n1k*n3k,MPI_DOUBLE,0,3,MPI_COMM_WORLD);	
//		printf("%d to done\n",id);
	}
	if(!id)
	{
		printf("write time = %lf\n",MPI_Wtime()-time);	
		time = MPI_Wtime();
	}
//	printf("id %d is all done\n",id);
    MPI_Finalize();
	return 0;
}
