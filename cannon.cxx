#include <stdio.h>
#include <sys/sysinfo.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <mpi.h>
#include <math.h>

// 定义方向代号
#define UP 0
#define RIGHT 1
#define DOWN 2
#define LEFT 3

#define MIN(a,b) (((a)>(b))? (b):(a))

struct threadArg {  // 定义线程参数结构
	int tid,n1,n2,n3,numthreads;
	double *B,*A,*C;
};

void my_read(FILE *f,double *p,int time,int size,int offset,int idi,int idj)// 矩阵分块读入
{
	int start = sizeof(int) * 2 + (idi * offset * size + idj * size)* sizeof(double);// 计算文件偏移量
///	printf("%d ,%d %d\n",start,idi,idj);
	for(int i=0;i<time;i++)
	{
		fseek(f,start,SEEK_SET); // 设置文件偏移量
		fread(p,sizeof(double),size,f); // 读取size大小的块
		start += offset * sizeof(double); // 文件跳入下一行
		p += size; // 写入指针增加偏移
	}
}

void my_write(FILE *f,double *p,int time,int size,int offset,int idi,int idj)// 矩阵分块写入
{
	int start = sizeof(int) * 2 + (idi * offset * size + idj * size)* sizeof(double); // 同上，计算初始文件偏移
//	printf("%d C\n",start);
	for(int i=0;i<time;i++)
	{
		fseek(f,start,SEEK_SET);// 设置文件偏移
		fwrite(p,sizeof(double),size,f);// 写入size块大小的数据
		start += offset * sizeof(double);// 文件跳入下一行
		p += size; // 数据指针增加偏移
	}
}

void output(double *p,int n,int m) // 辅助的输出查看函数
{
	for(int i= 0;i<n;i++)
	{
		for(int j= 0;j<m;j++)
			printf("%f\t",*(p + i*m + j));
		printf("\n");
	}
}

int get(int id,int k,int pos) // 获取标号i的上下左右标号是多少
{
	int x,y;
	x = id / k; // 获得行
	y = id % k; // 获得列
	switch (pos) 
	{
		case UP:
				x -= 1; // 如果是上，则让行减一
				break;
		case RIGHT:
				y += 1; // 如果是右，则让列加一
				break; 
		case DOWN:
				x += 1; // 如果是下，则让行加一
				break;
		case LEFT:
				y -= 1; // 如果是左，则让列减一
				break;	
	}
	x = (x + k) % k; // 补足负数偏移
	y = (y + k) % k;
	return x * k + y; // 返回结果
}

void* worker(void * arg) // 矩阵乘法线程函数
{
	struct threadArg* myarg = (struct threadArg *)arg;

	for(int i= myarg->tid;i < myarg->n1; i+=myarg->numthreads)  // 使用ikj顺序，减小cache缺失率
	for(int k= 0;k < myarg->n2;k++)
	for(int j= 0;j < myarg->n3;j++)
	*(myarg->C + i*myarg->n3 + j) += *(myarg->A + i*myarg->n2 + k) * *(myarg->B + k * myarg->n3 + j);

	return NULL;
}

void naive_mult(double *A,double *B,double *C,int n1k,int n2k,int n3k)// 普通矩阵乘法
{
	for(int i=0;i<n1k;i++)
	for(int j=0;j<n3k;j++)
	for(int k=0;k<n2k;k++)
		*(C + i * n3k + j) += *(A + i * n2k + k) * *(B + k * n3k + j);		
}

void thread_mult(double *A,double *B,double *C,int n1k,int n2k,int n3k)// 构造多线程矩阵乘法
{
	int numthreads;
	pthread_t *tids;
	struct threadArg * targs;

	numthreads = MIN(get_nprocs(),n1k); // 计算使用的线程个数
	tids = (pthread_t*)malloc(numthreads * sizeof(pthread_t)); // 新建线程数组

	targs = (struct threadArg *)malloc(numthreads*sizeof(struct threadArg)); // 新建线程参数数组
	for(int i =0;i < numthreads;i++) // 对线程的参数进行赋值
	{
		targs[i].tid	=i; // 表示线程标号
		targs[i].n1		= n1k;
		targs[i].n2		= n2k;
		targs[i].n3		= n3k;
		targs[i].A		=A;
		targs[i].B		=B;
		targs[i].C		=C;
		targs[i].numthreads=numthreads;
	}

	for(int i =0;i<numthreads;i++) // 创建线程
	{
		pthread_create(&tids[i],NULL,worker,&targs[i]);
	}
	for(int i =0;i<numthreads;i++)  // 回收计算线程
	{
		pthread_join(tids[i],NULL);
	}
}

inline void swap(double * &a,double * &b)// 交换两个指针
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
	double *A,*B,*C,*p,time,*tempA,*tempB,tt;
  	FILE *fA,*fB,*fC;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&id); // 获得当前节点号
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs); // 获得总节点号

	time = MPI_Wtime();	 // 计时桩
	tt = MPI_Wtime();

	if(!(fA = fopen("A_data", "r"))) { // 打开A矩阵文件
		printf("Can't open A_data file\n");
    	MPI_Finalize();
		return 0;
 	}

	if(!(fB = fopen("B_data", "r"))) { // 打开B矩阵文件
		printf("Can't open B_data file\n");
    	MPI_Finalize();
		return 0;
 	}

	if(!(fC = fopen("C_data", "w"))) { // 打开C矩阵文件
		printf("Can't open C_data file\n");
    	MPI_Finalize();
		return 0;
 	}


	fread(&n1,sizeof(int),1,fA);// 读入A矩阵的行列
	fread(&n2,sizeof(int),1,fA);	
	fread(&temp,sizeof(int),1,fB); // 读入B矩阵的行列
	fread(&n3,sizeof(int),1,fB);
	if(n2 != temp) // 若A矩阵的列和B矩阵的行不等，则报错
	{
		printf("error input size\n");
		MPI_Finalize();
	}


	k = (int) sqrt(numprocs); // 取得当前适宜的k
	while (n1 % k || n2 % k || n3 % k) // 如果矩阵的行列不被整除，则向下寻找
		k -= 1;
	numprocs = k*k; //重新计算当前使用的核数

	if(id >= numprocs)	 // 不使用的核退出
	{
    	MPI_Finalize();
		return 0;
	}

	n1k = n1 / k;
	n2k = n2 / k;
	n3k = n3 / k; // 计算每一块的大小
	if( id == 0) // 输出信息
	{ 
		printf("size = %d, %d, %d\n",n1,n2,n3);
		printf("core = %d*%d\n",k,k);
		printf("threads = %d\n\n",MIN(n1,get_nprocs()));
	}

	A = (double *) malloc(n1k * n2k * sizeof(double)); // 申请空间
	tempA = (double *) malloc(n1k * n2k * sizeof(double)); // 申请临时的A空间
	B = (double *) malloc(n2k * n3k * sizeof(double));
	tempB = (double *) malloc(n2k * n3k * sizeof(double)); // 申请临时的B空间
	C = (double *) malloc(n1k * n3k * sizeof(double));

	idi = id / k;
	idj = id % k;// 计算当前节点在第几行第几列的块
	my_read(fA,A,n1k,n2k,n2,idi,(idj + idi) % k); // 读入当前所需的A块
	my_read(fB,B,n2k,n3k,n3,(idi + idj) % k,idj); // 读入B块

//	printf("id = %d A=%f\n",id,*A);

	if(!id)
	{
		printf("read time = %lf\n",MPI_Wtime()-time);	 // 输出读取所花费的时间
		time = MPI_Wtime();
	}

	memset(C,0,n1k * n3k * sizeof(double));// 赋初值
	thread_mult(A,B,C,n1k,n2k,n3k); //第一次矩阵乘法
	MPI_Barrier(MPI_COMM_WORLD);
//	printf("id %d first mult done\n",id);
	for(int i=1;i<k;i++) // 一共需要交换k-1次
	{
	//	printf("id %d round %d\n",id,i);
		if(idj % 2) // 交叉发送接受A数组
		{
			MPI_Send(A,n1k*n2k,MPI_DOUBLE,get(id,k,LEFT),1,MPI_COMM_WORLD); 
			MPI_Recv(tempA,n1k*n2k,MPI_DOUBLE,get(id,k,RIGHT),1,MPI_COMM_WORLD,NULL);
		}
		else
		{
			MPI_Recv(tempA,n1k*n2k,MPI_DOUBLE,get(id,k,RIGHT),1,MPI_COMM_WORLD,NULL);
			MPI_Send(A,n1k*n2k,MPI_DOUBLE,get(id,k,LEFT),1,MPI_COMM_WORLD); 
		}

		if(idi % 2)  // 交替发送和接受B数组
		{
			MPI_Send(B,n2k*n3k,MPI_DOUBLE,get(id,k,UP),2,MPI_COMM_WORLD); 
			MPI_Recv(tempB,n2k*n3k,MPI_DOUBLE,get(id,k,DOWN),2,MPI_COMM_WORLD,NULL);
		}
		else
		{
			MPI_Recv(tempB,n2k*n3k,MPI_DOUBLE,get(id,k,DOWN),2,MPI_COMM_WORLD,NULL);
			MPI_Send(B,n2k*n3k,MPI_DOUBLE,get(id,k,UP),2,MPI_COMM_WORLD); 
		}
		swap(A,tempA); // 交换数组
		swap(B,tempB);
		thread_mult(A,B,C,n1k,n2k,n3k) ; //进行乘法
	}
	if(!id) // 输出计算所用的时间
	{
		printf("calc time = %lf\n",MPI_Wtime()-time);	
		time = MPI_Wtime();
	}
//	printf("id %d,n1=%d,n2=%d,n3=%d,k=%d,numprocs =%d\n",id,n1,n2,n3,k,numprocs);
	if(id==0) // 节点0回收结果
	{ 
		fseek(fC,0,SEEK_SET);
		fwrite(&n1,sizeof(int),1,fC);
		fwrite(&n3,sizeof(int),1,fC);
		my_write(fC,C,n1k,n3k,n3,idi,idj); // 首先写入自己的结果
		for(int i=1;i<numprocs;i++) // 从1到numprocs-1接受结果
		{
		//	printf("wait %d\n",i);
			MPI_Recv(C,n1k*n3k,MPI_DOUBLE,i,3,MPI_COMM_WORLD,NULL);
		//	printf("recv from %d\n",i);
			my_write(fC,C,n1k,n3k,n3,i / k,i % k); // 写入结果文件
		}
		fclose(fC);
	}
	else
	{
//		printf("%d to \n",id);
		MPI_Send(C,n1k*n3k,MPI_DOUBLE,0,3,MPI_COMM_WORLD);	 //往节点0发送自己的结果
//		printf("%d to done\n",id);
	}
	if(!id)
	{
		printf("write time = %lf\n",MPI_Wtime()-time);	 // 输出写入时间
		printf("total time = %lf\n",MPI_Wtime()-tt);    // 输出总时间
	}
//	printf("id %d is all done\n",id);
    MPI_Finalize();  //结束
	return 0;
}
