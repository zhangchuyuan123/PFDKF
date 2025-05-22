#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES

#include <sndfile.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <fftw3.h>



#define M 256               // 帧长，可修改
#define N 8                 // 块数，可修改  数据分析的实际FFT长度为N*M 频率分辨率提高
#define A 0.999             // 超参数，小幅调整回声频域响应H以及误差协方差P
#define P_initial 1         // 迭代初值
#define Size (N - 1) * (M + 1) * sizeof(float) * 2 // 内存移动大小


struct PBFDKF
{
	float d_n[M];                        // 输入人声采样点
	float x_n[M];                        // 扬声器输出信号
	float e_n[M];                        // 误差信号
	float y_n[2*M];                      // 估计回声信号
	float x_arr[2 * M];                  // 扬声器缓存信号，长度为2帧，寓意前1帧信号经过反射后叠加到当前帧上
	float P_arr[N][M + 1];               // 协方差矩阵，N块M+1个数据
	float X_arr[N][M + 1][2];            // x_n的频域数据，N块M+1个复数
	float H_arr[N][M + 1][2];            // echo响应h的频域数据，N块M+1个复数
	float e_fft2[2 * M];                 // 误差e_n前端补零得到
	int p;                               // 块更新索引，[0,N-1]
};
typedef struct PBFDKF pfdkf;

// 生成汉宁窗
void hanning_window(float* window, int length)
{
	if (window == NULL || length <= 0)
		return;

	for (int n = 0; n < length; n++) {
		window[n] = 0.5 * (1 - cos(2 * M_PI * n / (length - 1)));
	}
}

// 初始化P_arr
void Initial_P(pfdkf* S)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M+1; j++)
			S->P_arr[i][j] = P_initial;
}

// 复数乘法，1.以N维为基准求和（降维）  2.频域分块的叠加计算方法
static void ComplexMultSum(pfdkf* S,float Y[M+1][2])
{
	for (int m = 0; m < M + 1; m++)
	{
		float Y_i[2] = { 0 };
		for (int n = 0; n < N; n++)
		{
			Y_i[0] += S->H_arr[n][m][0] * S->X_arr[n][m][0] - S->H_arr[n][m][1] * S->X_arr[n][m][1];
			Y_i[1] += S->H_arr[n][m][0] * S->X_arr[n][m][1] + S->H_arr[n][m][1] * S->X_arr[n][m][0];
		}
		Y[m][0] = Y_i[0];
		Y[m][1] = Y_i[1];
	}
}

// 卡尔曼滤波
static void Filt(pfdkf* S, fftwf_complex* X, fftwf_plan plan1, fftwf_plan plan2)
{
	memcpy(S->x_arr+M, S->x_n, sizeof(S->x_n));
	fftwf_execute(plan1);
	memmove(S->X_arr+1, S->X_arr, Size); //把X_arr以第一维为基准，将X_arr[0:N-2][i][j]移动至X_arr[1:N-1][i][j]
	memcpy(S->X_arr, X, sizeof(float)*2*(M+1));  // 将FFT的输出X copy至X_arr[0][i][j]
	memcpy(S->x_arr, S->x_arr + M, M * sizeof(float)); // 将x_arr的fifo左移M个数据
	// 计算H_arr与X_arr的逐元素乘积，然后按块（以第一维为基准）做逐元素加法，获得Y，需要定义一个Y来接收计算结果
	// 计算Y时需注意，乘法和加法均为复数乘加运算
	float Y_sum[M + 1][2] = {0};
	ComplexMultSum(S,Y_sum);
	// 对Y进行IFFT，截取后M个数据
	memcpy(X, Y_sum, sizeof(float)*2*(M+1));
	fftwf_execute(plan2);
	float yn_truc[M] = { 0 };
	memcpy(yn_truc, S->y_n + M, M * sizeof(float));
	// e = d - y , d为结构体S中的d_n
	// 更新了结构体S中的数据，无需返回特定值
	for (int i = 0; i < M; i++)
	{
		S->e_n[i] = S->d_n[i] - yn_truc[i]/2/M; // IFFT后的结果需要归一化
	}
	
}

// 计算N块信号的总能量
static void abs2sum(float X_arr[][M+1][2], float* X)
{
	for (int m = 0; m < M+1; m++)
	{
		float tmp = 0;
		for (int n=0;n<N;n++)
		{
			tmp += X_arr[n][m][0] * X_arr[n][m][0] + X_arr[n][m][1] * X_arr[n][m][1];
		}
		X[m] = tmp;
	}
}

// 卡尔曼滤波参数更新
static void Updata(pfdkf* S, float* window, fftwf_complex* E, float h[], fftwf_complex* H, fftwf_plan plan1, fftwf_plan plan2, fftwf_plan plan3)
{
	float e_fft1[M] = { 0 };
	for (int i = 0; i < M; i++)
	{
		e_fft1[i] = S->e_n[i] * window[i];
	}
	memcpy(S->e_fft2 + M, e_fft1, sizeof(e_fft1));
	fftwf_execute(plan1);
	float X2[M + 1] = { 0 };
	abs2sum(S->X_arr,X2);
	float H2 = 0;
	float E2[M + 1] = { 0 };
	float Pe = 0;
	float mu = { 0 };
	float K[2] = { 0 };
	for (int n = 0; n < N; n++)
	{
		for (int m = 0; m < M + 1; m++)
		{
			if (n == 0)
			{
				H2 = (S->H_arr[n][m][0] * S->H_arr[n][m][0] + S->H_arr[n][m][1] * S->H_arr[n][m][1])*(1-A*A);
				E2[m] = (E[m][0] * E[m][0] + E[m][1] * E[m][1]) / N;
				Pe = S->P_arr[n][m] * X2[m] * 0.5 + E2[m];
				mu = S->P_arr[n][m] / (Pe + 1e-10);
				S->P_arr[n][m] = A * A * (1 - 0.5 * mu * X2[m]) * S->P_arr[n][m] + H2;
				K[0] = mu * S->X_arr[n][m][0];
				K[1] = mu * (- S->X_arr[n][m][1]);
				S->H_arr[n][m][0] = (E[m][0] * K[0] - E[m][1] * K[1]) + S->H_arr[n][m][0];
				S->H_arr[n][m][1] = (E[m][0] * K[1] + E[m][1] * K[0]) + S->H_arr[n][m][1];
			}
			else
			{
				H2 = (S->H_arr[n][m][0] * S->H_arr[n][m][0] + S->H_arr[n][m][1] * S->H_arr[n][m][1]) * (1 - A * A);
				Pe = S->P_arr[n][m] * X2[m] * 0.5 + E2[m];
				mu = S->P_arr[n][m] / (Pe + 1e-10);
				S->P_arr[n][m] = A * A * (1 - 0.5 * mu * X2[m]) * S->P_arr[n][m] + H2;
				K[0] = mu * S->X_arr[n][m][0];
				K[1] = mu * (-S->X_arr[n][m][1]);
				S->H_arr[n][m][0] = (E[m][0] * K[0] - E[m][1] * K[1]) + S->H_arr[n][m][0];
				S->H_arr[n][m][1] = (E[m][0] * K[1] + E[m][1] * K[0]) + S->H_arr[n][m][1];
			}
		}
	}
	memcpy(H, S->H_arr[S->p], sizeof(float) * 2 * (M + 1));
	fftwf_execute(plan3);
	// 归一化
	for (int i = 0; i < M; i++)
	{
		h[i] /= 2 * M;
	}
	memset(h + M, 0, sizeof(float)*M);
	fftwf_execute(plan2);
	memcpy(S->H_arr[S->p], H, sizeof(float) * 2 * (M + 1));
	S->p = (S->p + 1) % N;
}

int main()
{
	// 读啸叫音频
	SF_INFO sf_info = { 0 };
	SNDFILE* snd_file;

	snd_file = sf_open("HowlGen2.wav", SFM_READ, &sf_info);

	if (!snd_file) {
		printf("Error: %s\n", sf_strerror(NULL));
		return 1;
	}

	float* audio_data = (float*)malloc(sf_info.frames * sizeof(float));
	sf_count_t read_count = sf_read_float(snd_file, audio_data, sf_info.frames);

	//hanning窗生成函数
	float window[M] = { 0 };
	hanning_window(window, M);

	// 初始化
	pfdkf datbuf = { 0 };
	Initial_P(&datbuf);

	// 计算数据能完整截多少段
	int num_block = sf_info.frames / M;

	// 初始化数据存放数组
	float* dout = (float*)malloc(num_block * M * sizeof(float));

	// 创建FFT\IFFT PLAN
	fftwf_complex* X = fftwf_alloc_complex(M + 1);
	fftwf_plan fft_X = fftwf_plan_dft_r2c_1d(2*M, datbuf.x_arr, X, FFTW_ESTIMATE);
	fftwf_plan ifft_X = fftwf_plan_dft_c2r_1d(2*M, X, datbuf.y_n, FFTW_ESTIMATE);

	fftwf_complex* E = fftwf_alloc_complex(M + 1);
	fftwf_plan fft_E = fftwf_plan_dft_r2c_1d(2 * M, datbuf.e_fft2, E, FFTW_ESTIMATE);

	float h[2 * M] = { 0 };
	fftwf_complex* H = fftwf_alloc_complex(M + 1);
	fftwf_plan fft_H = fftwf_plan_dft_r2c_1d(2 * M, h, H, FFTW_ESTIMATE);
	fftwf_plan ifft_H = fftwf_plan_dft_c2r_1d(2 * M, H, h, FFTW_ESTIMATE);

	// 数据处理主循环
	int i = 0;
	for (i = 0; i < num_block; i++)
	{
		if (i == 0)
		{
			memcpy(datbuf.d_n, audio_data + i * M, M * sizeof(float));
			// 更新扬声器x_arr中数据，因为开始时x_arr无输出，所以i为0时可跳过
			Filt(&datbuf, X, fft_X, ifft_X);
			Updata(&datbuf,window,E,h,H,fft_E,fft_H,ifft_H);
			memcpy(dout,datbuf.e_n,sizeof(datbuf.e_n));
		}
		else
		{
			memcpy(datbuf.d_n, audio_data + i * M, M * sizeof(float));
			memcpy(datbuf.x_n, audio_data + (i - 1) * M, M * sizeof(float));
			Filt(&datbuf, X, fft_X, ifft_X);
			Updata(&datbuf, window, E, h, H, fft_E, fft_H, ifft_H);
			memcpy(dout+i*M, datbuf.e_n, sizeof(datbuf.e_n));
		}
	}

	// 写处理后数据
	SF_INFO sf_info_w = { 0 };
	sf_info_w.samplerate = sf_info.samplerate;
	sf_info_w.channels = sf_info.channels;
	sf_info_w.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

	SNDFILE* file2 = sf_open("DeHowl_PBFDKF.wav", SFM_WRITE, &sf_info_w);
	if (!file2)
	{
		printf("Error: % s\n", sf_strerror(NULL));
		return 1;
	}

	sf_write_float(file2, dout, num_block*M);

	// 清理内存空间
	sf_close(snd_file);
	sf_close(file2);
	free(audio_data);
	fftwf_destroy_plan(fft_X);
	fftwf_destroy_plan(ifft_X);
	fftwf_destroy_plan(fft_E);
	fftwf_destroy_plan(fft_H);
	fftwf_destroy_plan(ifft_H);
	fftwf_free(X);
	fftwf_free(E);
	fftwf_free(H);
	free(dout);

	return 0;
}