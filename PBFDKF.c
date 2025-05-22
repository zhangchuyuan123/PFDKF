#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES

#include <sndfile.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <fftw3.h>



#define M 256               // ֡�������޸�
#define N 8                 // ���������޸�  ���ݷ�����ʵ��FFT����ΪN*M Ƶ�ʷֱ������
#define A 0.999             // ��������С����������Ƶ����ӦH�Լ����Э����P
#define P_initial 1         // ������ֵ
#define Size (N - 1) * (M + 1) * sizeof(float) * 2 // �ڴ��ƶ���С


struct PBFDKF
{
	float d_n[M];                        // ��������������
	float x_n[M];                        // ����������ź�
	float e_n[M];                        // ����ź�
	float y_n[2*M];                      // ���ƻ����ź�
	float x_arr[2 * M];                  // �����������źţ�����Ϊ2֡��Ԣ��ǰ1֡�źž����������ӵ���ǰ֡��
	float P_arr[N][M + 1];               // Э�������N��M+1������
	float X_arr[N][M + 1][2];            // x_n��Ƶ�����ݣ�N��M+1������
	float H_arr[N][M + 1][2];            // echo��Ӧh��Ƶ�����ݣ�N��M+1������
	float e_fft2[2 * M];                 // ���e_nǰ�˲���õ�
	int p;                               // �����������[0,N-1]
};
typedef struct PBFDKF pfdkf;

// ���ɺ�����
void hanning_window(float* window, int length)
{
	if (window == NULL || length <= 0)
		return;

	for (int n = 0; n < length; n++) {
		window[n] = 0.5 * (1 - cos(2 * M_PI * n / (length - 1)));
	}
}

// ��ʼ��P_arr
void Initial_P(pfdkf* S)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M+1; j++)
			S->P_arr[i][j] = P_initial;
}

// �����˷���1.��NάΪ��׼��ͣ���ά��  2.Ƶ��ֿ�ĵ��Ӽ��㷽��
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

// �������˲�
static void Filt(pfdkf* S, fftwf_complex* X, fftwf_plan plan1, fftwf_plan plan2)
{
	memcpy(S->x_arr+M, S->x_n, sizeof(S->x_n));
	fftwf_execute(plan1);
	memmove(S->X_arr+1, S->X_arr, Size); //��X_arr�Ե�һάΪ��׼����X_arr[0:N-2][i][j]�ƶ���X_arr[1:N-1][i][j]
	memcpy(S->X_arr, X, sizeof(float)*2*(M+1));  // ��FFT�����X copy��X_arr[0][i][j]
	memcpy(S->x_arr, S->x_arr + M, M * sizeof(float)); // ��x_arr��fifo����M������
	// ����H_arr��X_arr����Ԫ�س˻���Ȼ�󰴿飨�Ե�һάΪ��׼������Ԫ�ؼӷ������Y����Ҫ����һ��Y�����ռ�����
	// ����Yʱ��ע�⣬�˷��ͼӷ���Ϊ�����˼�����
	float Y_sum[M + 1][2] = {0};
	ComplexMultSum(S,Y_sum);
	// ��Y����IFFT����ȡ��M������
	memcpy(X, Y_sum, sizeof(float)*2*(M+1));
	fftwf_execute(plan2);
	float yn_truc[M] = { 0 };
	memcpy(yn_truc, S->y_n + M, M * sizeof(float));
	// e = d - y , dΪ�ṹ��S�е�d_n
	// �����˽ṹ��S�е����ݣ����践���ض�ֵ
	for (int i = 0; i < M; i++)
	{
		S->e_n[i] = S->d_n[i] - yn_truc[i]/2/M; // IFFT��Ľ����Ҫ��һ��
	}
	
}

// ����N���źŵ�������
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

// �������˲���������
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
	// ��һ��
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
	// ��Х����Ƶ
	SF_INFO sf_info = { 0 };
	SNDFILE* snd_file;

	snd_file = sf_open("HowlGen2.wav", SFM_READ, &sf_info);

	if (!snd_file) {
		printf("Error: %s\n", sf_strerror(NULL));
		return 1;
	}

	float* audio_data = (float*)malloc(sf_info.frames * sizeof(float));
	sf_count_t read_count = sf_read_float(snd_file, audio_data, sf_info.frames);

	//hanning�����ɺ���
	float window[M] = { 0 };
	hanning_window(window, M);

	// ��ʼ��
	pfdkf datbuf = { 0 };
	Initial_P(&datbuf);

	// ���������������ض��ٶ�
	int num_block = sf_info.frames / M;

	// ��ʼ�����ݴ������
	float* dout = (float*)malloc(num_block * M * sizeof(float));

	// ����FFT\IFFT PLAN
	fftwf_complex* X = fftwf_alloc_complex(M + 1);
	fftwf_plan fft_X = fftwf_plan_dft_r2c_1d(2*M, datbuf.x_arr, X, FFTW_ESTIMATE);
	fftwf_plan ifft_X = fftwf_plan_dft_c2r_1d(2*M, X, datbuf.y_n, FFTW_ESTIMATE);

	fftwf_complex* E = fftwf_alloc_complex(M + 1);
	fftwf_plan fft_E = fftwf_plan_dft_r2c_1d(2 * M, datbuf.e_fft2, E, FFTW_ESTIMATE);

	float h[2 * M] = { 0 };
	fftwf_complex* H = fftwf_alloc_complex(M + 1);
	fftwf_plan fft_H = fftwf_plan_dft_r2c_1d(2 * M, h, H, FFTW_ESTIMATE);
	fftwf_plan ifft_H = fftwf_plan_dft_c2r_1d(2 * M, H, h, FFTW_ESTIMATE);

	// ���ݴ�����ѭ��
	int i = 0;
	for (i = 0; i < num_block; i++)
	{
		if (i == 0)
		{
			memcpy(datbuf.d_n, audio_data + i * M, M * sizeof(float));
			// ����������x_arr�����ݣ���Ϊ��ʼʱx_arr�����������iΪ0ʱ������
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

	// д���������
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

	// �����ڴ�ռ�
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