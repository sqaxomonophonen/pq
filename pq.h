#ifndef PQ_H

// config
#define PQ_CLOCK_HZ           (256000)
#define PQ_FIR_N              (27)
#define PQ_BUFFER_SIZE_LOG2   (8)
#define PQ_N_CHANNELS_LOG2    (1)
#define PQ_N_CHUNKS_LOG2      (1)

#include <math.h>
#include <string.h>
#include <stdint.h>

#define PQ_TAU                     (6.283185307179586)

#define PQ_N_CHANNELS              (1<<PQ_N_CHANNELS_LOG2)
#define PQ_BUFFER_SIZE             (1<<PQ_BUFFER_SIZE_LOG2)
#define PQ_BUFFER_MASK             (PQ_BUFFER_SIZE-1)
#define PQ_N_CHUNK_FRAMES_LOG2     (PQ_BUFFER_SIZE_LOG2 - PQ_N_CHUNKS_LOG2)
#define PQ_N_CHUNK_FRAMES          (1<<PQ_N_CHUNK_FRAMES_LOG2)
#define PQ_N_CHUNK_SAMPLES_LOG2    (PQ_N_CHUNK_FRAMES_LOG2 + PQ_N_CHANNELS_LOG2)
#define PQ_N_CHUNK_SAMPLES         (1<<PQ_N_CHUNK_SAMPLES_LOG2)

enum {
	PQ__LAYER_RAW = 0,
	PQ__LAYER_BANDLIMITED,
	PQ__LAYER_MAX,
};

struct pq {
	float buffer[PQ__LAYER_MAX][PQ_BUFFER_SIZE];
	float brick_fir[PQ_FIR_N];

	int n_raw_chunks_done;
	int n_bandlimited_chunks_done;

	int sample_rate;

	int x, x_acc;

	float xyzzy_step;
	float xyzzy;
};

static inline float bspline_4p_3o(float x, float A, float B, float C, float D)
{
	const float AC = A+C;
	const float S = 1.0f/6.0f;
	const float H = S*3;
	return (((H*(B-C)+S*(D-A))*x+(H*AC-B))*x+(H*(C-A)))*x+(S*AC+S*4*B);
}

void pq_init(struct pq* pq, int sample_rate)
{
	memset(pq, 0, sizeof *pq);

	pq->sample_rate = sample_rate;

	{ // make brickwall FIR filter: one part of the arbitrary resampler
		const int N_HALF = PQ_FIR_N / 2;
		const float xstep = (float)sample_rate / (float)PQ_CLOCK_HZ;
		float x = (float)-N_HALF * xstep;
		for (int i = 0; i <= N_HALF; i++) {
			const float y = (i == N_HALF) ? (1.0f) : (sinf(x)/x);
			const float wt = ((float)PQ_TAU * (float)i) / (float)(PQ_FIR_N-1);
			const float w = 0.355768f - 0.487396f*cosf(wt) + 0.144232f*cosf(2*wt) - 0.012604f*cosf(3.0f*wt); // Nuttall
			const float z = y*w;
			#if 0
			printf("fir[%d,%d]=%f\n", i, PQ_FIR_N-1-i, z);
			#endif
			pq->brick_fir[i] = z;
			pq->brick_fir[PQ_FIR_N-1-i] = z;
			x += xstep;
		}
	}
}

static inline void pq__step(struct pq* pq, float* out)
{
	float x = sinf(pq->xyzzy) * 0.1;
	for (int i = 0; i < PQ_N_CHANNELS; i++) out[i] = x;
	pq->xyzzy += pq->xyzzy_step;
	while (pq->xyzzy > PQ_TAU) pq->xyzzy -= PQ_TAU;
}

static void pq__freewheel(struct pq* pq, float* out, int n_frames)
{
	const int sample_rate = pq->sample_rate;
	int x_acc = pq->x_acc;
	int x = pq->x;
	float* outp = out;
	for (int i0 = 0; i0 < n_frames; i0++) {
		const int x_req = x+3;
		const int n_bandlimited_chunks_req = 1 + (x_req >> PQ_N_CHUNK_FRAMES_LOG2);
		const int n_raw_chunks_req = n_bandlimited_chunks_req + 1;

		// raw output
		while (n_raw_chunks_req > pq->n_raw_chunks_done) {
			float* wp = &pq->buffer[PQ__LAYER_RAW][(pq->n_raw_chunks_done << PQ_N_CHUNK_SAMPLES_LOG2) & PQ_BUFFER_MASK];
			for (int i1 = 0; i1 < PQ_N_CHUNK_FRAMES; i1++) {
				pq__step(pq, wp);
				wp += PQ_N_CHANNELS;
			}
			pq->n_raw_chunks_done++;
		}

		// apply FIR brickwall filter
		while (n_bandlimited_chunks_req > pq->n_bandlimited_chunks_done) {
			float* read_buffer = pq->buffer[PQ__LAYER_RAW];
			float* wp = &pq->buffer[PQ__LAYER_BANDLIMITED][(pq->n_bandlimited_chunks_done << PQ_N_CHUNK_SAMPLES_LOG2) & PQ_BUFFER_MASK];
			int bx = pq->n_bandlimited_chunks_done << PQ_N_CHUNK_FRAMES_LOG2;
			for (int i1 = 0; i1 < PQ_N_CHUNK_FRAMES; i1++, bx++) {
				float acc[PQ_N_CHANNELS] = {0};
				for (int fi = 0; fi < PQ_FIR_N; fi++) {
					const float fir = pq->brick_fir[fi];
					for (int channel_index = 0; channel_index < PQ_N_CHANNELS; channel_index++) {
						const int bi = (((bx+fi) << PQ_N_CHANNELS_LOG2) + channel_index) & PQ_BUFFER_MASK;
						acc[channel_index] += fir * read_buffer[bi];
					}
				}
				for (int channel_index = 0; channel_index < PQ_N_CHANNELS; channel_index++) {
					*(wp++) = acc[channel_index];
				}
			}
			pq->n_bandlimited_chunks_done++;
		}

		// interpolate
		const float xf = (float)x_acc * (1.0f / (float)PQ_CLOCK_HZ);
		for (int channel_index = 0; channel_index < PQ_N_CHANNELS; channel_index++) {
			float y[4];
			for (int yi = 0; yi < 4; yi++) {
				y[yi] = pq->buffer[PQ__LAYER_BANDLIMITED][(((x+yi) << PQ_N_CHANNELS_LOG2)+channel_index) & PQ_BUFFER_MASK];
			}
			*(outp++) = bspline_4p_3o(xf, y[0], y[1], y[2], y[3]);
		}

		// advance
		x_acc += PQ_CLOCK_HZ;
		x += x_acc / sample_rate;
		x_acc %= sample_rate;
	}

	pq->x_acc = x_acc;
	pq->x = x;
}

float xxxxxxxx = 100.0; // XXX
void pq_run(struct pq* pq, float* out, int n_frames)
{
	pq->xyzzy_step = ((PQ_TAU*xxxxxxxx) / (float)PQ_CLOCK_HZ);
	xxxxxxxx += 1.0f;
	int n_frames_remaining = n_frames;
	while (n_frames_remaining > 0) {
		const int n = n_frames; // XXX TODO
		pq__freewheel(pq, out, n);
		n_frames_remaining -= n;
	}
}

#define PQ_H
#endif
