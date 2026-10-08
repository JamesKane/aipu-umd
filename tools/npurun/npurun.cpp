/*
 * SPDX-License-Identifier: BSD-2-Clause
 *
 * Run a compiled Zhouyi graph on the NPU through libaipudrv's standard API:
 * each input image (float32, the graph's input layout, preprocessed) is
 * quantized with the scale and zero point the graph reports, run as a job,
 * and the output dequantized and ranked.
 *
 * Usage: npurun graph.elf labels.txt image.f32 ...
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "standard_api.h"

static const aipu_ctx_handle_t *ctx;

static void
check(aipu_status_t st, const char *what)
{
	const char *msg = nullptr;

	if (st == AIPU_STATUS_SUCCESS)
		return;
	aipu_get_error_message(ctx, st, &msg);
	fprintf(stderr, "%s: %s (%d)\n", what, msg ? msg : "?", (int)st);
	exit(1);
}

static std::vector<std::string>
read_labels(const char *path)
{
	std::vector<std::string> labels(1000);
	std::ifstream f(path);
	std::string line;

	/* Python dict lines: "N: 'name, other names',". */
	while (std::getline(f, line)) {
		size_t c = line.find(':'), q1 = line.find('\''),
		    q2 = line.rfind('\'');
		if (c == std::string::npos || q1 == std::string::npos ||
		    q2 <= q1)
			continue;
		int n = atoi(line.c_str() + (line[0] == '{' ? 1 : 0));
		if (n >= 0 && n < 1000)
			labels[n] = line.substr(q1 + 1, q2 - q1 - 1);
	}
	return labels;
}

int
main(int argc, char **argv)
{
	aipu_ctx_handle_t *c;
	aipu_tensor_desc_t in, out;
	uint64_t graph, job;
	uint32_t nin, nout, i;

	if (argc < 4) {
		fprintf(stderr, "usage: npurun graph.elf labels.txt image.f32 ...\n");
		return 2;
	}
	std::vector<std::string> labels = read_labels(argv[2]);

	check(aipu_init_context(&c), "init_context");
	ctx = c;
	check(aipu_load_graph(ctx, argv[1], &graph), "load_graph");
	check(aipu_get_tensor_count(ctx, graph, AIPU_TENSOR_TYPE_INPUT, &nin),
	    "input count");
	check(aipu_get_tensor_count(ctx, graph, AIPU_TENSOR_TYPE_OUTPUT, &nout),
	    "output count");
	check(aipu_get_tensor_descriptor(ctx, graph, AIPU_TENSOR_TYPE_INPUT, 0,
	    &in), "input descriptor");
	check(aipu_get_tensor_descriptor(ctx, graph, AIPU_TENSOR_TYPE_OUTPUT, 0,
	    &out), "output descriptor");
	printf("graph loaded: %u input(s), %u output(s)\n", nin, nout);
	printf("input 0: %u bytes, type %d, scale %g, zero point %d\n", in.size,
	    (int)in.data_type, in.scale, in.zero_point);
	printf("output 0: %u bytes, type %d, scale %g, zero point %d\n", out.size,
	    (int)out.data_type, out.scale, out.zero_point);
	if ((in.data_type != AIPU_DATA_TYPE_S8 &&
	    in.data_type != AIPU_DATA_TYPE_U8) ||
	    (out.data_type != AIPU_DATA_TYPE_S8 &&
	    out.data_type != AIPU_DATA_TYPE_U8)) {
		fprintf(stderr, "only 8-bit tensors handled\n");
		return 1;
	}
	check(aipu_create_job(ctx, graph, &job), "create_job");

	for (int a = 3; a < argc; a++) {
		std::vector<float> x(in.size);
		std::vector<uint8_t> q(in.size), y(out.size);
		FILE *f = fopen(argv[a], "rb");

		if (f == nullptr || fread(x.data(), sizeof(float), in.size, f) !=
		    in.size) {
			fprintf(stderr, "%s: need %u floats\n", argv[a], in.size);
			return 1;
		}
		fclose(f);
		for (i = 0; i < in.size; i++) {
			long v = lrintf(x[i] * in.scale) + in.zero_point;
			if (in.data_type == AIPU_DATA_TYPE_S8)
				q[i] = (uint8_t)(int8_t)std::max(-128L,
				    std::min(127L, v));
			else
				q[i] = (uint8_t)std::max(0L, std::min(255L, v));
		}
		check(aipu_load_tensor(ctx, job, 0, q.data()), "load_tensor");
		auto t0 = std::chrono::steady_clock::now();
		check(aipu_finish_job(ctx, job, 5000), "finish_job");
		auto t1 = std::chrono::steady_clock::now();
		check(aipu_get_tensor(ctx, job, AIPU_TENSOR_TYPE_OUTPUT, 0,
		    y.data()), "get_tensor");

		std::vector<std::pair<float, int>> r;
		for (i = 0; i < out.size; i++) {
			int v = out.data_type == AIPU_DATA_TYPE_S8 ?
			    (int)(int8_t)y[i] : (int)y[i];
			r.push_back({(v - out.zero_point) / out.scale, (int)i});
		}
		std::partial_sort(r.begin(), r.begin() + 5, r.end(),
		    [](auto &p, auto &q) { return p.first > q.first; });
		printf("%s: %.2f ms\n", argv[a],
		    std::chrono::duration<double, std::milli>(t1 - t0).count());
		for (i = 0; i < 5; i++)
			printf("  %3d %7.3f %s\n", r[i].second, r[i].first,
			    r[i].second < (int)labels.size() ?
			    labels[r[i].second].c_str() : "");
	}
	check(aipu_clean_job(ctx, job), "clean_job");
	check(aipu_unload_graph(ctx, graph), "unload_graph");
	check(aipu_deinit_context(ctx), "deinit_context");
	return 0;
}
