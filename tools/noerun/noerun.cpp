/*
 * SPDX-License-Identifier: BSD-2-Clause
 *
 * npurun through CIX's libnoe (binary only, Linux) in place of libaipudrv:
 * for the Linuxulator, where libnoe loads a Linux libaipudrv.so built from
 * this repository (tools/build-linux.sh; AIPU_LIB_PATH or the library path).
 * It loads CIX's .cix model files itself.
 *
 * Usage: noerun model.cix labels.txt image.f32 ...
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "cix_noe_standard_api.h"

static noe_context_t ctx;

static void
check(noe_status_t st, const char *what)
{
	const char *msg = nullptr;

	if (st == NOE_STATUS_SUCCESS)
		return;
	noe_get_error_message(ctx, st, &msg);
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
	tensor_desc_t in, out;
	uint64_t graph, job;
	uint32_t nin, nout, i;

	if (argc < 4) {
		fprintf(stderr, "usage: noerun model.cix labels.txt image.f32 ...\n");
		return 2;
	}
	std::vector<std::string> labels = read_labels(argv[2]);

	check(noe_init_context(&ctx), "init_context");
	check(noe_load_graph(ctx, argv[1], &graph), "load_graph");
	check(noe_get_tensor_count(ctx, graph, NOE_TENSOR_TYPE_INPUT, &nin),
	    "input count");
	check(noe_get_tensor_count(ctx, graph, NOE_TENSOR_TYPE_OUTPUT, &nout),
	    "output count");
	check(noe_get_tensor_descriptor(ctx, graph, NOE_TENSOR_TYPE_INPUT, 0,
	    &in), "input descriptor");
	check(noe_get_tensor_descriptor(ctx, graph, NOE_TENSOR_TYPE_OUTPUT, 0,
	    &out), "output descriptor");
	printf("graph loaded: %u input(s), %u output(s)\n", nin, nout);
	printf("input 0: %u bytes, type %d, scale %g, zero point %d\n", in.size,
	    (int)in.data_type, in.scale, in.zero_point);
	printf("output 0: %u bytes, type %d, scale %g, zero point %d\n", out.size,
	    (int)out.data_type, out.scale, out.zero_point);
	if ((in.data_type != NOE_DATA_TYPE_S8 &&
	    in.data_type != NOE_DATA_TYPE_U8) ||
	    (out.data_type != NOE_DATA_TYPE_S8 &&
	    out.data_type != NOE_DATA_TYPE_U8)) {
		fprintf(stderr, "only 8-bit tensors handled\n");
		return 1;
	}
	/*
	 * libnoe 3.1.2 reads config->conf_j_npu whatever the header's default
	 * (nullptr) says: a default configuration, then.
	 */
	job_config_npu_t npu_cfg = {};
	job_config_t job_cfg = { &npu_cfg };
	check(noe_create_job(ctx, graph, &job, &job_cfg), "create_job");

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
			if (in.data_type == NOE_DATA_TYPE_S8)
				q[i] = (uint8_t)(int8_t)std::max(-128L,
				    std::min(127L, v));
			else
				q[i] = (uint8_t)std::max(0L, std::min(255L, v));
		}
		check(noe_load_tensor(ctx, job, 0, q.data()), "load_tensor");
		auto t0 = std::chrono::steady_clock::now();
		check(noe_job_infer_sync(ctx, job, 5000), "job_infer_sync");
		auto t1 = std::chrono::steady_clock::now();
		check(noe_get_tensor(ctx, job, NOE_TENSOR_TYPE_OUTPUT, 0,
		    y.data()), "get_tensor");

		std::vector<std::pair<float, int>> r;
		for (i = 0; i < out.size; i++) {
			int v = out.data_type == NOE_DATA_TYPE_S8 ?
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
	check(noe_clean_job(ctx, job), "clean_job");
	check(noe_unload_graph(ctx, graph), "unload_graph");
	check(noe_deinit_context(ctx), "deinit_context");
	return 0;
}
