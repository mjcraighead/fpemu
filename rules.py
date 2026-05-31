# fpemu (https://github.com/mjcraighead/fpemu)
# Copyright (c) 2025-2026 Matt Craighead
# SPDX-License-Identifier: MIT

def configs(*, rms=(0,), dazs=(0,), ftzs=(0,), imms=(0,)):
    return [
        (rm, daz, ftz, imm)
        for rm in rms
        for daz in dazs
        for ftz in ftzs
        for imm in imms
    ]

def vcvtps2ph_configs():
    return (
        configs(dazs=(0, 1), imms=range(4)) +
        configs(rms=range(4), dazs=(0, 1), imms=range(4, 8))
    )

def roundss_configs():
    out = []
    for imm in range(16):
        # imm[2] selects MXCSR.RC instead of imm[1:0], so only those
        # encodings need all four MXCSR rounding modes.
        for rm in range(4 if (imm & 4) else 1):
            for daz in [0, 1]:
                out.append((rm, daz, 0, imm))
    return out

exhaustive_specs = [
    ('sqrtss', configs(rms=range(4), dazs=(0, 1), ftzs=(0, 1)), 10),
    ('roundss', roundss_configs(), 1),
    ('cvtss2sd', configs(dazs=(0, 1)), 1),
    ('vcvtph2ps_lane', configs(), 1),
    ('vcvtps2ph_lane', vcvtps2ph_configs(), 1),
    ('cvtsi2ss_i32', configs(rms=range(4)), 1),
    ('cvtsi2sd_i32', configs(), 1),
    ('cvtss2si32', configs(rms=range(4), dazs=(0, 1)), 1),
    ('cvttss2si32', configs(dazs=(0, 1)), 1),
    ('cvtss2si64', configs(rms=range(4), dazs=(0, 1)), 1),
    ('cvttss2si64', configs(dazs=(0, 1)), 1),
]

def rules(ctx):
    random_worker_count = 256
    exhaustive_shard_count = 8
    common_flags = ['-std=gnu11', '-Wall', '-Wextra', '-Werror', '-march=x86-64-v3']
    opt_flags = ['-O2', *common_flags]
    cov_flags = ['-O0', '-g', '-fprofile-instr-generate', '-fcoverage-mapping', *common_flags]
    cov_link_flags = ['-fprofile-instr-generate', '-fcoverage-mapping']

    def test_rule(binary, log_path, argv, *, latency=1, profraw_path=None):
        outputs = [log_path] if profraw_path is None else [log_path, profraw_path]
        cmd = [binary, *argv, log_path]
        if profraw_path is not None:
            cmd = ['env', f'LLVM_PROFILE_FILE={profraw_path}', *cmd]
        ctx.rule(outputs, binary, cmd=cmd, latency=latency)

    def random_logs(binary, prefix, log10_cases, *, latency=1, profraw=False):
        outputs = []
        for i in range(random_worker_count):
            worker_seed = 1 + i
            worker_log = f'_out/{prefix}-1e{log10_cases}-worker{i}.log'
            profraw_path = None
            if profraw:
                profraw_path = f'_out/coverage/{prefix}-1e{log10_cases}-worker{i}.profraw'
            test_rule(binary, worker_log, ['rand', str(log10_cases), str(worker_seed)],
                      latency=latency, profraw_path=profraw_path)
            outputs.append(worker_log)
            if profraw_path is not None:
                outputs.append(profraw_path)
        return outputs

    def build_binary(target, obj_suffix, cc, compile_flags, *, link_flags=None):
        link_flags = link_flags or []
        objs = []
        for name in ['fpemu', 'test']:
            src = f'{name}.c'
            obj = f'_out/{name}{obj_suffix}'
            ctx.rule(obj, [src, 'fpemu.h'],
                     cmd=[cc, *compile_flags, '-c', src, '-o', obj])
            objs.append(obj)
        ctx.rule(target, objs, cmd=[cc, *link_flags, *objs, '-o', target])

    def exhaustive_op_logs(binary, op, configs, latency):
        outputs = []
        for (rm, daz, ftz, imm) in configs:
            for i in range(exhaustive_shard_count):
                log_path = f'_out/exhaustive-{op}-rm{rm}-daz{daz}-ftz{ftz}-imm{imm}-shard{i}.log'
                argv = ['exhaustive', op, str(rm), str(daz), str(ftz),
                        str(imm), str(i), str(exhaustive_shard_count)]
                test_rule(binary, log_path, argv, latency=latency)
                outputs.append(log_path)
        return outputs

    def coverage_report_rules(binary, profdata, report, show):
        cmd = (f'llvm-cov-21 report {binary} -instr-profile={profdata} '
               f'fpemu.c test.c --show-branch-summary >{report}')
        ctx.rule(report, [binary, profdata], cmd=['bash', '-c', cmd])
        cmd = (f'llvm-cov-21 show {binary} -instr-profile={profdata} '
               f'fpemu.c test.c --show-line-counts-or-regions --show-branches=count >{show}')
        ctx.rule(show, [binary, profdata], cmd=['bash', '-c', cmd])

    target = '_out/fpemu'
    build_binary(target, '.o', 'gcc', opt_flags)
    ctx.rule(':build', target)

    # Directed edge-case suite. Runs a curated FP32/FP64 case matrix * 4 rounding modes
    # * FTZ/DAZ * clean/sticky MXCSR and verifies HW vs. SW bit-for-bit including MXCSR.
    edge_log = '_out/edge.log'
    test_rule(target, edge_log, ['edge'])
    ctx.rule(':edge', edge_log)

    # Typical fast sanity pass: directed edges plus a short deterministic random run.
    # Intended to finish very quickly on ordinary development machines.
    smoke_log = '_out/smoke.log'
    test_rule(target, smoke_log, ['rand', '7', '1'])
    ctx.rule(':smoke', [edge_log, smoke_log])

    # Standard validation pass: one deterministic 25.6B-case plan, run across
    # 256 independent 10^8-case workers for parallel scheduling.
    test_logs = random_logs(target, 'test', log10_cases=8)
    ctx.rule(':test', [edge_log] + test_logs)

    # Long soak: 2.56T total cases. Intended to run for
    # longer validation passes on high-core-count machines.
    long_logs = random_logs(target, 'test', log10_cases=10, latency=10)
    ctx.rule(':long', [edge_log] + long_logs)

    # Exhaustive single-input validation. Each operation has at most 2^32 data
    # inputs; rules enumerate the relevant MXCSR controls/immediates and shard
    # the data input space.
    exhaustive_all_outputs = [
        log_path
        for (op, op_configs, latency) in exhaustive_specs
        for log_path in exhaustive_op_logs(target, op, op_configs, latency)
    ]
    ctx.rule(':exhaustive', exhaustive_all_outputs)

    cov_target = '_out/fpemu_cov'
    build_binary(cov_target, '_cov.o', 'clang-21', cov_flags, link_flags=cov_link_flags)
    ctx.rule(':coverage-build', cov_target)

    cov_edge_log = '_out/cov-edge.log'
    cov_edge_profraw = '_out/coverage/cov-edge.profraw'
    test_rule(cov_target, cov_edge_log, ['edge'], profraw_path=cov_edge_profraw)

    cov_smoke_log = '_out/cov-smoke.log'
    cov_smoke_profraw = '_out/coverage/cov-smoke.profraw'
    test_rule(cov_target, cov_smoke_log, ['rand', '7', '1'], profraw_path=cov_smoke_profraw)

    cov_test_outputs = random_logs(cov_target, 'cov-test', log10_cases=8, profraw=True)
    cov_test_profraws = [path for path in cov_test_outputs if path.endswith('.profraw')]

    cov_profdata = '_out/coverage/cov.profdata'
    ctx.rule(cov_profdata, [cov_target, cov_edge_profraw, cov_smoke_profraw],
             cmd=['llvm-profdata-21', 'merge', '-sparse', cov_edge_profraw, cov_smoke_profraw,
                  '-o', cov_profdata])

    cov_report = '_out/coverage/cov-report.txt'
    cov_show = '_out/coverage/cov-show.txt'
    coverage_report_rules(cov_target, cov_profdata, cov_report, cov_show)
    ctx.rule(':coverage-report', [cov_report, cov_show])

    cov_test_profdata = '_out/coverage/cov-test.profdata'
    ctx.rule(cov_test_profdata,
             [cov_target, cov_edge_profraw, cov_smoke_profraw, *cov_test_profraws],
             cmd=['llvm-profdata-21', 'merge', '-sparse',
                  cov_edge_profraw, cov_smoke_profraw, *cov_test_profraws,
                  '-o', cov_test_profdata])

    cov_test_report = '_out/coverage/cov-test-report.txt'
    cov_test_show = '_out/coverage/cov-test-show.txt'
    coverage_report_rules(cov_target, cov_test_profdata, cov_test_report, cov_test_show)
    ctx.rule(':coverage-test-report', [cov_test_report, cov_test_show])
