# Strata Engine - top-level build (P1.S1).
#
# Targets (phase-1 A�§P1.S1): `strata_artifact` (no CUDA dependency), `tests`.  `strata_plan` and the C++
# `strata-pack` are added when their sources exist - a target with no sources would make the build graph
# claim something the tree does not have, which is worse than an absent target.
#
# This is a dual-target AMD build: CMAKE_HIP_ARCHITECTURES defaults to gfx1031 (RX 6700 XT / RDNA2) AND
# gfx1201 (RX 9070 XT / RDNA4, the host this tree is developed and verified against). It does not reject
# any other arch. A stale comment here used to claim the engine was written against sm_120 "RTX 5000 /
# Blackwell" and that the build rejected every other architecture -- both were untrue and they made people
# read a 12 GB RDNA2 capacity analysis as if it described the machine in front of them. See section 0 of
# docs/RESEARCH_NOTES.md for the measured host.
cmake_minimum_required(VERSION 3.24)
project(strata VERSION 0.1.15 LANGUAGES CXX)
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")
# issue #31: release builds keep a PDB beside the exe (not shipped in it), so a minidump from a stalled engine can
# be read back to the exact source line
if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zi>)
  add_link_options(/DEBUG /OPT:REF /OPT:ICF)
endif()

# The prebuilt release (setup.py downloads it, so users need no compiler): ggml's CPU backend for any AVX2 CPU
# instead of this machine's (Strata's own AVX-512 kernels are per-file and chosen at run time either way), and
# the MSVC runtime linked in, so no Visual C++ redistributable is needed.
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_ENABLE_HIP "Build the HIP/ROCm targets (Phase 2+)" OFF)
# The test suite and the micro benchmarks live in tests/ and bench/micro/, which the published source leaves out.
set(_strata_tests_default OFF)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro)
  set(_strata_tests_default ON)
endif()
option(STRATA_BUILD_TESTS "Build the test suite" ${_strata_tests_default})
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)

if(STRATA_ENABLE_HIP)

  if(NOT DEFINED ROCM_INSTALL_DIR)
    set(ROCM_INSTALL_DIR "$ENV{ROCM_INSTALL_DIR}" CACHE PATH "ROCm installation root")
  endif()
  if(NOT EXISTS "${ROCM_INSTALL_DIR}")
    set(ROCM_INSTALL_DIR "G:/ROCM10RT-gfx1201" CACHE PATH "ROCm installation root")
  endif()
  set(ROCM_LIBRARY "${ROCM_INSTALL_DIR}/lib")
  set(ROCM_INCLUDE "${ROCM_INSTALL_DIR}/include")
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx1031 gfx1201 CACHE STRING "HIP architectures" FORCE)
  endif()
  foreach(_a IN LISTS CMAKE_HIP_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" _base "${_a}")
    if(_base STREQUAL "native" OR _base STREQUAL "all" OR _base STREQUAL "all-major")
      continue()
    endif()
  endforeach()
  message(STATUS "Strata: HIP enabled, arch ${CMAKE_HIP_ARCHITECTURES}")
  # CMake 3.29 with CMAKE_HIP_COMPILER_FORCED=ON does not generate HIP compilation rules for .cu files.
  # Apply -x hip flags manually to .cu sources so clang-cl treats them as HIP.
  function(strata_apply_hip_flags target_name)
    get_target_property(_sources ${target_name} SOURCES)
    if(NOT _sources)
      return()
    endif()
    foreach(_src ${_sources})
      if(_src MATCHES "\\.cu$")
        if(IS_ABSOLUTE "${_src}")
          set(_src_abs "${_src}")
        else()
          set(_src_abs "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
        endif()
        if(EXISTS "${_src_abs}")
          set(_hip_flags "-x;hip")
          foreach(_a IN LISTS CMAKE_HIP_ARCHITECTURES)
            string(REGEX REPLACE "-.*$" "" _base "${_a}")
            if(_base STREQUAL "native")
              list(APPEND _hip_flags "--offload-arch=native")
            elseif(_base STREQUAL "all")
              list(APPEND _hip_flags "--offload-arch=all")
            elseif(_base STREQUAL "all-major")
              list(APPEND _hip_flags "--offload-arch=all-major")
            else()
              list(APPEND _hip_flags "--offload-arch=${_a}")
            endif()
          endforeach()
          get_source_file_property(_existing_opts ${_src} COMPILE_OPTIONS)
          if(_existing_opts)
            list(APPEND _hip_flags ${_existing_opts})
          endif()
          set_source_files_properties(${_src_abs} PROPERTIES
            COMPILE_OPTIONS "${_hip_flags}"
            COMPILE_DEFINITIONS "__HIP__"
            LANGUAGE CXX)
        endif()
      endif()
    endforeach()
  endfunction()
endif()

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose -W4 warnings are not this project's to fix and
# whose /WX failures would be indistinguishable from our own.
#
# Every flag is guarded by COMPILE_LANGUAGE:CXX.  Without that guard hipcc receives bare `/W4 /utf-8` and dies
# with "A single input file is required for a non-link phase" - it parses them as inputs rather than host
# flags.  Host flags for HIP must go through -Xcompiler, which is what COMPILE_LANGUAGE:HIP does below.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/W4>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>)
endif()

# ---- strata_artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

add_executable(strata-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(strata-gguf PRIVATE strata_artifact)

add_executable(strata-dequant src/artifact/dequant.cpp)
target_link_libraries(strata-dequant PRIVATE strata_artifact)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(strata-dequant PRIVATE STRATA_DEQUANT_SELFTEST)

# ---- strata_plan ----------------------------------------------------------
# Also header-only.  A "planner" that hides state would be one whose decision is not inspectable, and the
# whole point of P1.S9 is that the decision is arithmetic you can check by hand.
add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

add_executable(strata-plan src/plan/plan_main.cpp)
target_link_libraries(strata-plan PRIVATE strata_plan)

# ---- strata_core (HIP, Phase 2) ------------------------------------------
# Only built when HIP is on, so the Phase 1 targets keep working on a machine with no toolkit.
if(STRATA_ENABLE_HIP)
  add_library(strata_core STATIC src/core/device.cu src/core/pinned.cu src/platform/memory.cpp src/core/graph.cpp src/core/weights.cpp
                               src/core/layout.cpp)
  target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${ROCM_INCLUDE})
  # HIP headers (hip/hip_runtime.h) require __HIP_PLATFORM_AMD__ to be defined
  # when not going through the hipcc wrapper (e.g. for CXX files that include HIP headers).
  target_compile_definitions(strata_core PUBLIC __HIP_PLATFORM_AMD__)
  # amdhip64 is needed for graph.cpp (a .cpp that calls hipGraphLaunch and hipEventQuery):
  # the .cu sources get the HIP runtime through the HIP language rules, a .cpp does not.
  target_link_libraries(strata_core PUBLIC strata_plan strata_warnings ${ROCM_LIBRARY}/amdhip64.lib)
  strata_apply_hip_flags(strata_core)

  add_executable(strata-device src/core/device_main.cpp)
  target_link_libraries(strata-device PRIVATE strata_core)

  add_executable(strata-load src/core/load_main.cpp)
  target_link_libraries(strata-load PRIVATE strata_core strata_kernels)

  add_executable(strata-concurrent src/core/concurrent_main.cpp)
  target_include_directories(strata-concurrent PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(strata-concurrent PRIVATE strata_core strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)

  add_executable(strata-overlap src/core/overlap_main.cpp)
  # a .cpp that calls cudaMemcpyAsync itself needs the toolkit headers and cudart explicitly
  target_include_directories(strata-overlap PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(strata-overlap PRIVATE strata_core strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)

  # The load test needs the 34 GB artifact, so it is registered ONLY where that artifact exists.  A test that
  # silently skips is a test that stops running; an absent registration is visible in the configure output.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin)
    add_test(NAME expert_arena_load
             COMMAND strata-load --file ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin --layers 1)
    set_tests_properties(expert_arena_load PROPERTIES PASS_REGULAR_EXPRESSION "per-layer checksums")
  else()
    message(STATUS "strata: pack/full/experts.bin absent - the expert arena load test is NOT registered")
  endif()

  # ---- strata_kernels (P2.S2) ----
  # Naive first, each with a parity test against a reference that was itself validated against ggml.  The
  # kernels link `strata_artifact` for the scalar dequantizers, because using the VALIDATED scalar path as the
  # reference is the whole point - a kernel checked against another copy of itself proves nothing.
  add_library(strata_kernels STATIC src/kernels/cuda/dequant_s2.cu src/kernels/cuda/s2_gemv.cu
                          src/kernels/cuda/s_gemv.cu
                          src/kernels/cuda/s2_gemv_quads.cu
                          src/kernels/cuda/s2_gemv_fast.cu
                          src/kernels/cuda/router_top10.cu
                          src/kernels/cuda/quantize_act.cu
                          src/kernels/cuda/rope.cu
                          src/kernels/cuda/sampler.cu
                          src/kernels/cuda/s2_gemv_q8.cu
    src/kernels/cuda/s2_expert_grouped.cu
                          src/kernels/cuda/shared_expert.cu
                          src/kernels/cuda/gr.cu
                          src/kernels/cuda/gdn.cu
                          src/kernels/cuda/qsa.cu
                          src/kernels/cuda/elementwise.cu
                          src/kernels/cuda/bf16_gemv.cu
                          src/kernels/cuda/native_mmvq.cu
                          src/kernels/cuda/native_bf16.cu
                          src/kernels/cuda/native_gr_norm.cu
                          src/kernels/cuda/native_moe.cu
                          src/kernels/cuda/native_gdn.cu
                          src/kernels/cuda/native_gr_postops.cu
                          src/kernels/cuda/native_router.cu
                          src/kernels/cuda/native_gdn_preprocess.cu
                          src/kernels/cuda/native_qsa.cu
                          src/kernels/cuda/native_rope.cu
                          src/kernels/cuda/mla.cu
                          src/kernels/cuda/ds2_moe.cu
                          src/kernels/cuda/native_ple_postops.cu
                          src/kernels/cuda/native_flash_attn.cu
                          src/kernels/cuda/native_qsa_indexer.cu
                          src/kernels/cuda/native_qsa_score.cu
                          src/kernels/cuda/ple.cu
                          src/kernels/cuda/kv_q8.cu
                          src/kernels/cuda/kv_stream.cu
                          src/kernels/cuda/kv_q4.cu
                          src/kernels/cuda/fused_gr.cu
                          src/kernels/cuda/cvec.cu
                          src/kernels/cuda/qsa_decode_attn.cu
                          src/kernels/cuda/fused_gdn.cu
                         src/kernels/cuda/dequant_bf16.cu
                         src/kernels/cuda/qsa_select.cu
                         src/kernels/cuda/verify_kernels.cu
                         src/kernels/cuda/iq_kernels.cu
                          src/kernels/cuda/fp4_gemv.cu
                          src/kernels/cuda/fp4_gemv_fast.cu
                          src/kernels/cuda/fp4_gemv_tiled.cu
                          src/kernels/cuda/fp4_gemv_coalesced.cu
                          src/kernels/cuda/fp4_gemv_q8.cu
                          src/kernels/ngram.cpp
                          src/ngram/ple_reader.cpp
                          src/platform/direct_file.cpp)
  target_link_libraries(strata_kernels PUBLIC strata_core strata_artifact)
  # plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/ggml)
  target_include_directories(strata_kernels PRIVATE third_party/ggml)
  # Match the pinned llama.cpp CUDA arithmetic contract for this adapter only.
  set_source_files_properties(src/kernels/cuda/native_mmvq.cu src/kernels/cuda/native_bf16.cu
    src/kernels/cuda/native_gr_norm.cu
    src/kernels/cuda/native_moe.cu
    src/kernels/cuda/native_gdn.cu
    src/kernels/cuda/native_router.cu
    src/kernels/cuda/native_qsa.cu
    src/kernels/cuda/native_rope.cu
    PROPERTIES COMPILE_OPTIONS "-ffast-math")
  # nvcc treats the explicit-rounding intrinsics (__fmul_rn/__fadd_rn/__fsub_rn/__fdiv_rn)
  # as non-fusible, but clang's AMDGPU backend contracts them into an FMA unless
  # contraction is off. These files use those intrinsics, so pin the contract.
  set_source_files_properties(src/kernels/cuda/native_gr_postops.cu
    src/kernels/cuda/native_gdn_preprocess.cu
    src/kernels/cuda/native_ple_postops.cu
    src/kernels/cuda/native_flash_attn.cu
    src/kernels/cuda/native_qsa_indexer.cu
    src/kernels/cuda/native_qsa_score.cu
    PROPERTIES COMPILE_OPTIONS "-ffast-math;-ffp-contract=off")
  set_source_files_properties(src/kernels/cuda/elementwise.cu
    src/kernels/cuda/verify_kernels.cu
    src/kernels/cuda/qsa.cu
    src/kernels/cuda/quantize_act.cu
    src/kernels/cuda/s2_expert_grouped.cu
    PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  strata_apply_hip_flags(strata_kernels)

  set(STRATA_ORACLE_SOURCE_DIR "" CACHE PATH "Pinned llama.cpp source for independent CUDA parity tests")
  set(STRATA_ORACLE_BUILD_DIR "" CACHE PATH "Matching existing shared llama.cpp build")
  if(STRATA_ORACLE_SOURCE_DIR AND STRATA_ORACLE_BUILD_DIR)
    foreach(parity IN ITEMS native_mmvq native_bf16 native_shared native_gr_norm native_moe native_gdn native_gr_postops native_router native_gdn_preprocess native_gdn_pipeline native_qsa native_rope native_ple_postops native_flash_attn native_qsa_indexer native_qsa_score)
    add_executable(${parity}_parity bench/micro/${parity}_parity.cpp)
    target_include_directories(${parity}_parity PRIVATE
      "${STRATA_ORACLE_SOURCE_DIR}/ggml/include" ${ROCM_INCLUDE})
    target_compile_definitions(${parity}_parity PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
    target_link_libraries(${parity}_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
    foreach(component IN ITEMS ggml ggml-base ggml-cpu ggml-cuda)
      string(REPLACE "-" "_" variable "STRATA_ORACLE_${component}")
      find_library(${variable} NAMES ${component} PATHS
        "${STRATA_ORACLE_BUILD_DIR}/ggml/src" "${STRATA_ORACLE_BUILD_DIR}/ggml/src/${component}"
        "${STRATA_ORACLE_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
      target_link_libraries(${parity}_parity PRIVATE "${${variable}}")
    endforeach()
    add_test(NAME ${parity}_parity COMMAND ${parity}_parity --selftest)
    if(WIN32)
      set_tests_properties(${parity}_parity PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${STRATA_ORACLE_BUILD_DIR}/bin")
    endif()
    endforeach()
  endif()

  # ---- strata_engine (P2.S5) -----------------------------------------------------
  # The COMPOSED layer lives here and not in `strata_core`, because it calls kernels and `strata_kernels`
  # already links `strata_core` - putting `gdn_layer` in core would make the two mutually dependent.  A third
  # library is the honest shape of the dependency graph: core knows about devices and weights, kernels know
  # about arithmetic, and this one knows the ORDER the arithmetic happens in.
  add_library(strata_engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp
    src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp)
  target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu ${ROCM_LIBRARY}/amdhip64.lib)

  # `strata generate` (P2.S6).  The FIRST user-facing program: everything under it is a component, and this is
  # what composes them into a token.  No tokenizer - the prompt arrives as ids - because Gate C1 compares logits
  # against llama.cpp on the SAME ids and a tokenizer on one side of that comparison is a second variable.
  # plan v0.3 P5: batched prompt processing (tensor-core GEMMs through cuBLAS).
   add_library(strata_prefill STATIC src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/prefill.cpp)
   target_include_directories(strata_prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${ROCM_INCLUDE})
   target_link_libraries(strata_prefill PUBLIC strata_engine ${ROCM_LIBRARY}/hipblas.lib ${ROCM_LIBRARY}/amdhip64.lib)
   strata_apply_hip_flags(strata_prefill)

  add_executable(strata src/program/generate.cpp)
  target_include_directories(strata PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec ${ROCM_LIBRARY}/amdhip64.lib)

  add_executable(dequant_s2_parity src/kernels/dequant_s2_parity.cpp)
  # The tests are HOST C++ files that call cudaMalloc/cudaMemcpy themselves, so they need the toolkit headers
  # and cudart explicitly: enabling the CUDA language gives those to .cu sources automatically, but a .cpp
  # target is just a C++ target and gets neither.
  target_include_directories(dequant_s2_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(dequant_s2_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME dequant_s2_parity COMMAND dequant_s2_parity --selftest)

  # MXFP4/NVFP4 GEMV: software FP4 decode (no FP4 tensor core on gfx1201/gfx1031), parity against the same
  # scalar dequantizers `dequant_fp4_test` checks bit-for-bit against ggml.
  add_executable(fp4_gemv_parity src/kernels/fp4_gemv_parity.cpp)
  target_include_directories(fp4_gemv_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(fp4_gemv_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME fp4_gemv_parity COMMAND fp4_gemv_parity --selftest)

  add_executable(s2_gemv_parity src/kernels/s2_gemv_parity.cpp)
  target_include_directories(s2_gemv_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(s2_gemv_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME s2_gemv_parity COMMAND s2_gemv_parity --selftest)

  add_executable(iq_parity src/kernels/iq_parity.cpp)
  target_include_directories(iq_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(iq_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_executable(shared_expert_parity src/kernels/shared_expert_parity.cpp)
  target_include_directories(shared_expert_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(shared_expert_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME shared_expert_parity COMMAND shared_expert_parity --selftest)

  add_executable(gr_parity src/kernels/gr_parity.cpp)
  target_include_directories(gr_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(gr_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME gr_parity COMMAND gr_parity --selftest)

  add_executable(gdn_parity src/kernels/gdn_parity.cpp)
  target_include_directories(gdn_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(gdn_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME gdn_parity COMMAND gdn_parity --selftest)

  add_executable(s2_gemv_q8_parity src/kernels/s2_gemv_q8_parity.cpp)
  target_include_directories(s2_gemv_q8_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(s2_gemv_q8_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME s2_gemv_q8_parity COMMAND s2_gemv_q8_parity --selftest)

  add_executable(sampler_parity src/kernels/sampler_parity.cpp)
  target_include_directories(sampler_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(sampler_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME sampler_parity COMMAND sampler_parity --selftest)

  add_executable(rope_parity src/kernels/rope_parity.cpp)
  target_include_directories(rope_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(rope_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME rope_parity COMMAND rope_parity --selftest)

  add_executable(quantize_act_parity src/kernels/quantize_act_parity.cpp)
  target_include_directories(quantize_act_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(quantize_act_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME quantize_act_parity COMMAND quantize_act_parity --selftest)

  add_executable(router_top10_parity src/kernels/router_top10_parity.cpp)
  target_include_directories(router_top10_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(router_top10_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME router_top10_parity COMMAND router_top10_parity --selftest)

  add_executable(s_gemv_parity src/kernels/s_gemv_parity.cpp)
  target_include_directories(s_gemv_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(s_gemv_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME s_gemv_parity COMMAND s_gemv_parity --selftest)

  add_executable(elementwise_parity src/kernels/elementwise_parity.cpp)
  target_include_directories(elementwise_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(elementwise_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME elementwise_parity COMMAND elementwise_parity --selftest)

  add_executable(bf16_gemv_parity src/kernels/bf16_gemv_parity.cpp)
  target_include_directories(bf16_gemv_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(bf16_gemv_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME bf16_gemv_parity COMMAND bf16_gemv_parity --selftest)

  add_executable(s_gemv_q8k_parity src/kernels/s_gemv_q8k_parity.cpp)
  target_include_directories(s_gemv_q8k_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(s_gemv_q8k_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME s_gemv_q8k_parity COMMAND s_gemv_q8k_parity --selftest)

  add_executable(qsa_parity src/kernels/qsa_parity.cpp)
  target_include_directories(qsa_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(qsa_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME qsa_parity COMMAND qsa_parity --selftest)

  add_executable(mla_parity src/kernels/mla_parity.cpp)
  target_include_directories(mla_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(mla_parity PRIVATE strata_engine ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME mla_parity COMMAND mla_parity --selftest)

  add_executable(ds2_moe_parity src/kernels/ds2_moe_parity.cpp)
  target_include_directories(ds2_moe_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(ds2_moe_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME ds2_moe_parity COMMAND ds2_moe_parity --selftest)

  # ---- P2.S4: the PLE n-gram hash, the IQ4_NL table read, and the PLE block ----
  # `ngram.cpp` is host-only but lives in `strata_kernels` because `ple.cu` shares its geometry constants, and
  # splitting them across two libraries would only add a layer.
  #
  # The pack, the original GGUF and the ggml capture are all REQUIRED inputs: this test's oracles are real
  # artifacts, so the working directory is pinned to the source tree and a missing input is a hard error
  # INSIDE the test.  A test that passes by finding nothing to compare is the failure mode this project keeps
  # re-learning.
  add_executable(ple_parity src/kernels/ple_parity.cpp)
  target_include_directories(ple_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(ple_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  set(STRATA_PLE_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro" CACHE PATH
      "Directory containing required ple_in.bin and ple_out.bin parity fixtures")
  add_test(NAME ple_parity COMMAND ple_parity --selftest
    --in "${STRATA_PLE_FIXTURE_DIR}/ple_in.bin" --out "${STRATA_PLE_FIXTURE_DIR}/ple_out.bin"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  # Plan v0.3 P7: INT8 KV storage vs a host reference and vs the FP16 path (GPU, synthetic, no model).
  add_executable(kv_q8_parity src/kernels/kv_q8_parity.cpp)
  target_include_directories(kv_q8_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(kv_q8_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME kv_q8_parity COMMAND kv_q8_parity)

  # KV streaming (docs/kv-streaming-design.md): streamed vs resident attention, bitwise (GPU, synthetic, no model).
  add_executable(kv_stream_parity src/kernels/kv_stream_parity.cpp)
  target_include_directories(kv_stream_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(kv_stream_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME kv_stream_parity COMMAND kv_stream_parity)

  # PR #21: Q4_0 KV with Walsh-Hadamard rotation vs a host reference (GPU, synthetic, no model).
  add_executable(kv_q4_parity src/kernels/kv_q4_parity.cpp)
  target_include_directories(kv_q4_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(kv_q4_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME kv_q4_parity COMMAND kv_q4_parity)
  add_executable(cvec_parity src/kernels/cvec_parity.cpp)
  target_include_directories(cvec_parity PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(cvec_parity PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME cvec_parity COMMAND cvec_parity)

  # Plan v0.3 P3: native MMVQ for 1..8 tokens (bitwise selftest + --bench timing); synthetic data, no model.
  if(STRATA_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/native_mmvq_multi.cpp)
  add_executable(native_mmvq_multi bench/micro/native_mmvq_multi.cpp)
  target_include_directories(native_mmvq_multi PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(native_mmvq_multi PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME native_mmvq_multi COMMAND native_mmvq_multi --selftest)
  else()
  message(STATUS "strata: bench/micro/native_mmvq_multi.cpp absent - that micro-bench is NOT registered")
  endif()

  # Plan v0.3 P0.1: working-set lock of a host region (CPU only).
  add_executable(platform_memory_test src/platform/memory_test.cpp)
  target_link_libraries(platform_memory_test PRIVATE strata_core ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME platform_memory_test COMMAND platform_memory_test)

  # Plan v0.3 P2: the n-gram table read straight from the SSD. The selftest needs only the CPU and a temp file.
  add_executable(ple_reader_test src/ngram/ple_reader_test.cpp)
  target_include_directories(ple_reader_test PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(ple_reader_test PRIVATE strata_kernels ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME ple_reader_selftest COMMAND ple_reader_test --selftest)

  # The GPU test is registered only when CUDA is on, and it runs ON THE TARGET PC.  `--selftest` allocates
  # and writes real VRAM, so a machine without a device fails this test rather than skipping it silently -
  # which is correct: an engine that targets one GPU should not report a pass for having no GPU.
  add_test(NAME cuda_device_selftest COMMAND strata-device --selftest)

  # ---- P2.S5: the GraphRegistry --------------------------------------------------
  # A .cu because the test needs real kernels to capture; it links `strata_core` and not `strata_kernels`,
  # because what it exercises is the capture machinery and not any kernel in it.
  if(STRATA_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/graph_registry_test.cu)
  add_executable(graph_registry_test bench/micro/graph_registry_test.cu)
  target_include_directories(graph_registry_test PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(graph_registry_test PRIVATE strata_core ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME graph_registry_test COMMAND graph_registry_test)

  # ---- P2.S5's GATE: is a device<->pinned-memory handoff graph-capturable? --------------------
  # The doorbell protocol puts the D2H of `x_f` and the miss list INSIDE G[l] - that is the whole overlap.  The
  # phase doc says to verify this "before building on it", and this project has already lost two rounds to
  # building first (a memory-only spin that DEADLOCKS on Windows, and a `cudaEventRecord` inside a capture that
  # is SILENTLY DROPPED).  It links only cudart: it is about the capture machinery, not about any kernel.
  add_executable(pinned_capture bench/micro/pinned_capture.cu)
  target_include_directories(pinned_capture PRIVATE ${ROCM_INCLUDE})
  target_link_libraries(pinned_capture PRIVATE ${ROCM_LIBRARY}/amdhip64.lib)
  add_test(NAME pinned_capture COMMAND pinned_capture)
  else()
  message(STATUS "strata: bench/micro/graph_registry_test.cu absent - GraphRegistry + pinned_capture are NOT registered")
  endif()

  # Tells tests/CMakeLists.txt to add the CUDA-dependent Catch2 target.  It cannot be declared there
  # unconditionally: tests/ is added at the end of this file, but the flag has to be known before it runs.
  set(STRATA_CUDA_TESTS ON)
endif()

# ---- strata_kernels_cpu (P2.S3) ----
#
# A SEPARATE LIBRARY from `strata_kernels`, and `/arch:AVX512` is set on the SOURCE rather than on the target
# so that the flag applies only to the TU that needs it.  Both parts matter:
#
#   * the CPU expert kernel uses `_mm256_dpbusd_epi32` (AVX512-VNNI) and `_mm256_multishift_epi64_epi8`
#     (AVX512-VBMI); without the flag MSVC reports undeclared intrinsics and never mentions AVX-512;
#   * a TU compiled WITH `/arch:AVX512` may use AVX-512 in ANY of its code, so the scalar fallback and the
#     test drivers must NOT be in it - otherwise the fallback is not a fallback, it is the same bet with
#     worse opt-out.  `cpu_require_expert_support()` therefore refuses at startup on a CPU without the
#     features, and the fallback exists for tests, exactly as P2.S3 specifies.
# Plan v0.3 P6: speculation components that need no GPU.
add_library(strata_spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp)
target_include_directories(strata_spec PUBLIC include)
if(STRATA_ENABLE_HIP)
  target_include_directories(strata_spec PUBLIC ${ROCM_INCLUDE})
  target_compile_definitions(strata_spec PUBLIC __HIP_PLATFORM_AMD__)
endif()
if(STRATA_BUILD_TESTS)
  add_executable(suffix_drafter_test src/spec/suffix_drafter_test.cpp)
  target_link_libraries(suffix_drafter_test PRIVATE strata_spec)
  add_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
  add_executable(controller_test src/spec/controller_test.cpp)
  target_link_libraries(controller_test PRIVATE strata_spec)
  add_test(NAME controller_test COMMAND controller_test)
  add_executable(draft_policy_test src/spec/draft_policy_test.cpp)
  target_link_libraries(draft_policy_test PRIVATE strata_spec)
  add_test(NAME draft_policy_test COMMAND draft_policy_test)
endif()

add_library(strata_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
            src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp)
target_include_directories(strata_kernels_cpu PUBLIC include)
# Plan v0.3 P6: only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
if(MSVC)
  set_source_files_properties(src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  # GCC/Clang spell it differently, and `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
  set_source_files_properties(src/kernels/cpu/expert.cpp
                              PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC Threads::Threads)

# ---- plan v0.3 P6: the i-quant models' CPU experts (IQ2_XS / IQ3_XXS files) ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
# weight means exactly what it means in llama.cpp.  ggml is built static from a llama.cpp checkout: a local one
# with -DSTRATA_GGML_DIR=..., otherwise the pinned upstream commit (third_party/ggml/VERSION.txt).
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(STRATA_NATIVE_EXPERTS)
  enable_language(C)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  if(STRATA_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
  if(STRATA_GGML_DIR)
    add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(strata_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
    FetchContent_GetProperties(strata_llamacpp)
    if(NOT strata_llamacpp_POPULATED)
      FetchContent_Populate(strata_llamacpp)
    endif()
    add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  endif()
  # part of strata_kernels_cpu (the pool calls these and they call its CPU probe: one library, so GNU ld's
  # one-pass static linking has no cycle to resolve)
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp
                                           src/kernels/cpu/iq_avx2.cpp)
   target_include_directories(strata_kernels_cpu PRIVATE third_party/ggml ${ROCM_INCLUDE})
   target_compile_definitions(strata_kernels_cpu PRIVATE __HIP_PLATFORM_AMD__)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  endif()
  
  # clang-cl needs the same AVX-512 flags even under MSVC simulation
  if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
    set_source_files_properties(src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  endif()
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
  if(STRATA_ENABLE_HIP)
    # prompt-speed plan step 2b: llama.cpp's MMQ kernels (ggml-cuda, MIT) for the prompt path's experts, compiled
    # from the same checkout as ggml above - only mmq.cuh's instances for the packs' types and the quantizer, with
    # the few host symbols of ggml-cuda.cu they use in src/prefill/ggml_cuda_host.cu
    # NOTE: llama.cpp .cu files (quantize.cu, mmq-instance-*.cu) are CUDA code that needs separate hipification.
    # The strata_mmo target is built only when STRATA_HIP_HIPPY_MM is ON (requires llama.cpp CUDA->HIP conversion).
    option(STRATA_HIP_HIPPY_MM "Build the MMQ prefill path (requires llama.cpp .cu files to be hipified)" OFF)
    if(STRATA_GGML_DIR)
      set(_strata_gg ${STRATA_GGML_DIR}/ggml)
    else()
      set(_strata_gg ${strata_llamacpp_SOURCE_DIR}/ggml)
    endif()
    set(_strata_mmq ${_strata_gg}/src/ggml-cuda)
    if(STRATA_HIP_HIPPY_MM)
      add_library(strata_mmq STATIC src/prefill/moe_mmq.cu src/prefill/ggml_cuda_host.cu ${_strata_mmq}/quantize.cu)
      foreach(_t q2_0 q4_0 q5_0 q8_0 q2_k q3_k q4_k q5_k q6_k iq1_s iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs)
        target_sources(strata_mmq PRIVATE ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
      endforeach()
      target_include_directories(strata_mmq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                                 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
      target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
      if(MSVC)
        target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Xcompiler=/bigobj>)
      endif()
      target_compile_definitions(strata_mmq PUBLIC GGML_USE_HIP)
      if(CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1201")
        target_compile_definitions(strata_mmq PUBLIC RDNA4)
      elseif(CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1031" OR CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1030")
        target_compile_definitions(strata_mmq PUBLIC RDNA2)
      endif()
      target_link_libraries(strata_mmq PUBLIC ggml-base ${ROCM_LIBRARY}/amdhip64.lib)
      strata_apply_hip_flags(strata_mmq)
      target_link_libraries(strata_prefill PUBLIC strata_mmq)
      target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1)
    else()
      message(STATUS "Strata: MMQ prefill path disabled (llama.cpp .cu files need hipification)")
    endif()
  endif()
  if(STRATA_ENABLE_HIP)
    add_executable(native_expert_parity src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE ${ROCM_INCLUDE})
    target_link_libraries(native_expert_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${ROCM_LIBRARY}/amdhip64.lib)
  endif()
endif()

if(STRATA_ENABLE_HIP AND STRATA_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/hit_cpu_order_parity.cu)
  add_executable(hit_cpu_order_parity bench/micro/hit_cpu_order_parity.cu)
  target_link_libraries(hit_cpu_order_parity PRIVATE strata_kernels strata_kernels_cpu ${ROCM_LIBRARY}/amdhip64.lib)
elseif(STRATA_ENABLE_HIP AND STRATA_BUILD_TESTS)
  message(STATUS "strata: bench/micro/hit_cpu_order_parity.cu absent - that parity test is NOT registered")
endif()

if(STRATA_BUILD_TESTS)
  # Plan v0.3 P6: one expert applied to several tokens, bitwise vs the single-token kernel (synthetic, CPU only).
  # plan v0.3 P5: the device dequantizers vs the artifact's CPU dequantizers on the real shards (not a ctest: needs
  # the 38 GB model; run it by hand or from tools/gpu_session.py)
  add_executable(dequant_bf16_test src/kernels/dequant_bf16_test.cpp)
  target_link_libraries(dequant_bf16_test PRIVATE strata_kernels strata_artifact ${ROCM_LIBRARY}/amdhip64.lib)

  # FP4 reference dequant (MXFP4/NVFP4) vs ggml's own dequantize_row_mxfp4/dequantize_row_nvfp4, BITWISE.
  # The project's rule is that a reference validated against a copy of itself proves nothing, and these
  # two formats have no vec_dot kernel in any ggml backend - so this scalar path is the reference the
  # future __dp4a path is measured against, which makes it load-bearing.  ggml-base exports both symbols.
  # Host-only (no GPU, no model), so it is a plain ctest:  the 256-encoding scale sweeps and the
  # nibble-permutation cases are the parts a random differential alone would let through.
  if(STRATA_NATIVE_EXPERTS)
    add_executable(dequant_fp4_test src/kernels/dequant_fp4_test.cpp)
    target_link_libraries(dequant_fp4_test PRIVATE strata_artifact ggml-base)
    add_test(NAME dequant_fp4_test COMMAND dequant_fp4_test)
  else()
    message(STATUS "strata: STRATA_NATIVE_EXPERTS off - the FP4 reference-dequant differential test is NOT registered")
  endif()

  add_executable(expert_multi_test src/kernels/cpu/expert_multi_test.cpp)
  target_link_libraries(expert_multi_test PRIVATE strata_kernels_cpu)
  add_test(NAME expert_multi_test COMMAND expert_multi_test)

  add_executable(expert_parity src/kernels/cpu/expert_parity.cpp)
  target_link_libraries(expert_parity PRIVATE strata_kernels_cpu)
  # The pack is a REQUIRED input, not an optional one: this test compares against a real expert blob, and a
  # missing file would otherwise be reported as a pass by a test that skipped everything.
  add_test(NAME expert_parity COMMAND expert_parity --selftest
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(pool_test src/kernels/cpu/pool_test.cpp)
  target_link_libraries(pool_test PRIVATE strata_kernels_cpu)
  add_test(NAME pool_test COMMAND pool_test --selftest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  # issue #29: tiny batches with sleeping workers, as on a big-VRAM card; a hang fails it
  add_executable(pool_stress src/kernels/cpu/pool_stress.cpp)
  target_link_libraries(pool_stress PRIVATE strata_kernels_cpu)
  add_test(NAME pool_stress COMMAND pool_stress 20)

  # contract E7 / TODO P3 #29: per-layer admission must give each layer its OWN slot range (the startup
  # verify_slot read-back compared two experts because open_sized zeroed the per-layer counters).
  add_executable(expert_cache_per_layer_test src/core/expert_cache_per_layer_test.cpp)
  target_link_libraries(expert_cache_per_layer_test PRIVATE strata_engine ${ROCM_LIBRARY}/amdhip64.lib)
  target_include_directories(expert_cache_per_layer_test PRIVATE ${ROCM_INCLUDE})
  add_test(NAME expert_cache_per_layer_test COMMAND expert_cache_per_layer_test)

  # M2: the deepseek2/GLM native serving gate.  Reads the local GLM pack + GGUF and skips when they are absent;
  # it is the only check that the 3-D `attn_k_b`/`attn_v_b` are actually attached.
  add_executable(native_dense_3d_test src/core/native_dense_3d_test.cpp)
  target_link_libraries(native_dense_3d_test PRIVATE strata_engine ${ROCM_LIBRARY}/amdhip64.lib)
  target_include_directories(native_dense_3d_test PRIVATE ${ROCM_INCLUDE})
  add_test(NAME native_dense_3d_test COMMAND native_dense_3d_test)
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # Guarded like the default above: the per-file *_test executables above are all in-tree and build
  # fine, but tests/ ships separately, and an unguarded add_subdirectory of a missing directory makes
  # -DSTRATA_BUILD_TESTS=ON fail to CONFIGURE - which silently disables every other test in the project
  # rather than just this one. An absent registration is visible; a failed configure is not.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
  else()
    message(STATUS "strata: tests/ absent - the tests/ subdirectory is NOT added (in-tree tests still register)")
  endif()
endif()

