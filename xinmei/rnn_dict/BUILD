package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl",
     "tf_copts",
     "if_android",
     "if_not_mobile")

LINKER_SCRIPT = ":android_version_script.lds"

cc_binary(
    name = "librnn_dict.so",
    srcs = glob(["utils/*.h"])
        + [
            "utils.h",
            "rnn_dict.h",
            "rnn_dict.cc",
            "//xinmei/quantized_graph_loader:quantized_graph_loader_src",
        ]
        + if_android([
            "abort_wrapper.cc",
            "rnn_dict_wrap.cxx",
            "//tensorflow/core:android_op_registrations_and_gradients",
            ":kernels_src",
            # ":missing_impls",
          ])
        + if_not_mobile(glob([
            "java_home/include/**/*.h",
          ])),
    copts = tf_copts() + [
            "-fexceptions",
            "-DSUPPORT_SELECTIVE_REGISTRATION",
          ]
        + if_not_mobile([
            "-Ixinmei/rnn_dict/java_home/include -Ixinmei/rnn_dict/java_home/include/darwin -Ixinmei/rnn_dict/java_home/include/linux",
          ]),
    linkopts = ["-lm"] + if_android([
        "-landroid",
        "-llog",
        "-lz",
        "-z defs",
        "-Wl,--version-script", LINKER_SCRIPT,
        "-Wl,--wrap=abort",
    ]),
    linkshared = 1,
    deps = [
            "//xinmei/quantized_graph_loader:quantized_graph_proto_cc",
        ] + if_android([
            "@gemmlowp//:gemmlowp",
            "//tensorflow/core:android_tensorflow_lib_selective_registration",
            LINKER_SCRIPT])
        + if_not_mobile([
            "//tensorflow/cc:cc_ops",
            "//tensorflow/core:framework",
            "//tensorflow/core:tensorflow",
        ]),
)

filegroup(
    name = "missing_impls",
    srcs = [
    ],
)

filegroup(
    name = "kernels_src",
    srcs = [
        "//tensorflow/core/kernels:no_op.h",
        "//tensorflow/core/kernels:no_op.cc",
    ],
)

cc_binary(
    name = "libdeeptype.so",
    srcs = [
        "deeptype.h",
        "deeptype.cc",
        "deeptype_jni.cc",
        # "deeptype_test.cc",
        "deeptype_v0.1_val.h",
        "utils.h",
        "//tensorflow/core:android_op_registrations_and_gradients",
        ":android_core_ops_rs_training",
        # "abort_wrapper.cc",
        # ":missed_impls_from_lib",
    ],
    copts = tf_copts() + [
        "-DSUPPORT_SELECTIVE_REGISTRATION",
        "-fexceptions",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    linkopts = [
        # "-pie",
        "-landroid",
        "-llog",
        # "-ljnigraphics",
        # "-ldl",
        "-lz",
        "-lm",
        "-z defs",
        # "-Wl,--wrap=abort",
        "-Wl,--gc-sections",
        "-Wl,--version-script",
        LINKER_SCRIPT,
    ],
    linkshared = 1,
    deps = [
        "@gemmlowp//:gemmlowp",
        "//tensorflow/core:android_tensorflow_lib_selective_registration",
        # "//tensorflow/core:android_deeptype_lib",
        # "//tensorflow/core/kernels:android_tensorflow_kernels_deeptype",
        LINKER_SCRIPT,
    ],
)

cc_binary(
    name = "deeptype_test",
    srcs = [
        "deeptype.h",
        "deeptype.cc",
        "deeptype_test.cc",
        "deeptype_v0.1_val.h",
        "utils.h",
        "//tensorflow/core:android_op_registrations_and_gradients",
        ":android_core_ops_rs_training",
        # "abort_wrapper.cc",
        # ":missed_impls_from_lib",
    ],
    copts = tf_copts() + [
        "-DSUPPORT_SELECTIVE_REGISTRATION",
        "-fexceptions",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    linkopts = [
        "-pie",
        "-landroid",
        "-llog",
        "-ljnigraphics",
        "-ldl",
        "-lz",
        "-lm",
        "-z defs",
        # "-Wl,--wrap=abort",
        "-Wl,--gc-sections",
        "-Wl,--version-script",
        LINKER_SCRIPT,
    ],
    linkstatic = 1,
    # linkshared = 1,
    deps = [
        "@gemmlowp//:gemmlowp",
        "//tensorflow/core:android_tensorflow_lib_selective_registration",
        # "//tensorflow/core:android_deeptype_lib",
        # "//tensorflow/core/kernels:android_tensorflow_kernels_deeptype",
        LINKER_SCRIPT,
    ],
)

filegroup(
    name = "missed_impls_from_lib",
    srcs = [
        "//tensorflow/core/kernels:cwise_op_floor_mod.cc",
        "//tensorflow/core/kernels:sparse_xent_op.h",
        "//tensorflow/core/kernels:sparse_xent_op.cc",
        "//tensorflow/core/kernels:l2loss_op.h",
        "//tensorflow/core/kernels:l2loss_op.cc",
        "//tensorflow/core/kernels:scatter_functor.h",
        "//tensorflow/core/kernels:scatter_op.cc",
        # "//tensorflow/core/kernels:save_restore_tensor.h",
        # "//tensorflow/core/kernels:save_restore_tensor.cc",
        # "//tensorflow/core/kernels:save_restore_v2_ops.cc",
    ]
)

filegroup(
    name = "android_core_ops_rs_training",
    srcs = [
        "//tensorflow/core/kernels:tile_ops_impl.h",
        "//tensorflow/core/kernels:tile_ops_cpu_impl.h",
        "//tensorflow/core/kernels:tile_ops_cpu_impl_1.cc",
        "//tensorflow/core/kernels:tile_ops_cpu_impl_2.cc",
        "//tensorflow/core/kernels:tile_ops_cpu_impl_3.cc",
        "//tensorflow/core/kernels:tile_ops_cpu_impl_4.cc",
        "//tensorflow/core/kernels:tile_ops_cpu_impl_5.cc",
        "//tensorflow/core/kernels:tile_ops_cpu_impl_6.cc",
        "//tensorflow/core/kernels:tile_ops_cpu_impl_7.cc",
        "//tensorflow/core/kernels:strided_slice_op_impl.h",
        "//tensorflow/core/kernels:strided_slice_op_inst_0.cc",
        "//tensorflow/core/kernels:strided_slice_op_inst_1.cc",
        "//tensorflow/core/kernels:strided_slice_op_inst_2.cc",
        "//tensorflow/core/kernels:strided_slice_op_inst_3.cc",
        "//tensorflow/core/kernels:strided_slice_op_inst_4.cc",
        "//tensorflow/core/kernels:strided_slice_op_inst_5.cc",
        "//tensorflow/core/kernels:strided_slice_op_inst_6.cc",
        "//tensorflow/core/kernels:strided_slice_op_inst_7.cc",
        "//tensorflow/core/kernels:slice_op_cpu_impl.h",
        "//tensorflow/core/kernels:slice_op_cpu_impl_1.cc",
        "//tensorflow/core/kernels:slice_op_cpu_impl_2.cc",
        "//tensorflow/core/kernels:slice_op_cpu_impl_3.cc",
        "//tensorflow/core/kernels:slice_op_cpu_impl_4.cc",
        "//tensorflow/core/kernels:slice_op_cpu_impl_5.cc",
        "//tensorflow/core/kernels:slice_op_cpu_impl_6.cc",
        "//tensorflow/core/kernels:slice_op_cpu_impl_7.cc",
        "//tensorflow/core/kernels:function_ops.cc",
        "//tensorflow/core/kernels:cwise_op_sigmoid.cc",
        "//tensorflow/core/kernels:no_op.h",
        "//tensorflow/core/kernels:no_op.cc",
        "//tensorflow/core/kernels:assign_op.h",
        "//tensorflow/core/kernels:bias_op.cc",
        "//tensorflow/core/kernels:bias_op.h",
        "//tensorflow/core/kernels:bias_op_gpu.h",
        "//tensorflow/core/kernels:bounds_check.h",
        "//tensorflow/core/kernels:concat_lib.h",
        "//tensorflow/core/kernels:concat_lib_cpu.cc",
        "//tensorflow/core/kernels:concat_lib_cpu.h",
        "//tensorflow/core/kernels:concat_op.cc",
        "//tensorflow/core/kernels:constant_op.cc",
        "//tensorflow/core/kernels:constant_op.h",
        "//tensorflow/core/kernels:control_flow_ops.cc",
        "//tensorflow/core/kernels:control_flow_ops.h",
        "//tensorflow/core/kernels:cuda_device_array.h",
        "//tensorflow/core/kernels:cuda_device_array_gpu.h",
        "//tensorflow/core/kernels:cwise_op_add_1.cc",
        "//tensorflow/core/kernels:cwise_op_greater.cc",
        "//tensorflow/core/kernels:cwise_op_floor_mod.cc",
        "//tensorflow/core/kernels:cwise_op_floor.cc",
        "//tensorflow/core/kernels:cwise_op_sqrt.cc",
        "//tensorflow/core/kernels:cwise_op_less_equal.cc",
        "//tensorflow/core/kernels:cwise_op_mul_1.cc",
        "//tensorflow/core/kernels:cwise_op_select.cc",
        "//tensorflow/core/kernels:cwise_op_neg.cc",
        "//tensorflow/core/kernels:cwise_op_sub.cc",
        "//tensorflow/core/kernels:cwise_op_minimum.cc",
        "//tensorflow/core/kernels:cwise_op_div.cc",
        "//tensorflow/core/kernels:cwise_op_tanh.cc",
        "//tensorflow/core/kernels:cwise_ops.h",
        "//tensorflow/core/kernels:cwise_ops_common.cc",
        "//tensorflow/core/kernels:cwise_ops_common.h",
        "//tensorflow/core/kernels:cwise_ops_gradients.h",
        "//tensorflow/core/kernels:cwise_ops_sycl_common.h",
        "//tensorflow/core/kernels:dense_update_ops.cc",
        "//tensorflow/core/kernels:dense_update_ops.h",
        "//tensorflow/core/kernels:fill_functor.cc",
        "//tensorflow/core/kernels:fill_functor.h",
        "//tensorflow/core/kernels:gather_functor.cc",
        "//tensorflow/core/kernels:gather_functor.h",
        "//tensorflow/core/kernels:gather_op.cc",
        "//tensorflow/core/kernels:identity_op.cc",
        "//tensorflow/core/kernels:identity_op.h",
        "//tensorflow/core/kernels:matmul_op.cc",
        "//tensorflow/core/kernels:matmul_op.h",
        "//tensorflow/core/kernels:mkl_matmul_op.cc",
        "//tensorflow/core/kernels:ops_util.h",
        "//tensorflow/core/kernels:pack_op.cc",
        "//tensorflow/core/kernels:reduction_ops.h",
        "//tensorflow/core/kernels:reduction_ops_common.cc",
        "//tensorflow/core/kernels:reduction_ops_common.h",
        "//tensorflow/core/kernels:reduction_ops_max.cc",
        "//tensorflow/core/kernels:reduction_ops_min.cc",
        "//tensorflow/core/kernels:reduction_ops_sum.cc",
        "//tensorflow/core/kernels:reshape_op.cc",
        "//tensorflow/core/kernels:reshape_op.h",
        "//tensorflow/core/kernels:sendrecv_ops.cc",
        "//tensorflow/core/kernels:sendrecv_ops.h",
        "//tensorflow/core/kernels:sequence_ops.cc",
        "//tensorflow/core/kernels:shape_ops.cc",
        "//tensorflow/core/kernels:shape_ops.h",
        "//tensorflow/core/kernels:slice_op.cc",
        "//tensorflow/core/kernels:slice_op.h",
        "//tensorflow/core/kernels:softmax_op.cc",
        "//tensorflow/core/kernels:softmax_op.h",
        "//tensorflow/core/kernels:softmax_op_functor.h",
        "//tensorflow/core/kernels:split_lib.h",
        "//tensorflow/core/kernels:split_lib_cpu.cc",
        "//tensorflow/core/kernels:split_op.cc",
        "//tensorflow/core/kernels:split_v_op.cc",
        "//tensorflow/core/kernels:strided_slice_op.cc",
        "//tensorflow/core/kernels:strided_slice_op.h",
        "//tensorflow/core/kernels:tile_ops.cc",
        # "//tensorflow/core/kernels:tile_functor.h",
        # "//tensorflow/core/kernels:tile_functor_cpu.cc",
        "//tensorflow/core/kernels:transpose_functor.h",
        "//tensorflow/core/kernels:transpose_functor_cpu.cc",
        "//tensorflow/core/kernels:transpose_op.cc",
        "//tensorflow/core/kernels:transpose_op.h",
        "//tensorflow/core/kernels:unpack_op.cc",
        "//tensorflow/core/kernels:variable_ops.cc",
        "//tensorflow/core/kernels:variable_ops.h",
        "//tensorflow/core/kernels:bcast_ops.cc",
        "//tensorflow/core/kernels:random_op.h",
        "//tensorflow/core/kernels:random_op.cc",
        "//tensorflow/core/kernels:l2loss_op.h",
        "//tensorflow/core/kernels:l2loss_op.cc",
        "//tensorflow/core/kernels:scatter_functor.h",
        "//tensorflow/core/kernels:scatter_op.cc",
        "//tensorflow/core/kernels:sparse_xent_op.h",
        "//tensorflow/core/kernels:sparse_xent_op.cc",
        "//tensorflow/core/kernels:aggregate_ops.h",
        "//tensorflow/core/kernels:aggregate_ops_cpu.h",
        "//tensorflow/core/kernels:aggregate_ops.cc",
        "//tensorflow/core/kernels:training_op_helpers.h",
        "//tensorflow/core/kernels:training_op_helpers.cc",
        "//tensorflow/core/kernels:training_ops.h",
        "//tensorflow/core/kernels:training_ops.cc",
        "//tensorflow/core/kernels:topk_op.cc",
        # "//tensorflow/core/kernels:topk_op.h",
        "//tensorflow/core/kernels:save_restore_tensor.h",
        "//tensorflow/core/kernels:save_restore_tensor.cc",
        "//tensorflow/core/kernels:save_restore_v2_ops.cc",
    ],
)