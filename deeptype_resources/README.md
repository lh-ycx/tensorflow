# DeepType 编译日志

## 编译环境

```
Ubuntu 16.04
OpenJDK 1.8.0_222
Android SDK build-tools 26.0.0
Android SDK platform Android-26
Android NDK r14b
bazel 0.19.2
TensorFlow r1.13
```

## 编译命令

```shell
bazel build xinmei/rnn_dict:deeptype_test --copt=-DSELECTIVE_REGISTRATION  --crosstool_top=//external:android/crosstool    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain    --cpu=armeabi-v7a --verbose_failures --copt=-Wno-c++11-narrowing --cxxopt=--std=c++11
```

## 编译

针对“ops_to_register.h”缺失的问题

```shell
bazel build tensorflow/python/tools:print_selective_registration_header
```

```bash
bazel-bin/tensorflow/python/tools/print_selective_registration_header \
    --graphs=path/to/graph.pb > ops_to_register.h
```



<https://docs.bazel.build/versions/master/android-ndk.html>

## WORKSPACE

在WORKSPACE中增加如下内容，可能需要注释掉一些android_workspace()。

```bash
load("//tensorflow:workspace.bzl", "tf_workspace")

# please set ANDROID_HOME env to android sdk root
android_sdk_repository(
    name = "androidsdk",
    api_level = 26,
    # Ensure that you have the build_tools_version below installed in the
    # SDK manager as it updates periodically.
    build_tools_version = "26.0.0",
)

# please set ANDROID_NDK_HOME env to android ndk root
android_ndk_repository(
    name="androidndk",
    # This needs to be 14 or higher to compile TensorFlow.
    # Note that the NDK version is not the API level.
    api_level=14)


# Please add all new TensorFlow dependencies in workspace.bzl.
tf_workspace()
```

<https://github.com/tensorflow/tensorflow/blob/r1.13/WORKSPACE>

## this rule is missing dependency declarations for the following files included by xxx.cc

```bash
ERROR: /home/pku/Documents/deeptype/tensorflow/xinmei/rnn_dict/BUILD:118:1: undeclared inclusion(s) in rule '//xinmei/rnn_dict:deeptype_test':
this rule is missing dependency declarations for the following files included by 'tensorflow/core/kernels/dense_update_ops.cc':
  'tensorflow/core/kernels/dense_update_functor.h'
Target //xinmei/rnn_dict:deeptype_test failed to build
```

修改tensorflow/core/kernels/BUILD

```bash
tf_kernel_library(
    name = "dense_update_ops",
    prefix = "dense_update_ops",
    deps = STATE_DEPS + [":dense_update_functor"] + [":headers"],
)
```

<https://cloud.tencent.com/developer/article/1379773>

## Void* 运算出错

```c
xinmei/rnn_dict/deeptype.cc:121:3: error: use of undeclared identifier 'printGraphInfo'

  printGraphInfo(meta_graph_def.graph_def());

  ^

xinmei/rnn_dict/deeptype.cc:380:25: error: arithmetic on a pointer to void

​      memcpy(logits_ptr + i * logits_chunk_size, temp_ptr, logits_chunk_size);

​             \~~~~~~~~~~ ^

xinmei/rnn_dict/deeptype.cc:395:21: error: arithmetic on a pointer to void

​      memcpy(sm_ptr + i * sm_chunk_size, temp_ptr, sm_chunk_size);

​             \~~~~~~ ^
```

强转为long*