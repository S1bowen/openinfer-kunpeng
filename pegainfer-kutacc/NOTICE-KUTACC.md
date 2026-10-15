# NOTICE — vendored KuTACC

This crate contains source code copied verbatim from Huawei's **KuTACC**
(Kunpeng Unified Transformer Accelerated Library), 25.1.RC1 era.

```
Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under a modified version of the MIT license.
See LICENSE-KUTACC (unmodified upstream text) in this crate.
```

Source: `/home/share/fengguangnan/sibow/mlsysu/SGLang_920F/third_party/kutacc/source`
at commit `1502ca3f1ef1c1349ca499008214a3fe204bc356`.
See `csrc/KUTACC_PROVENANCE.md` for the exact file set and the delta against
upstream's own `SOURCE_SNAPSHOT.sha256`.

## Additional conditions that bind this repository

`LICENSE-KUTACC` is MIT **plus** the following, reproduced here because
condition (c) requires these restrictions to appear as an enforceable provision
in anything governing use or distribution of the software:

> a. You shall not use the Software or its derivative works in systems with
>    processors other than Kunpeng processors.
> b. You shall not make any use of the Software or its derivative works to
>    develop or distribute any software for use in systems with processors
>    other than Kunpeng processors.
> c. You must include the same restrictions in Section a and b as an
>    enforceable provision in any agreement (e.g., license agreement, terms of
>    use, etc.) governing the use and/or distribution of the Software or
>    Derivative Works.
>
> "Kunpeng processors" mean the computing processor chipsets branded with
> "Kunpeng", "鲲鹏" or other brands owned or controlled by Huawei.

Practically: **any distribution containing this crate may only target Kunpeng
processors**, and anyone redistributing it must carry these terms forward.
PegaInfer's own code remains Apache-2.0; the restriction attaches to the
vendored KuTACC portion. Do not ship a PegaInfer build that includes this crate
to non-Kunpeng users without resolving this.

## What is ours

Everything outside `csrc/kutacc/` — the build script, the `extern "C"` shims in
`csrc/shim/`, and the Rust API in `src/` — is PegaInfer code under Apache-2.0.
`csrc/kutacc/` additionally holds three files this project added, none of which
modify upstream code:

- `PROVENANCE.md`
- `verify_against_snapshot.sh`
- `SOURCE_SNAPSHOT.sha256` (an unmodified copy of upstream's manifest)
