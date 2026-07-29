#!/usr/bin/env bash
# CTorch 公共头文件 ABI 变更审计脚本
# 用途：在 CI 或本地运行，检测当前工作树与基线分支之间的公共头文件变化，
#       提醒维护者确认是否已更新 ABI_POLICY.md、static_assert 与 kernel 注册表。
#
# 用法：
#   ./scripts/abi_audit.sh [base_branch]
# 默认基线分支为 origin/main 或 main。

set -euo pipefail

BASE_BRANCH="${1:-}"
if [[ -z "$BASE_BRANCH" ]]; then
    if git rev-parse --verify origin/main >/dev/null 2>&1; then
        BASE_BRANCH="origin/main"
    elif git rev-parse --verify main >/dev/null 2>&1; then
        BASE_BRANCH="main"
    else
        echo "[ABI_AUDIT] 无法确定基线分支，请显式传入：./scripts/abi_audit.sh <base_branch>"
        exit 1
    fi
fi

PUBLIC_HEADERS=(
    "include/Storage.h"
    "include/Tensor.h"
    "include/Ctools.h"
    "include/CtorchScheduler.h"
    "include/AutoGrad/Node.h"
    "src/kernels/kernels.h"
)

CHANGED_FILES=$(git diff --name-only "$BASE_BRANCH" -- "${PUBLIC_HEADERS[@]}" 2>/dev/null || true)

if [[ -z "$CHANGED_FILES" ]]; then
    echo "[ABI_AUDIT] 公共头文件无变化。"
    exit 0
fi

echo "[ABI_AUDIT] 检测到以下公共头文件发生变化（与 $BASE_BRANCH 对比）："
echo "$CHANGED_FILES" | sed 's/^/  - /'
echo ""
echo "[ABI_AUDIT] 请确认："
echo "  1. 是否已在 ABI_POLICY.md 第 4 节记录本次 ABI 破坏？"
echo "  2. 是否已更新 CtorchScheduler.h 中的 static_assert？"
echo "  3. 是否已同步所有 backend kernel 注册表？"
echo "  4. 是否已补充/更新相关单元测试？"
echo ""
echo "[ABI_AUDIT] 若以上均已确认，可在 CI 中标记为通过。"

# 非 CI 环境下退出码为 1，用于拦截未审查的变更；CI 中可配置为 warning。
exit 1
