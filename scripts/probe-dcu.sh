#!/usr/bin/env bash
# CTorch × 海光 DCU 环境探针
# 用途：单次运行收集 DTK24.04 / DCU 硬件 / 编译器 / 网络拓扑 / 存储配额 /
#      CTorch 编译环境的全量快照，作为 DCU 适配工程的第 0 步基线。
#
# 用法：
#   ./scripts/probe-dcu.sh                       # 落盘到 ./dcu-probe-<timestamp>.md
#   ./scripts/probe-dcu.sh /path/to/report.md    # 落盘到指定路径
#   bash <(curl -sL <raw-url>/probe-dcu.sh)     # 一行远程执行
#
# 设计原则：
#   1. 任何探针失败不终止脚本（除了最基本的 timestamp / 输出目录）
#   2. 不需要 root 权限；遇到 sudo 命令直接 skip
#   3. 命令不存在时输出 [N/A] 而非 error
#   4. 输出到 stdout 的同时落盘 markdown 报告
#   5. 包含 CTorch 当前 commit / branch / dirty 状态，方便事后回溯

set -o pipefail  # 不开 -e / -u：探针必须容错，未定义环境变量也允许

# ---------- 参数与初始化 ----------
OUTPUT_FILE="${1:-}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
TS_HUMAN="$(date '+%Y-%m-%d %H:%M:%S %z')"
HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
HOST_FULL="$(hostname -f 2>/dev/null || hostname)"

if [[ -z "${OUTPUT_FILE}" ]]; then
    OUTPUT_FILE="$(pwd)/dcu-probe-${HOST_SHORT}-${TS}.md"
fi

# 临时缓冲文件
BUF="$(mktemp -t probe-dcu.XXXXXX)"
trap 'rm -f "${BUF}"' EXIT

# ---------- 输出工具 ----------
# section <title>  : 输出 ### 块头
section() {
    printf '\n===== %s =====\n' "$1" | tee -a "${BUF}"
}

# kv <key> <value> : 输出 key: value
kv() {
    printf '  %-26s : %s\n' "$1" "$2" | tee -a "${BUF}"
}

# blank : 输出空行
blank() {
    printf '\n' | tee -a "${BUF}"
}

# run <label> <cmd...> : 执行命令，输出 stdout，标注 [OK]/[FAIL]/[N/A]
run() {
    local label="$1"; shift
    local cmd_desc="$*"
    printf -- '--- %s ---\n' "${label}" | tee -a "${BUF}"
    printf '  $ %s\n' "${cmd_desc}" | tee -a "${BUF}"
    if ! command -v "${1%% *}" >/dev/null 2>&1; then
        printf '  [N/A] command not found: %s\n' "${1%% *}" | tee -a "${BUF}"
        return 0
    fi
    local out rc=0
    out="$("$@" 2>&1)" || rc=$?
    if [[ ${rc} -eq 0 ]]; then
        printf '  [OK] exit=0\n' | tee -a "${BUF}"
    else
        printf '  [FAIL] exit=%d\n' "${rc}" | tee -a "${BUF}"
    fi
    if [[ -n "${out}" ]]; then
        printf '%s\n' "${out}" | sed 's/^/    /' | tee -a "${BUF}"
    fi
    return 0
}

# header <title>  : 输出 markdown 风格 ## 块头（仅在最终落盘时使用，由 collect 重写）

# ---------- 开始 ----------
{
    printf '# CTorch × 海光 DCU 环境探针报告\n\n'
    printf -- '- **生成时间**：%s\n' "${TS_HUMAN}"
    printf -- '- **主机名**：%s (%s)\n' "${HOST_FULL}" "${HOST_SHORT}"
    printf -- '- **脚本版本**：probe-dcu.sh 2026-08-08\n'
    printf -- '- **用途**：C3 JIT × DCU 适配工程第 0 步基线\n\n'
    printf '> 输出格式：每节包含探针目的、命令、状态（OK / FAIL / N/A）、原始输出。\n'
    printf '> 任何 FAIL 或 N/A 项都需要人工评估对 DCU 适配的影响。\n\n'
} > "${BUF}"

# ---------- 1. 系统信息 ----------
section "1. 系统信息"
kv "内核"      "$(uname -srm 2>/dev/null || echo N/A)"
kv "发行版"    "$(cat /etc/os-release 2>/dev/null | grep -E '^(NAME|VERSION)=' | tr '\n' ' ' || echo N/A)"
kv "架构"      "$(uname -m 2>/dev/null || echo N/A)"
kv "CPU 型号"  "$(lscpu 2>/dev/null | awk -F: '/Model name/{print $2; exit}' | sed 's/^ *//' || echo N/A)"
kv "CPU 核数"  "$(nproc 2>/dev/null || echo N/A)"
kv "内存"      "$(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo N/A)"
kv "已运行"    "$(uptime -p 2>/dev/null || uptime || echo N/A)"
kv "当前用户"  "$(whoami 2>/dev/null || echo N/A)"
kv "UID/GID"   "$(id -u 2>/dev/null)/$(id -g 2>/dev/null)"

# ---------- 2. DCU 硬件 ----------
section "2. DCU 硬件"
run "rocm-smi"           rocm-smi
run "rocm-smi -d"        rocm-smi -d
run "rocm-smi --topo"    rocm-smi --topo
run "rocm-smi --showmeminfo" rocm-smi --showmeminfo vram 2>/dev/null || true
run "rocminfo"           rocminfo
run "hipconfig"          hipconfig
run "hipconfig --version" hipconfig --version
run "hipconfig --path"   hipconfig --path
run "lspci (AMD only)"   bash -c "lspci 2>/dev/null | grep -iE 'amd|ati|hygon' || echo 'lspci empty or unavailable'"

# ---------- 3. DTK24.04 软件栈 ----------
section "3. DTK24.04 软件栈"
run "hipcc --version"   hipcc --version
run "hipconfig --cpp_config" hipconfig --cpp_config
run "DTK env vars"      bash -c "env | grep -iE 'rocm|hip|hcc|dtk' || echo '(no env vars)'"
run "MIOpen version"    bash -c "find /opt -name 'MIOpen.h' 2>/dev/null | head -3; miopen --version 2>/dev/null || echo 'miopen CLI not found'"
run "rocBLAS version"   bash -c "find /opt -name 'rocblas.h' 2>/dev/null | head -3; rocblas-bench --version 2>/dev/null || echo 'rocblas-bench not found'"
run "hipBLASLt version" bash -c "find /opt -name 'hipblaslt.h' 2>/dev/null | head -3; hipblaslt-bench --version 2>/dev/null || echo 'hipblaslt-bench not found'"
run "RCCL version"      bash -c "rccl-Info 2>/dev/null || find /opt -name 'rccl.h' 2>/dev/null | head -3 || echo 'rccl not found'"
run "DTK install root"  bash -c "ls -d /opt/dtk* /opt/rocm* /opt/hygon* 2>/dev/null || echo 'no standard DTK install root found'"
run "DTK module (Lmod)" bash -c "module list 2>/dev/null | grep -iE 'rocm|hip|dtk' || echo 'Lmod not available or no DTK module loaded'"

# ---------- 4. 编译器与构建工具 ----------
section "4. 编译器与构建工具"
run "clang --version"   clang --version
run "gcc --version"     gcc --version
run "g++ --version"     g++ --version
run "hipcc (recheck)"   hipcc --version
run "cmake --version"   cmake --version
run "ninja --version"   ninja --version
run "make --version"    make --version
run "python3 --version" python3 --version
run "git --version"     git --version
run "nasm/yasm"         bash -c "nasm --version 2>/dev/null || yasm --version 2>/dev/null || echo 'no nasm/yasm'"

# 检查 __builtin_amdgcn_* 内建支持（DCU 兼容性关键）
run "amdgcn builtin test" bash -c "cat > /tmp/_builtin_test.c <<'EOF'
#include <stdio.h>
int main(){
  int x = 0;
  // 几个常见的 amdgcn 内建
  #ifdef __HIPCC__
  x = __builtin_amdgcn_workitem_id_x();
  #endif
  printf(\"amdgcn builtin available: %d\\n\", x);
  return 0;
}
EOF
hipcc /tmp/_builtin_test.c -o /tmp/_builtin_test 2>&1 && /tmp/_builtin_test || echo 'hipcc compile failed'"
rm -f /tmp/_builtin_test.c /tmp/_builtin_test 2>/dev/null

# ---------- 5. 网络与拓扑 ----------
section "5. 网络与拓扑"
run "ip address"         ip a
run "ip route"           ip r
run "ibstat (InfiniBand)" ibstat
run "ibv_devinfo"        ibv_devinfo
run "numactl --hardware" numactl --hardware
run "lscpu (cache)"      lscpu

# ---------- 6. 存储与配额 ----------
section "6. 存储与配额"
run "df -h"              df -h
run "mount (lustre/nfs)" bash -c "mount | grep -iE 'lustre|nfs|gfs|ceph' || echo 'no lustre/nfs/gfs/ceph mounts'"
run "/tmp size"          df -h /tmp
run "user quota"         bash -c "quota -u $(whoami) 2>&1 || echo 'quota command not available or no quota set'"
run "project quota"      bash -c "quota -p $(whoami) 2>&1 || echo 'project quota not available'"
run "inode usage"        bash -c "df -i $(pwd) 2>/dev/null | tail -1 || echo 'inode info unavailable'"

# ---------- 7. 超算平台特有 ----------
section "7. 超算平台特有"
run "Slurm sinfo"        sinfo 2>&1 | head -20
run "Slurm squeue"       squeue -u "$(whoami)" 2>&1 | head -10
run "Slurm scontrol"     bash -c "scontrol show config 2>&1 | head -10 || echo 'scontrol unavailable'"
run "Lmod module list"   module list 2>&1
run "Lmod module avail dtk" module avail dtk 2>&1 | head -20
run "DTK env probe"      bash -c "echo DTK_HOME=${DTK_HOME:-<unset>}; echo ROCM_PATH=${ROCM_PATH:-<unset>}; echo HIP_PATH=${HIP_PATH:-<unset>}; echo HCC_HOME=${HCC_HOME:-<unset>}"

# ---------- 8. CTorch 编译环境 ----------
section "8. CTorch 编译环境"
# 自动识别仓库根（向上找 .git / CMakeLists.txt）
find_repo_root() {
    local d
    d="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
    if [[ -d "${d}/.git" ]] || [[ -f "${d}/CMakeLists.txt" ]]; then
        printf '%s' "${d}"
        return 0
    fi
    d="$(pwd)"
    while [[ "${d}" != "/" ]]; do
        if [[ -d "${d}/.git" ]] || [[ -f "${d}/CMakeLists.txt" ]]; then
            printf '%s' "${d}"
            return 0
        fi
        d="$(dirname "${d}")"
    done
    return 1
}

REPO_ROOT="$(find_repo_root 2>/dev/null || echo N/A)"
kv "仓库根（自动检测）" "${REPO_ROOT}"
if [[ "${REPO_ROOT}" != "N/A" ]] && cd "${REPO_ROOT}" 2>/dev/null; then
    run "git status"        git status --short --branch
    run "git HEAD"          git rev-parse HEAD
    run "git branch"         git rev-parse --abbrev-ref HEAD
    run "git describe"       git describe --tags --always --dirty 2>/dev/null || echo "(no tags)"
    run "git log (5)"        git log --oneline -5
    run "git diff stat"      git diff --stat
    run "CMake version req"  bash -c "grep -E 'cmake_minimum_required' CMakeLists.txt | head -1"
    run "WITH_DCU flag"      bash -c "grep -nE 'WITH_DCU|WITH_HIP|WITH_ROCM' CMakeLists.txt || echo 'no WITH_DCU flag (尚未集成)'"
    run "existing build dir" bash -c "ls -ld build*/ 2>/dev/null | head -5 || echo '(no build dir)'"
    run "include dir"        bash -c "ls include/ 2>/dev/null | head -20"
    run "C3 dir"             bash -c "ls include/C3 src/C3 2>/dev/null | head -20 || echo '(C3 目录布局待确认)'"
    run "test dir"           bash -c "ls tests/ 2>/dev/null | head -10 || echo '(tests/ 不存在)'"
fi
cd - >/dev/null 2>&1 || true

# ---------- 9. 风险初评 ----------
section "9. 风险初评（脚本侧自动判断）"
# 把上面的输出做几个简单 grep 判断，给出 P0/P1/P2 提示
# 注意：grep 模式必须带 4 空格缩进（run() 真实输出格式），
#       避免误中脚本源代码里的字面量。
{
    printf '\n```text\n'

    # rocm-smi
    if grep -qE '^\s+\[OK\] exit=0\s*$' <(grep -B0 -A3 '^--- rocm-smi ---' "${BUF}"); then
        printf '[OK]   rocm-smi 可用 → DCU 驱动正常\n'
    elif grep -q '\[N/A\] command not found: rocm-smi' "${BUF}"; then
        printf '[P0]   rocm-smi 不可用 → DCU 驱动未装或 PATH 不通\n'
    else
        printf '[P0]   rocm-smi 异常（FAIL）→ 需查看原始输出\n'
    fi

    # hipcc --version（必须有 HIP version: X.Y 字样）
    if grep -qE 'HIP version: [0-9]+\.[0-9]+' "${BUF}"; then
        printf '[OK]   hipcc 可见 + HIP 版本号已捕获\n'
    else
        printf '[P0]   hipcc 不可用或未输出版本号 → DTK 未正确加载\n'
    fi

    # __builtin_amdgcn_* 内建可执行（真实输出有 4 空格缩进 + 字面量）
    if grep -qE '^    amdgcn builtin available: [0-9]+$' "${BUF}"; then
        printf '[OK]   __builtin_amdgcn_* 内建函数在 hipcc 下可执行\n'
    else
        printf '[P1]   amdgcn 内建测试失败 → 需排查 hipcc 与 DTK 版本匹配\n'
    fi

    # InfiniBand
    if grep -qE 'CA .* InfiniBand' "${BUF}"; then
        printf '[OK]   InfiniBand 设备可见 → 多卡通信可能 OK\n'
    elif grep -q '\[N/A\] command not found: ibstat' "${BUF}"; then
        printf '[P2]   ibstat 不可用 → 多卡通信可能走 TCP（性能受限）\n'
    else
        printf '[P2]   InfiniBand 状态未知（需人工看 ibstat 输出）\n'
    fi

    # Lustre（精确判断：真挂载是 `... type lustse` 行；echo 提示是 `no lustre... mounts`）
    if grep -qE '^    no lustre/nfs/gfs/ceph mounts$' "${BUF}"; then
        printf '[P2]   未发现 Lustre 挂载 → 大模型权重可能要本地化\n'
    elif grep -qE '^    .* type lustre' "${BUF}"; then
        printf '[OK]   Lustre 共享存储已挂载\n'
    else
        printf '[P2]   Lustre 状态未知（需人工看 mount 输出）\n'
    fi

    # Slurm
    if grep -qE 'sinfo.*CLUSTER' "${BUF}"; then
        printf '[OK]   Slurm 作业系统可用 → 需用 sbatch 提交任务\n'
    elif grep -q '\[N/A\] command not found: sinfo' "${BUF}"; then
        printf '[P2]   Slurm 不可用 → 可能不是 HPC 节点\n'
    else
        printf '[P2]   Slurm 状态未知（需人工看 sinfo 输出）\n'
    fi
    printf '```\n'
} | tee -a "${BUF}"

# ---------- 落盘 ----------
{
    printf '\n---\n\n'
    printf '## 元数据\n\n'
    printf -- '- **生成时间**：%s\n' "${TS_HUMAN}"
    printf -- '- **主机名**：%s\n' "${HOST_FULL}"
    printf -- '- **脚本**：probe-dcu.sh (CTorch 仓库 scripts/ 下)\n'
    printf -- '- **后续行动**：本报告 + 任何 [P0] 项必须 HITL 确认后才能进入第 1 步（链路打通）\n'
} >> "${BUF}"

# 把 markdown 报告从 BUF 抽出
mv "${BUF}" "${OUTPUT_FILE}"

printf '\n[probe-dcu] 报告已落盘：%s\n' "${OUTPUT_FILE}"
printf '[probe-dcu] 建议下一步：将此文件 scp 回本地后查看，标注 [P0] 项\n'

exit 0
