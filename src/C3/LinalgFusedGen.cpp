/**
 * @file LinalgFusedGen.cpp
 * @generation JIT-2.0 声明式区域融合后端（Linalg 纯逐元素融合实现）
 * @brief linalg.generic 多节点/多输出声明式融合编译器实现
 * @date 2026/08/15
 */

#include "C3/LinalgFusedGen.h"
#include "C3/JITCache.h"
#include "Ctools.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/Support/TargetSelect.h>
#include <llvm/IR/Module.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Target/TargetMachine.h>

#include <algorithm>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <queue>

namespace ct {
namespace c3 {

// ======================= 纯逐元素图判定 =======================

static bool isBroadcastableTo(const std::vector<size_t>& src, const std::vector<size_t>& dest) {
    if (src.size() > dest.size()) return false;
    size_t src_offset = dest.size() - src.size();
    for (size_t i = 0; i < src.size(); ++i) {
        size_t s_dim = src[i];
        size_t d_dim = dest[i + src_offset];
        if (s_dim != d_dim && s_dim != 1) {
            return false;
        }
    }
    return true;
}

bool isPureElementwiseGraph(const Graph& graph) {
    static const bool disabled = [] {
        const char* v = std::getenv("C3_LINALG_FUSED");
        return v != nullptr && std::string(v) == "0";
    }();
    if (disabled) return false;

    const auto& nodes = graph.nodes();
    if (nodes.empty()) return false;

    // 1. 检查是否含有不支持的算子（如 MatMul, SumReduce, Transpose 属于高阶几何算子，不在此融合）
    for (const auto& node : nodes) {
        // 输入节点用 ConstNode 占位，可以跳过
        if (std::find(graph.inputs().begin(), graph.inputs().end(), node.id) != graph.inputs().end()) {
            continue;
        }
        bool ok = std::visit([](auto&& arg) -> bool {
            using T = std::decay_t<decltype(arg)>;
            return std::is_same_v<T, AddNode> ||
                   std::is_same_v<T, SubNode> ||
                   std::is_same_v<T, MulNode> ||
                   std::is_same_v<T, DivNode> ||
                   std::is_same_v<T, NegNode> ||
                   std::is_same_v<T, ReLUNode> ||
                   std::is_same_v<T, SigmoidNode> ||
                   std::is_same_v<T, TanhNode> ||
                   std::is_same_v<T, ExpNode> ||
                   std::is_same_v<T, LogNode> ||
                   std::is_same_v<T, GtNode> ||
                   std::is_same_v<T, ConstNode> ||
                   std::is_same_v<T, FusedNode>; // 允许内置 FusedNode 展开
        }, node.op);
        if (!ok) return false;
    }

    // 2. [2026-08-16] 多维高级广播 Linalg 化支持
    //    支持各节点形状安全广播到图的最终输出形状。
    const auto& graph_outputs = graph.outputs();
    if (graph_outputs.empty()) return false;
    const auto& out_shape = graph.node(graph_outputs[0]).out_desc.shape;

    for (const auto& node : nodes) {
        if (!isBroadcastableTo(node.out_desc.shape, out_shape)) {
            return false;
        }
    }

    return true;
}

// ======================= MemRef 描述符展开 ABI =======================

struct MemRefDesc {
    float* alloc;
    float* aligned;
    int64_t offset;
    int64_t sizes[1];
    int64_t strides[1];
};

static void appendMemRefDescArgs(const MemRefDesc& desc, void** args, int& idx) {
    args[idx++] = const_cast<float**>(&desc.alloc);
    args[idx++] = const_cast<float**>(&desc.aligned);
    args[idx++] = const_cast<int64_t*>(&desc.offset);
    args[idx++] = const_cast<int64_t*>(&desc.sizes[0]);
    args[idx++] = const_cast<int64_t*>(&desc.strides[0]);
}

// ======================= MLIR 模块构建 =======================

namespace {

void buildLinalgFusedFunc(mlir::OpBuilder& builder, mlir::Location loc,
                          mlir::MLIRContext& context, const Graph& graph,
                          size_t num_inputs, size_t num_outputs) {
    auto f32Type = builder.getF32Type();
    auto memrefType = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, f32Type);

    // 函数签名：N 个输入 memrefs + M 个输出 memrefs
    std::vector<mlir::Type> arg_types(num_inputs + num_outputs, memrefType);
    auto funcType = builder.getFunctionType(arg_types, {});
    auto func = builder.create<mlir::func::FuncOp>(loc, "c3_kernel", funcType);
    for (size_t i = 0; i < arg_types.size(); ++i) {
        func.setArgAttr(i, "llvm.noalias", builder.getUnitAttr());
    }

    auto* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    std::vector<mlir::Value> in_memrefs;
    std::vector<mlir::Value> out_memrefs;
    for (size_t i = 0; i < num_inputs; ++i) {
        in_memrefs.push_back(entry->getArgument(static_cast<unsigned>(i)));
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        out_memrefs.push_back(entry->getArgument(static_cast<unsigned>(num_inputs + i)));
    }

    // 1D Identity 索引映射
    mlir::AffineExpr d0 = builder.getAffineDimExpr(0);
    auto identityMap = mlir::AffineMap::get(1, 0, {d0}, &context);

    std::vector<mlir::AffineMap> indexingMaps(num_inputs + num_outputs, identityMap);
    std::vector<mlir::utils::IteratorType> iteratorTypes{
        mlir::utils::IteratorType::parallel};

    // 构建 linalg.generic
    builder.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange{},
        mlir::ValueRange{in_memrefs},
        mlir::ValueRange{out_memrefs},
        indexingMaps,
        iteratorTypes,
        [&](mlir::OpBuilder& b, mlir::Location regionLoc, mlir::ValueRange args) {
            // 内部临时变量映射：node_id -> MLIR Value (scalar)
            std::unordered_map<size_t, mlir::Value> val_map;

            // 1. 初始化输入节点的值（对应 linalg.generic 的前 N 个 block arguments）
            const auto& graph_inputs = graph.inputs();
            for (size_t i = 0; i < num_inputs; ++i) {
                val_map[graph_inputs[i]] = args[static_cast<unsigned>(i)];
            }

            // 预定义常量
            auto zero_f = mlir::arith::ConstantFloatOp::create(b, regionLoc, f32Type, llvm::APFloat(0.0f));
            auto one_f  = mlir::arith::ConstantFloatOp::create(b, regionLoc, f32Type, llvm::APFloat(1.0f));

            // 拓扑排序计算节点
            std::vector<const Node*> compute_nodes;
            for (const auto& node : graph.nodes()) {
                // 跳过图输入节点
                if (std::find(graph_inputs.begin(), graph_inputs.end(), node.id) != graph_inputs.end()) {
                    continue;
                }
                compute_nodes.push_back(&node);
            }

            // 2. 遍历计算并构建 scalar IR
            for (const auto* node : compute_nodes) {
                mlir::Value result;

                std::visit([&](auto&& op_node) {
                    using T = std::decay_t<decltype(op_node)>;
                    
                    if constexpr (std::is_same_v<T, ConstNode>) {
                        result = mlir::arith::ConstantFloatOp::create(
                            b, regionLoc, f32Type, llvm::APFloat(static_cast<float>(op_node.value)));
                    }
                    else if constexpr (std::is_same_v<T, NegNode>) {
                        auto in = val_map.at(node->inputs[0]);
                        result = b.create<mlir::arith::NegFOp>(regionLoc, in);
                    }
                    else if constexpr (std::is_same_v<T, ReLUNode>) {
                        auto in = val_map.at(node->inputs[0]);
                        result = b.create<mlir::arith::MaxNumFOp>(regionLoc, in, zero_f);
                    }
                    else if constexpr (std::is_same_v<T, SigmoidNode>) {
                        auto in = val_map.at(node->inputs[0]);
                        mlir::Value neg_x = b.create<mlir::arith::NegFOp>(regionLoc, in);
                        mlir::Value exp_x = b.create<mlir::math::ExpOp>(regionLoc, neg_x);
                        mlir::Value denom = b.create<mlir::arith::AddFOp>(regionLoc, one_f, exp_x);
                        result = b.create<mlir::arith::DivFOp>(regionLoc, one_f, denom);
                    }
                    else if constexpr (std::is_same_v<T, TanhNode>) {
                        auto in = val_map.at(node->inputs[0]);
                        result = b.create<mlir::math::TanhOp>(regionLoc, in);
                    }
                    else if constexpr (std::is_same_v<T, ExpNode>) {
                        auto in = val_map.at(node->inputs[0]);
                        result = b.create<mlir::math::ExpOp>(regionLoc, in);
                    }
                    else if constexpr (std::is_same_v<T, LogNode>) {
                        auto in = val_map.at(node->inputs[0]);
                        result = b.create<mlir::math::LogOp>(regionLoc, in);
                    }
                    else if constexpr (std::is_same_v<T, AddNode>) {
                        auto lhs = val_map.at(node->inputs[0]);
                        auto rhs = val_map.at(node->inputs[1]);
                        result = b.create<mlir::arith::AddFOp>(regionLoc, lhs, rhs);
                    }
                    else if constexpr (std::is_same_v<T, SubNode>) {
                        auto lhs = val_map.at(node->inputs[0]);
                        auto rhs = val_map.at(node->inputs[1]);
                        result = b.create<mlir::arith::SubFOp>(regionLoc, lhs, rhs);
                    }
                    else if constexpr (std::is_same_v<T, MulNode>) {
                        auto lhs = val_map.at(node->inputs[0]);
                        auto rhs = val_map.at(node->inputs[1]);
                        result = b.create<mlir::arith::MulFOp>(regionLoc, lhs, rhs);
                    }
                    else if constexpr (std::is_same_v<T, DivNode>) {
                        auto lhs = val_map.at(node->inputs[0]);
                        auto rhs = val_map.at(node->inputs[1]);
                        result = b.create<mlir::arith::DivFOp>(regionLoc, lhs, rhs);
                    }
                    else if constexpr (std::is_same_v<T, GtNode>) {
                        auto lhs = val_map.at(node->inputs[0]);
                        auto rhs = val_map.at(node->inputs[1]);
                        mlir::Value cmp = b.create<mlir::arith::CmpFOp>(
                            regionLoc, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
                        result = b.create<mlir::arith::SelectOp>(regionLoc, cmp, one_f, zero_f);
                    }
                    else if constexpr (std::is_same_v<T, FusedNode>) {
                        // 展开内置 FusedNode 节点
                        const auto& fnode = op_node;
                        std::unordered_map<size_t, mlir::Value> f_map;
                        // 建立 FusedNode 输入映射
                        for (size_t aidx = 0; aidx < fnode.arg_node_ids.size(); ++aidx) {
                            f_map[fnode.arg_node_ids[aidx]] = val_map.at(node->inputs[aidx]);
                        }
                        
                        mlir::Value f_prev_val;
                        // 拓扑计算 FusedNode 内部操作
                        for (size_t f_idx = 0; f_idx < fnode.ops.size(); ++f_idx) {
                            const auto& f_op = fnode.ops[f_idx];
                            const auto& f_inputs = fnode.op_inputs[f_idx];
                            
                            std::vector<size_t> ext_inputs;
                            for (size_t in_id : f_inputs) {
                                if (f_idx > 0 && in_id == f_inputs[0]) continue;
                                ext_inputs.push_back(in_id);
                            }
                            
                            auto loadExt = [&](size_t id) -> mlir::Value {
                                return f_map.at(id);
                            };

                            mlir::Value f_res;
                            std::visit([&](auto&& f_node) {
                                using FT = std::decay_t<decltype(f_node)>;
                                if constexpr (std::is_same_v<FT, NegNode>) {
                                    mlir::Value f_in = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                                    f_res = b.create<mlir::arith::NegFOp>(regionLoc, f_in);
                                }
                                else if constexpr (std::is_same_v<FT, ReLUNode>) {
                                    mlir::Value f_in = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                                    f_res = b.create<mlir::arith::MaxNumFOp>(regionLoc, f_in, zero_f);
                                }
                                else if constexpr (std::is_same_v<FT, SigmoidNode>) {
                                    mlir::Value f_in = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                                    mlir::Value neg_x = b.create<mlir::arith::NegFOp>(regionLoc, f_in);
                                    mlir::Value exp_x = b.create<mlir::math::ExpOp>(regionLoc, neg_x);
                                    mlir::Value denom = b.create<mlir::arith::AddFOp>(regionLoc, one_f, exp_x);
                                    f_res = b.create<mlir::arith::DivFOp>(regionLoc, one_f, denom);
                                }
                                else if constexpr (std::is_same_v<FT, TanhNode>) {
                                    mlir::Value f_in = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                                    f_res = b.create<mlir::math::TanhOp>(regionLoc, f_in);
                                }
                                else if constexpr (std::is_same_v<FT, ExpNode>) {
                                    mlir::Value f_in = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                                    f_res = b.create<mlir::math::ExpOp>(regionLoc, f_in);
                                }
                                else if constexpr (std::is_same_v<FT, LogNode>) {
                                    mlir::Value f_in = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                                    f_res = b.create<mlir::math::LogOp>(regionLoc, f_in);
                                }
                                else if constexpr (std::is_same_v<FT, AddNode>) {
                                    mlir::Value f_lhs = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                                    mlir::Value f_rhs = loadExt(ext_inputs[f_idx > 0 ? 0 : 1]);
                                    f_res = b.create<mlir::arith::AddFOp>(regionLoc, f_lhs, f_rhs);
                                }
                                else if constexpr (std::is_same_v<FT, SubNode>) {
                                    mlir::Value f_lhs = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                                    mlir::Value f_rhs = loadExt(ext_inputs[f_idx > 0 ? 0 : 1]);
                                    f_res = b.create<mlir::arith::SubFOp>(regionLoc, f_lhs, f_rhs);
                                }
                                else if constexpr (std::is_same_v<FT, MulNode>) {
                                    mlir::Value f_lhs = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                                    mlir::Value f_rhs = loadExt(ext_inputs[f_idx > 0 ? 0 : 1]);
                                    f_res = b.create<mlir::arith::MulFOp>(regionLoc, f_lhs, f_rhs);
                                }
                                else if constexpr (std::is_same_v<FT, DivNode>) {
                                    mlir::Value f_lhs = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                                    mlir::Value f_rhs = loadExt(ext_inputs[f_idx > 0 ? 0 : 1]);
                                    f_res = b.create<mlir::arith::DivFOp>(regionLoc, f_lhs, f_rhs);
                                }
                                else if constexpr (std::is_same_v<FT, GtNode>) {
                                    mlir::Value f_lhs = (f_idx > 0) ? f_prev_val : loadExt(ext_inputs[0]);
                                    mlir::Value f_rhs = loadExt(ext_inputs[f_idx > 0 ? 0 : 1]);
                                    mlir::Value cmp = b.create<mlir::arith::CmpFOp>(
                                        regionLoc, mlir::arith::CmpFPredicate::OGT, f_lhs, f_rhs);
                                    f_res = b.create<mlir::arith::SelectOp>(regionLoc, cmp, one_f, zero_f);
                                }
                            }, f_op);
                            
                            f_prev_val = f_res;
                            if (f_idx == fnode.ops.size() - 1) {
                                result = f_res;
                            }
                        }
                    }
                }, node->op);

                val_map[node->id] = result;
            }

            // 3. 将所有输出节点对应的值 yield 出来
            std::vector<mlir::Value> yield_vals;
            const auto& graph_outputs = graph.outputs();
            for (size_t out_id : graph_outputs) {
                yield_vals.push_back(val_map.at(out_id));
            }

            b.create<mlir::linalg::YieldOp>(regionLoc, yield_vals);
        });

    builder.create<mlir::func::ReturnOp>(loc);
}

mlir::OwningOpRef<mlir::ModuleOp> buildLinalgFusedModule(mlir::MLIRContext& context,
                                                        const Graph& graph,
                                                        size_t num_inputs,
                                                        size_t num_outputs) {
    auto loc = mlir::UnknownLoc::get(&context);
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    buildLinalgFusedFunc(builder, loc, context, graph, num_inputs, num_outputs);
    return module;
}

void applyLinalgLoweringPipeline(mlir::ModuleOp module) {
    // 阶段 0.5：Linalg 级特化与内联优化
    {
        mlir::PassManager pm(module.getContext());
        pm.addPass(mlir::createLinalgInlineScalarOperandsPass());
        pm.addPass(mlir::createLinalgSpecializeGenericOpsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        if (mlir::failed(pm.run(module))) {
            throw std::runtime_error("LinalgFusedGen: Linalg optimization failed");
        }
    }
    // 阶段 1：linalg.generic → loops
    {
        mlir::PassManager pm(module.getContext());
        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        if (mlir::failed(pm.run(module))) {
            throw std::runtime_error("LinalgFusedGen: linalg-to-loops failed");
        }
    }
    // 阶段 2：scf → cf → LLVM
    {
        mlir::PassManager pm(module.getContext());
        pm.addPass(mlir::createControlFlowSinkPass());
        pm.addPass(mlir::createRemoveDeadValuesPass());
        pm.addPass(mlir::createLoopInvariantCodeMotionPass());
        pm.addPass(mlir::createSCFToControlFlowPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertMathToLLVMPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        if (mlir::failed(pm.run(module))) {
            throw std::runtime_error("LinalgFusedGen: lowering pipeline failed");
        }
    }
}

std::unique_ptr<mlir::ExecutionEngine> createEngine(
    mlir::ModuleOp module, int opt_level, const std::string& cache_graph,
    std::function<std::unique_ptr<llvm::Module>(mlir::Operation*, llvm::LLVMContext&)>&
        builder_slot) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    auto tm = std::shared_ptr<llvm::TargetMachine>(
        llvm::EngineBuilder()
            .setEngineKind(llvm::EngineKind::JIT)
            .setOptLevel(static_cast<llvm::CodeGenOptLevel>(opt_level))
            .selectTarget());

    std::function<llvm::Error(llvm::Module*)> opt_transformer =
        tm ? mlir::makeOptimizingTransformer(static_cast<unsigned>(opt_level), 0, tm.get())
           : std::function<llvm::Error(llvm::Module*)>();

    mlir::ExecutionEngineOptions engineOpts;
    if (opt_transformer) {
        engineOpts.transformer = opt_transformer;
    }
    engineOpts.jitCodeGenOptLevel = (opt_level >= 3)
        ? llvm::CodeGenOptLevel::Aggressive
        : (opt_level == 2) ? llvm::CodeGenOptLevel::Default
        : (opt_level == 1) ? llvm::CodeGenOptLevel::Less
        : llvm::CodeGenOptLevel::None;

    // read path AOT JITCache read-path
    if (JITCache::isEnabled()) {
        try {
            std::string jit_key = JITCache::makeKey(cache_graph, opt_level);
            std::string bc_path = JITCache::getInstance().lookup(jit_key);
            if (!bc_path.empty()) {
                builder_slot = [bc_path](mlir::Operation*, llvm::LLVMContext& ctx) {
                    auto m = JITCache::getInstance().loadBitcode(bc_path, ctx);
                    return m;
                };
            } else {
                builder_slot = [module, jit_key](mlir::Operation*, llvm::LLVMContext& ctx) {
                    auto llvm_module = mlir::translateModuleToLLVMIR(module, ctx);
                    if (llvm_module) {
                        auto st = JITCache::getInstance().store(jit_key, *llvm_module);
                        (void)st;
                    }
                    return llvm_module;
                };
            }
            engineOpts.llvmModuleBuilder = builder_slot;
        } catch (...) {
            // 静默降级
        }
    }

    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOpts);
    if (!maybeEngine) {
        throw std::runtime_error("LinalgFusedGen: failed to create ExecutionEngine");
    }
    return std::move(*maybeEngine);
}

} // namespace

// ======================= LinalgFusedKernel =======================

struct LinalgFusedKernel::Impl {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context;

    mlir::OwningOpRef<mlir::ModuleOp> heldModule;
    std::function<std::unique_ptr<llvm::Module>(mlir::Operation*, llvm::LLVMContext&)>
        aotBuilder;

    std::unique_ptr<mlir::ExecutionEngine> engine;

    Impl() : context(registry) {
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::math::MathDialect>();
        registry.insert<mlir::scf::SCFDialect>();
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::memref::MemRefDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
        registry.insert<mlir::linalg::LinalgDialect>();
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();
    }
};

LinalgFusedKernel::LinalgFusedKernel(const Graph& graph, int opt_level)
    : impl_(std::make_unique<Impl>()), num_inputs_(graph.inputs().size()),
      num_outputs_(graph.outputs().size()) {

    impl_->heldModule = buildLinalgFusedModule(impl_->context, graph, num_inputs_, num_outputs_);
    mlir::ModuleOp module = *impl_->heldModule;
    applyLinalgLoweringPipeline(module);

    if (std::getenv("C3_MLIR_DUMP")) {
        llvm::errs() << "==== C3 Fused Lowered Module ====\n";
        module.dump();
        llvm::errs() << "==== end ====\n";
    }

    // 缓存 Key 串，混入 graph.toString() 保证唯一性，防止不同操作图由于节点数/输入输出数相同而撞 key
    std::string cache_graph = std::string("linalg_fused_") + graph.toString()
                              + "_ol" + std::to_string(opt_level);
    impl_->engine = createEngine(module, opt_level, cache_graph, impl_->aotBuilder);
    if (!impl_->engine->lookup("c3_kernel")) {
        throw std::runtime_error("LinalgFusedGen: lookup c3_kernel failed");
    }
}

LinalgFusedKernel::~LinalgFusedKernel() = default;
LinalgFusedKernel::LinalgFusedKernel(LinalgFusedKernel&&) noexcept = default;
LinalgFusedKernel& LinalgFusedKernel::operator=(LinalgFusedKernel&&) noexcept = default;

void LinalgFusedKernel::execute(const float* const* in_ptrs, float* const* out_ptrs,
                                size_t n) const {
    const size_t num_memrefs = num_inputs_ + num_outputs_;

    // descs/args 需函数级保活
    std::vector<MemRefDesc> descs(num_memrefs);
    std::vector<void*> args(num_memrefs * 5);
    int arg_idx = 0;

    for (size_t i = 0; i < num_inputs_; ++i) {
        descs[i] = MemRefDesc{const_cast<float*>(in_ptrs[i]), const_cast<float*>(in_ptrs[i]),
                              0, {static_cast<int64_t>(n)}, {1}};
        appendMemRefDescArgs(descs[i], args.data(), arg_idx);
    }
    for (size_t i = 0; i < num_outputs_; ++i) {
        size_t desc_idx = num_inputs_ + i;
        descs[desc_idx] = MemRefDesc{out_ptrs[i], out_ptrs[i], 0, {static_cast<int64_t>(n)}, {1}};
        appendMemRefDescArgs(descs[desc_idx], args.data(), arg_idx);
    }

    auto err = impl_->engine->invokePacked("c3_kernel", args);
    if (err) {
        throw std::runtime_error("LinalgFusedGen: invokePacked failed: "
                                 + llvm::toString(std::move(err)));
    }
}

// ======================= 共享融合 kernel 缓存工厂 =======================

std::shared_ptr<LinalgFusedKernel> getCachedLinalgFusedKernel(
    const Graph& graph, const std::string& graph_key, int opt_level) {
    static const bool cache_disabled = [] {
        const char* v = std::getenv("C3_LINALG_CACHE");
        return v != nullptr && std::string(v) == "0";
    }();
    if (cache_disabled) {
        return std::make_shared<LinalgFusedKernel>(graph, opt_level);
    }

    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::weak_ptr<LinalgFusedKernel>> cache;

    std::string key = graph_key + "_ol" + std::to_string(opt_level);

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        if (auto sp = it->second.lock()) {
            return sp;
        }
        cache.erase(it);
    }

    auto kernel = std::make_shared<LinalgFusedKernel>(graph, opt_level);
    cache[key] = kernel;
    return kernel;
}

} // namespace c3
} // namespace ct
