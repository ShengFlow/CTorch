/**
 * @file AutoDiff.h
 * @brief Ctorch 自动微分系统
 * @author GhostFace
 * @date 2025/12/21
 * @version v3.1
 * @details
 * 定义了自动微分系统的核心组件，包括自动微分上下文、自动微分类和相关辅助函数
 */

#ifndef AUTODIFF_H
#define AUTODIFF_H

#include "Ctools.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// 前向声明
class Tensor;
class AutoDiff;

/**
 * @class AutoDiffContext
 * @brief 自动微分上下文类，用于管理线程本地的自动微分对象
 */
class AutoDiffContext {
public:
  /**
   * @brief 获取当前线程的自动微分对象
   * @return 当前线程的自动微分对象引用
   */
  static AutoDiff *&current() {
    static thread_local AutoDiff *ctx = nullptr;
    return ctx;
  }

  /**
   * @class Guard
   * @brief 自动微分上下文的保护类，用于管理自动微分对象的生命周期
   */
  class Guard {
  public:
    /**
     * @brief 构造函数，设置新的自动微分上下文
     * @param new_ctx 新的自动微分对象
     */
    explicit Guard(AutoDiff *new_ctx) : prev_ctx(current()) {
      current() = new_ctx; // 现在可以正确赋值
    }

    /**
     * @brief 析构函数，恢复之前的自动微分上下文
     */
    ~Guard() { current() = prev_ctx; }

    Guard(const Guard &) = delete;
    Guard &operator=(const Guard &) = delete;

  private:
    AutoDiff *prev_ctx; // 之前的自动微分对象
  };
};

/**
 * @class AutoDiff
 * @brief 自动微分类，用于管理计算图和执行反向传播
 */
class AutoDiff {
private:
  /**
   * @struct Node
   * @brief 计算图节点定义
   */
  struct Node {
    size_t tensor_id;               ///< 张量ID
    std::unique_ptr<Tensor> tensor; ///< 张量指针（拥有所有权）
    std::unique_ptr<Tensor> grad;   ///< 梯度指针
    std::vector<size_t> input_ids;  ///< 输入节点ID列表
    op operation;                   ///< 操作类型
    int op_param_i = 0;             ///< 算子整型参数（目前用于 Softmax dim）
    bool requires_grad;             ///< 是否需要梯度
    bool is_leaf;                   ///< 是否为叶子节点

    /**
     * @brief 节点构造函数
     * @param id 张量ID
     * @param t 张量指针
     * @param req_grad 是否需要梯度
     * @param leaf 是否为叶子节点
     */
    Node(size_t id, std::unique_ptr<Tensor> t, bool req_grad, bool leaf = true);

    /**
     * @brief 安全清理梯度的方法
     */
    void clear_grad_safely();
  };

  /**
   * @struct PendingRecord
   * @brief 待处理记录，用于延迟记录计算图操作
   */
  struct PendingRecord {
    op operation;                                  ///< 操作类型
    std::vector<size_t> input_ids;                 ///< 输入节点ID列表
    std::vector<std::vector<size_t>> input_shapes; ///< 输入节点形状列表
    int op_param_i = 0;     ///< 算子整型参数（目前用于 Softmax dim）
    bool committed = false; ///< 是否已提交
  };

  std::unordered_map<size_t, std::unique_ptr<Node>>
      id_to_node;                                            ///< ID到节点的映射
  std::unordered_map<size_t, PendingRecord> pending_records; ///< 待处理记录
  std::mutex records_mutex;                                  ///< 记录锁
  bool retain_graph = false;                                 ///< 是否保留计算图

  /**
   * @brief 辅助方法：根据ID获取节点
   * @param id 节点ID
   * @return 节点指针
   */
  Node *get_node_by_id(size_t id);

  /**
   * @brief 私有辅助函数：DFS拓扑排序
   * @param node 当前节点
   * @param visited 已访问节点集合
   * @param result 拓扑序列结果
   */
  void dfs_topological_sort(Node *node, std::unordered_set<Node *> &visited,
                            std::vector<Node *> &result);

  /**
   * @brief 私有辅助函数：检查并调整梯度形状
   * @param grad 输入梯度张量
   * @param target_shape 目标形状
   * @return 调整后的梯度张量
   */
  Tensor check_and_adjust_grad_shape(const Tensor &grad,
                                     const std::vector<size_t> &target_shape);

public:
  /**
   * @brief 添加详细的调试方法
   * @param context 调试上下文
   */
  void debug_print_state(const std::string &context);

  /**
   * @brief 获取张量的梯度
   * @param t 张量指针
   * @return 梯度张量
   */
  Tensor get_grad(const Tensor *t);

  /**
   * @brief 将张量设为叶子节点
   * @param t 张量引用
   * @param requires_grad 是否需要梯度
   */
  void make_leaf(Tensor &t, bool requires_grad);

  /**
   * @brief 清零指定张量的梯度（用于每个batch训练前，避免梯度累积）
   * @param t 需要清零梯度的张量
   */
  void zero_grad(Tensor &t);

  /**
   * @brief 析构函数
   */
  ~AutoDiff();

  /**
   * @brief 延迟记录操作
   * @param output_id 输出张量ID
   * @param operation 操作类型
   * @param inputs 输入张量列表
   */
  void defer_record(size_t output_id, op operation,
                    const std::vector<Tensor *> &inputs, int op_param_i = 0);

  /**
   * @brief 提交延迟记录
   * @param output 输出张量
   */
  void commit_record(Tensor &output);

  /**
   * @brief 更新梯度需求
   * @param t 张量引用
   * @param requires_grad 是否需要梯度
   */
  void update_requires_grad(Tensor &t, bool requires_grad);

  /**
   * @brief 设置是否保留计算图
   * @param retain 是否保留计算图
   */
  void set_retain_graph(bool retain);

  /**
   * @brief 反向传播（版本1：用户不提供grad_output，使用默认的1.0）
   * @param root 根张量
   */
  void backward(Tensor &root);

  /**
   * @brief 反向传播（版本2：用户提供grad_output）
   * @param root 根张量
   * @param grad_output 输出梯度
   */
  void backward(Tensor &root, Tensor grad_output);

  /**
   *@brief 清理计算图
   **/
  void clear_graph();
};

// ======================= 辅助函数 =======================

/**
 * @brief 创建空张量的辅助函数
 * @return 空张量
 */
Tensor create_empty_tensor();

#endif // AUTODIFF_H