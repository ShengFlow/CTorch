module; // 全局模块片段

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include <memory>

export module AutoDiff;
import Ctools;

export {

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
    static AutoDiff*& current() {
        static thread_local AutoDiff* ctx = nullptr;
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
        explicit Guard(AutoDiff* new_ctx) : prev_ctx(current()) {
            current() = new_ctx;  // 现在可以正确赋值
        }

        /**
         * @brief 析构函数，恢复之前的自动微分上下文
         */
        ~Guard() {
            current() = prev_ctx;
        }

        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;

    private:
        AutoDiff* prev_ctx;  // 之前的自动微分对象
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
        size_t tensor_id;        ///< 张量ID
        std::unique_ptr<Tensor> tensor;  ///< 张量指针
        std::unique_ptr<Tensor> grad;    ///< 梯度指针
        std::vector<size_t> input_ids;   ///< 输入节点ID列表
        op operation;          ///< 操作类型
        bool requires_grad;    ///< 是否需要梯度
        bool is_leaf;          ///< 是否为叶子节点

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
        op operation;                  ///< 操作类型
        std::vector<size_t> input_ids;     ///< 输入节点ID列表
        std::vector<std::vector<size_t>> input_shapes;  ///< 输入节点形状列表
        bool committed = false;         ///< 是否已提交
    };

    std::unordered_map<size_t, std::unique_ptr<Node>> id_to_node;    ///< ID到节点的映射
    std::unordered_map<size_t, PendingRecord> pending_records;      ///< 待处理记录
    std::mutex records_mutex;       ///< 记录锁
    bool retain_graph = false;      ///< 是否保留计算图

    /**
     * @brief 辅助方法：根据ID获取节点
     * @param id 节点ID
     * @return 节点指针
     */
    Node* get_node_by_id(size_t id);

    /**
     * @brief 私有辅助函数：DFS拓扑排序
     * @param node 当前节点
     * @param visited 已访问节点集合
     * @param result 拓扑序列结果
     */
    void dfs_topological_sort(Node* node, std::unordered_set<Node*>& visited, std::vector<Node*>& result);
    
    /**
     * @brief 私有辅助函数：检查并调整梯度形状
     * @param grad 输入梯度张量
     * @param target_shape 目标形状
     * @return 调整后的梯度张量
     */
    Tensor check_and_adjust_grad_shape(const Tensor& grad, const std::vector<size_t>& target_shape);

public:
    /**
     * @brief 添加详细的调试方法
     * @param context 调试上下文
     */
    void debug_print_state(const std::string& context);

    /**
     * @brief 获取张量的梯度
     * @param t 张量指针
     * @return 梯度张量
     */
    Tensor get_grad(const Tensor* t);

    /**
     * @brief 将张量设为叶子节点
     * @param t 张量引用
     * @param requires_grad 是否需要梯度
     */
    void make_leaf(Tensor& t, bool requires_grad);

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
    void defer_record(size_t output_id, op operation, const std::vector<Tensor*>& inputs);

    /**
     * @brief 提交延迟记录
     * @param output 输出张量
     */
    void commit_record(Tensor& output);

    /**
     * @brief 更新梯度需求
     * @param t 张量引用
     * @param requires_grad 是否需要梯度
     */
    void update_requires_grad(Tensor& t, bool requires_grad);

    /**
     * @brief 设置是否保留计算图
     * @param retain 是否保留计算图
     */
    void set_retain_graph(bool retain);

    /**
     * @brief 反向传播（版本1：用户不提供grad_output，使用默认的1.0）
     * @param root 根张量
     */
    void backward(Tensor& root);

    /**
     * @brief 反向传播（版本2：用户提供grad_output）
     * @param root 根张量
     * @param grad_output 输出梯度
     */
    void backward(Tensor& root, Tensor grad_output);
};

// ======================= 辅助函数 =======================

/**
 * @brief 创建空张量的辅助函数
 * @return 空张量
 */
Tensor create_empty_tensor();

}

// 实现部分

// Node 构造函数
AutoDiff::Node::Node(size_t id, std::unique_ptr<Tensor> t, bool req_grad, bool leaf)
    : tensor_id(id), tensor(std::move(t)), grad(nullptr), requires_grad(req_grad), is_leaf(leaf) {}

// Node::clear_grad_safely 方法
void AutoDiff::Node::clear_grad_safely() {
    if (grad) {
        // 这里可以添加安全清理的逻辑
        grad.reset();
    }
}

// AutoDiff::get_node_by_id 方法
AutoDiff::Node* AutoDiff::get_node_by_id(size_t id) {
    auto it = id_to_node.find(id);
    if (it != id_to_node.end()) {
        return it->second.get();
    }
    return nullptr;
}

// AutoDiff::dfs_topological_sort 方法
void AutoDiff::dfs_topological_sort(Node* node, std::unordered_set<Node*>& visited, std::vector<Node*>& result) {
    if (visited.count(node)) {
        return;
    }
    visited.insert(node);
    for (size_t input_id : node->input_ids) {
        Node* input_node = get_node_by_id(input_id);
        if (input_node) {
            dfs_topological_sort(input_node, visited, result);
        }
    }
    result.push_back(node);
}

// AutoDiff::check_and_adjust_grad_shape 方法
Tensor AutoDiff::check_and_adjust_grad_shape(const Tensor& grad, const std::vector<size_t>& target_shape) {
    // 这里需要实现检查并调整梯度形状的逻辑
    // 暂时返回原始梯度
    return grad;
}

// AutoDiff::debug_print_state 方法
void AutoDiff::debug_print_state(const std::string& context) {
    // 这里需要实现调试打印的逻辑
    // 暂时为空
}

// AutoDiff::get_grad 方法
Tensor AutoDiff::get_grad(const Tensor* t) {
    // 这里需要实现获取梯度的逻辑
    // 暂时返回空张量
    return create_empty_tensor();
}

// AutoDiff::make_leaf 方法
void AutoDiff::make_leaf(Tensor& t, bool requires_grad) {
    // 这里需要实现将张量设为叶子节点的逻辑
    // 暂时为空
}

// AutoDiff 析构函数
AutoDiff::~AutoDiff() {
    // 这里需要实现析构函数的逻辑
    // 暂时为空
}

// AutoDiff::defer_record 方法
void AutoDiff::defer_record(size_t output_id, op operation, const std::vector<Tensor*>& inputs) {
    // 这里需要实现延迟记录操作的逻辑
    // 暂时为空
}

// AutoDiff::commit_record 方法
void AutoDiff::commit_record(Tensor& output) {
    // 这里需要实现提交延迟记录的逻辑
    // 暂时为空
}

// AutoDiff::update_requires_grad 方法
void AutoDiff::update_requires_grad(Tensor& t, bool requires_grad) {
    // 这里需要实现更新梯度需求的逻辑
    // 暂时为空
}

// AutoDiff::set_retain_graph 方法
void AutoDiff::set_retain_graph(bool retain) {
    retain_graph = retain;
}

// AutoDiff::backward 方法（版本1）
void AutoDiff::backward(Tensor& root) {
    // 这里需要实现反向传播的逻辑
    // 暂时为空
}

// AutoDiff::backward 方法（版本2）
void AutoDiff::backward(Tensor& root, Tensor grad_output) {
    // 这里需要实现反向传播的逻辑
    // 暂时为空
}

// create_empty_tensor 函数
Tensor create_empty_tensor() {
    // 这里需要实现创建空张量的逻辑
    // 暂时返回默认构造的张量
    return Tensor();
}
