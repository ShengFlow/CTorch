//
// Created by renyz on 2026/3/13.
//

#ifndef CTORCH_ARRAY_H
#define CTORCH_ARRAY_H

#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <array>

#include "Assertion.h"

namespace ct::tl { inline namespace array {
namespace details {
// Markers for static dispatching
struct ReserveAxis {};
struct NewAxis { int64_t repeat; };
struct Ellipse {};
struct Range { int64_t start, end, step; }; // end is exclusive
} // namespace details

/**
 * @brief Marker object for reserving an axis during slicing operations.
 *
 * When used in operator(), this marker indicates that the corresponding axis
 * should be kept (not eliminated) in the resulting view.
 *
 * @example
 *   Array<float, 3> arr = ...;  // shape [4, 5, 6]
 *   auto view = arr(0, reserve, reserve);  // shape [1, 5, 6], first axis kept
 *
 * @note This is a compile-time marker and incurs no runtime overhead.
 */
static constexpr auto reserve = details::ReserveAxis();

/**
 * @brief Marker object for ellipsis slicing (placeholder for multiple dimensions).
 *
 * @warning Currently NOT supported. Reserved for future implementation.
 *           Will expand to cover all unspecified dimensions when implemented.
 */
static constexpr auto ellipse = details::Ellipse(); // TODO currently not supported

/**
 * @brief Creates a marker for inserting a new axis (dimension) during slicing.
 *
 * This is used to expand the dimensionality of an array view by adding
 * a new axis with _size 1 (or repeated multiple times).
 *
 * @param repeat Number of times to repeat the new axis. Default is 1.
 *               Setting repeat > 1 creates multiple consecutive dimensions of _size 1.
 * @return A NewAxis marker object for use in slicing operations.
 *
 * @pre repeat >= 0 (assertion failure if negative)
 *
 * @example
 *   Array<float, 2> arr = ...;  // shape [3, 4]
 *   auto view = arr(new_axis(), reserve, reserve);  // shape [1, 3, 4]
 *   auto view2 = arr(new_axis(2), reserve);         // shape [1, 1, 3, 4]
 *
 * @note The new axis has stride 0, meaning all elements along this axis
 *       share the same underlying data (broadcasting semantics).
 */
static constexpr auto new_axis(int64_t repeat = 1) {
  CT_ASSERT(repeat >= 0, "Cannot repeat negative times");
  return details::NewAxis{repeat};
}

/**
 * @brief Creates a range marker for slicing a subset of elements along an axis.
 *
 * The range follows Python-style semantics where 'end' is exclusive.
 * This is used in operator() to extract a slice of elements.
 *
 * @param start Starting index of the range (inclusive).
 * @param end Ending index of the range (exclusive).
 * @param step Step _size for the slice. Default is 1.
 *             Can be negative to reverse the slice order.
 * @return A Range marker object for use in slicing operations.
 *
 * @note No validation is performed at creation time. Validation occurs
 *       during the actual slicing operation in operator().
 *
 * @example
 *   Array<float, 2> arr = ...;  // shape [10, 20]
 *   auto view1 = arr(range(0, 5), reserve);     // shape [5, 20], first 5 rows
 *   auto view2 = arr(range(0, 10, 2), reserve); // shape [5, 20], every other row
 *   auto view3 = arr(range(9, -1, -1), reserve); // shape [10, 20], reversed rows
 *
 * @warning Unlike Python, negative indices for start/end are NOT supported.
 *          You must specify explicit positive indices.
 */
static constexpr auto range(int64_t start, int64_t end, int64_t step = 1) {
  return details::Range{start, end, step};
}

/**
 * @brief Flags for specifying memory contiguity guarantees of an Array.
 *
 * These flags provide compile-time hints about the memory layout of the array,
 * enabling optimizations in slicing and indexing operations. The flags are
 * cumulative in a bitmask fashion.
 *
 * @note These are static assertions. If the actual memory layout doesn't match
 *       the specified flags, behavior is undefined (assertion in debug mode).
 */
enum ArrayFlags : int {
  /** No contiguity guarantees provided */
  AF_NONE = 0,

  /**
   * @brief Static assertion that the last axis of the array is contiguous.
   *
   * This means stride[N-1] == 1, allowing optimized access to the innermost
   * dimension. This is the minimum requirement for efficient element-wise
   * operations on the last dimension.
   */
  AF_LAST_CONTIGUOUS = 1,

  /**
   * @brief Static assertion that the last 2 axes of the array are contiguous.
   *
   * This means:
   *   - stride[N-1] == 1
   *   - stride[N-2] == _size[N-1]
   *
   * Enables optimized 2D operations like matrix access patterns.
   * Note: This is a bitmask value of 3 (AF_LAST_CONTIGUOUS | additional flag).
   */
  AF_LAST2_CONTIGUOUS = 3,

  /**
   * @brief Static assertion that the entire array is contiguous in memory.
   *
   * This means the array has a dense row-major layout where:
   *   - stride[N-1] == 1
   *   - stride[N-2] == _size[N-1]
   *   - stride[N-3] == _size[N-1] * size[N-2]
   *   - ... and so on
   *
   * This is the most restrictive guarantee but enables maximum optimization.
   * Note: This is a bitmask value of 7.
   */
  AF_CONTIGUOUS = 7,

  /** Internal: mask for all contiguity-related flags */
  _AF_ALL_CONTIGUITY_FLAGS = 7,
};

/**
 * @brief A non-owning view of memory as a dense tensor with N dimensions.
 *
 * Array provides a lightweight, zero-cost abstraction over raw memory,
 * supporting arbitrary strides for flexible memory layouts. It does NOT
 * own the underlying memory and does NOT perform any memory management.
 *
 * @tparam T The element data type. Must be a trivially copyable type.
 * @tparam N Number of dimensions. Must be positive (> 0).
 * @tparam FLAGS Compile-time flags for contiguity guarantees (default: AF_NONE).
 *
 * @section memory_ownership Memory Ownership
 * Array is a VIEW type. It does NOT:
 *   - Allocate memory
 *   - Free memory
 *   - Manage memory lifetime
 *
 * The caller is responsible for ensuring the underlying memory remains valid
 * for the lifetime of the Array object.
 *
 * @section thread_safety Thread Safety
 * Array operations are thread-safe for read-only access. Concurrent writes
 * to overlapping memory regions (via different Array views) require external
 * synchronization.
 *
 * @section performance Performance
 *   - All operations are constexpr and can be evaluated at compile time
 *   - CT_ALWAYS_FORCEINLINE ensures aggressive inlining for zero overhead
 *   - Contiguity flags enable compile-time optimizations
 *
 * @section example Basic Usage
 * @code
 *   float data[24];
 *   int64_t sizes[] = {4, 6};
 *   int64_t strides[] = {6, 1};
 *   Array<float, 2> arr(data, sizes, strides);
 *   float val = arr(1, 2);  // Access element
 *   auto row = arr(1, reserve);  // Get row view
 * @endcode
 */
template <typename T, int64_t N, int FLAGS = AF_NONE>
class Array {
  static_assert(N > 0, "Ndim must be positive");
public:
  /** Type alias for Array with last-axis contiguity guarantee */
  using L1Contiguous = Array<T, N, FLAGS | AF_LAST_CONTIGUOUS>;

  /** Type alias for Array with last-2-axes contiguity guarantee */
  using L2Contiguous = Array<T, N, FLAGS | AF_LAST2_CONTIGUOUS>;

  /** Type alias for Array with full contiguity guarantee */
  using Contiguous = Array<T, N, FLAGS | AF_CONTIGUOUS>;

  /** Type alias for Array<int64_t, 1> used for metadata (sizes/strides) */
  using IntsMetaRef = Array<int64_t, 1, AF_CONTIGUOUS>;

  /** The element type T */
  using ElementType = T;

  /**
   * @brief Constructs an Array view with explicit sizes and strides.
   *
   * Creates a view over existing memory with custom stride information.
   * This allows creating views of non-contiguous memory layouts.
   *
   * @param data Pointer to the underlying data. Can be device memory pointer.
   * @param sizes Pointer to array of N dimension sizes.
   * @param strides Pointer to array of N dimension strides (in elements).
   *
   * @pre data != nullptr (assertion failure otherwise)
   * @pre sizes != nullptr (assertion failure otherwise)
   * @pre strides != nullptr (assertion failure otherwise)
   * @pre All sizes[i] >= 0 (checked in debug mode)
   * @pre If FLAGS include contiguity flags, strides must match the guarantee
   *      (checked in debug mode)
   *
   * @warning The sizes and strides arrays are COPIED into the Array object.
   *          The caller does not need to keep the input arrays alive.
   *
   * @note The data pointer is stored as mutable internally to support both
   *       const and non-const access patterns, but const-correctness is
   *       maintained through the API.
   */
  CT_ALWAYS_FORCEINLINE
  constexpr Array(
      const T* data, // may be device ptr
      const int64_t* sizes,
      const int64_t* strides
  ) : _data((T*) data) {
    CT_ASSERT(data != nullptr, "data ptr should not be null");
    CT_ASSERT(sizes != nullptr, "sizes ptr should not be null");
    CT_ASSERT(strides != nullptr, "strides ptr should not be null");

    for (int i = 0; i < N; ++i) {
      _sizes[i] = sizes[i];
      _strides[i] = strides[i];
    }
    _check_meta_sanity();
  }

  /**
   * @brief Constructs an Array view with contiguous memory layout.
   *
   * Creates a view assuming row-major contiguous memory layout.
   * Strides are computed automatically from sizes in row-major order:
   *   stride[N-1] = 1
   *   stride[N-2] = _size[N-1]
   *   stride[N-3] = size[N-1] * _size[N-2]
   *   ... etc
   *
   * @param data Pointer to the underlying data.
   * @param sizes Pointer to array of N dimension sizes.
   *
   * @pre data != nullptr (assertion failure otherwise)
   * @pre sizes != nullptr (assertion failure otherwise)
   * @pre All sizes[i] >= 0 (checked in debug mode)
   * @pre The memory layout must actually be contiguous
   *
   * @note Consider using as_contiguous() after construction if the memory
   *       is known to be contiguous, to enable compile-time optimizations.
   */
  CT_ALWAYS_FORCEINLINE
  constexpr Array(
      const T* data,
      const int64_t* sizes
  ) : _data((T*) data) {
    CT_ASSERT(data != nullptr, "data ptr should not be null");
    CT_ASSERT(sizes != nullptr, "sizes ptr should not be null");

    int64_t numel = 1;
    for (int64_t d = N - 1; d >= 0; --d) {
      _strides[d] = numel;
      _sizes[d] = sizes[d];
      numel *= sizes[d];
    }
    _check_meta_sanity();
  }

  /**
   * @brief Constructs an Array from initializer lists for sizes and strides.
   *
   * Convenience constructor for inline array creation.
   *
   * @param data Pointer to the underlying data.
   * @param sizes Initializer list of N dimension sizes.
   * @param strides Initializer list of N dimension strides.
   *
   * @pre sizes._size() == N (assertion failure otherwise)
   * @pre strides._size() == N (assertion failure otherwise)
   *
   * @example
   *   float data[24];
   *   Array<float, 2> arr(data, {4, 6}, {6, 1});
   */
  CT_ALWAYS_FORCEINLINE
  constexpr Array(
      const T* data,
      std::initializer_list<int64_t> sizes,
      std::initializer_list<int64_t> strides
  ) : Array(data, _get_ints_meta_size_checked(sizes, "sizes._size()"), _get_ints_meta_size_checked(strides, "strides._size()")) {}

  /**
   * @brief Constructs a contiguous Array from an initializer list of sizes.
   *
   * Convenience constructor for inline contiguous array creation.
   *
   * @param data Pointer to the underlying data.
   * @param sizes Initializer list of N dimension sizes.
   *
   * @pre sizes._size() == N (assertion failure otherwise)
   *
   * @example
   *   float data[24];
   *   Array<float, 2> arr(data, {4, 6});  // Creates 4x6 contiguous array
   */
  CT_ALWAYS_FORCEINLINE
  constexpr Array(
      const T* data,
      std::initializer_list<int64_t> sizes
  ) : Array(data, _get_ints_meta_size_checked(sizes, "sizes._size()")) {}

  /**
   * @brief Constructs an Array from std::array for sizes and strides.
   *
   * Type-safe constructor that ensures correct array _size at compile time.
   *
   * @param data Pointer to the underlying data.
   * @param sizes std::array of N dimension sizes.
   * @param strides std::array of N dimension strides.
   */
  CT_ALWAYS_FORCEINLINE
  constexpr Array(
      const T* data,
      std::array<int64_t, N> sizes,
      std::array<int64_t, N> strides
  ) : Array(data, sizes.data(), strides.data()) {}

  /**
   * @brief Constructs a contiguous Array from std::array of sizes.
   *
   * Type-safe constructor that ensures correct array _size at compile time.
   *
   * @param data Pointer to the underlying data.
   * @param sizes std::array of N dimension sizes.
   */
  CT_ALWAYS_FORCEINLINE
  constexpr Array(
      const T* data,
      std::array<int64_t, N> sizes
  ) : Array(data, sizes.data()) {}

  /**
   * @brief Constructs an Array from IntsMetaRef for sizes and strides.
   *
   * Allows dynamic metadata (sizes/strides) from other Array objects.
   *
   * @param data Pointer to the underlying data.
   * @param sizes Array view containing N dimension sizes.
   * @param strides Array view containing N dimension strides.
   *
   * @pre sizes._size(0) == N (assertion failure otherwise)
   * @pre strides._size(0) == N (assertion failure otherwise)
   */
  CT_ALWAYS_FORCEINLINE
  constexpr Array(
      const T* data,
      IntsMetaRef sizes,
      IntsMetaRef strides
  ) : Array(data, _get_ints_meta_size_checked(sizes, "sizes._size(0)"), _get_ints_meta_size_checked(strides, "strides._size(0)")) {}

  /**
   * @brief Constructs a contiguous Array from IntsMetaRef of sizes.
   *
   * Allows dynamic metadata from other Array objects.
   *
   * @param data Pointer to the underlying data.
   * @param sizes Array view containing N dimension sizes.
   *
   * @pre sizes._size(0) == N (assertion failure otherwise)
   */
  CT_ALWAYS_FORCEINLINE
  constexpr Array(
      const T* data,
      IntsMetaRef sizes
  ) : Array(data, _get_ints_meta_size_checked(sizes, "sizes._size(0)")) {}

  /**
   * @brief Copy constructor.
   *
   * Creates a shallow copy of the Array view. Both the original and the copy
   * will reference the same underlying data.
   *
   * @param other The Array to copy from.
   *
   * @note This is a shallow copy - the data pointer is copied, not the data.
   */
  CT_ALWAYS_FORCEINLINE
  constexpr Array(
      const Array& other
  ) : Array(other._data, other._sizes, other._strides) {}

  /**
   * @brief Returns the _size of a specific dimension.
   *
   * @param i Dimension index (0-based).
   * @return The number of elements along dimension i.
   *
   * @pre 0 <= i < N (assertion failure otherwise)
   *
   * @note This is a constexpr operation with zero runtime overhead.
   */
  CT_ALWAYS_FORCEINLINE
  int64_t size(int64_t i) const {
    CT_ASSERT(0 <= i && i < N, "n !in 0:N");
    return _sizes[i];
  }

  /**
   * @brief Returns the stride of a specific dimension.
   *
   * Stride is the number of elements to skip when moving one position
   * along the given dimension.
   *
   * @param i Dimension index (0-based).
   * @return The stride (in elements, not bytes) for dimension i.
   *
   * @pre 0 <= i < N (assertion failure otherwise)
   *
   * @note A stride of 0 indicates a broadcasted dimension.
   */
  CT_ALWAYS_FORCEINLINE
  int64_t stride(int64_t i) const {
    CT_ASSERT(0 <= i && i < N, "n !in 0:N");
    return _strides[i];
  }

  /**
   * @brief Returns a view of all dimension sizes.
   *
   * @return A 1D Array<int64_t> of length N containing all dimension sizes.
   *
   * @note The returned view is guaranteed contiguous.
   */
  CT_ALWAYS_FORCEINLINE
  const IntsMetaRef sizes() const {
    return {_sizes, {N}, {int64_t(1)}};
  }

  /**
   * @brief Returns a view of all dimension strides.
   *
   * @return A 1D Array<int64_t> of length N containing all dimension strides.
   *
   * @note The returned view is guaranteed contiguous.
   */
  CT_ALWAYS_FORCEINLINE
  const IntsMetaRef strides() const {
    return {_strides, {N}, {int64_t(1)}};
  }

  /**
   * @brief Returns a pointer to the underlying data.
   *
   * @return Mutable pointer to the first element.
   *
   * @warning Modifying data through this pointer affects all Array views
   *          referencing the same memory.
   */
  CT_ALWAYS_FORCEINLINE
  T* data() {
    return _data;
  }

  /**
   * @brief Returns a const pointer to the underlying data.
   *
   * @return Const pointer to the first element.
   */
  CT_ALWAYS_FORCEINLINE
  const T* data() const {
    return _data;
  }

  /**
   * @brief Returns the total number of elements in the array.
   *
   * Computed as the product of all dimension sizes.
   *
   * @return Total element count (product of sizes).
   *
   * @note For a non-contiguous array, numel() may not equal the actual
   *       number of unique memory locations accessed (due to overlapping
   *       strides or broadcasting).
   *
   * @warning This is NOT cached and recomputes on every call.
   *          Store the result if called repeatedly.
   */
  int64_t numel() const {
    int64_t numel = _sizes[0];
    for (int i = 1; i < N; ++i) {
      numel *= _sizes[i];
    }
    return numel;
  }

  /**
   * @brief Returns the number of dimensions.
   *
   * @return The compile-time constant N.
   *
   * @note This is a constexpr function - the value is known at compile time.
   */
  CT_ALWAYS_FORCEINLINE
  constexpr int64_t ndim() const {
    return N;
  }

  /**
   * @brief Checks if the last axis is contiguous in memory.
   *
   * The last axis is contiguous if stride[N-1] == 1.
   *
   * @return true if the last dimension is contiguous, false otherwise.
   *
   * @note If FLAGS includes AF_LAST_CONTIGUOUS, this returns true at
   *       compile time without runtime computation.
   */
  CT_ALWAYS_FORCEINLINE
  constexpr bool is_last_contiguous() const {
    if constexpr (N < 1) return false; // note: actually unreachable
    if constexpr ((FLAGS & AF_LAST_CONTIGUOUS) == AF_LAST_CONTIGUOUS) return true;
    return _strides[N - 1] == 1;
  }

  /**
   * @brief Checks if the last two axes are contiguous in memory.
   *
   * The last two axes are contiguous if:
   *   - stride[N-1] == 1
   *   - stride[N-2] == _size[N-1]
   *
   * @return true if the last two dimensions are contiguous, false otherwise.
   *
   * @note Returns false for N < 2.
   * @note If FLAGS includes AF_LAST2_CONTIGUOUS, this returns true at
   *       compile time without runtime computation.
   */
  CT_ALWAYS_FORCEINLINE
  constexpr bool is_last2_contiguous() const {
    if constexpr (N < 2) return false;
    if constexpr ((FLAGS & AF_LAST2_CONTIGUOUS) == AF_LAST2_CONTIGUOUS) return true;
    return _strides[N - 1] == 1 && _strides[N - 2] == _sizes[N - 1];
  }

  /**
   * @brief Checks if the entire array is contiguous in memory.
   *
   * An array is fully contiguous if it follows row-major layout:
   *   stride[i] = product of sizes[i+1:] for all dimensions
   *
   * @return true if the array is fully contiguous, false otherwise.
   *
   * @note If FLAGS includes AF_CONTIGUOUS, this returns true at
   *       compile time without runtime computation.
   */
  constexpr bool is_contiguous() const {
    if constexpr ((FLAGS & AF_CONTIGUOUS) == AF_CONTIGUOUS) return true;
    int64_t numel = 1;
    for (int64_t d = N - 1; d >= 0; --d) {
      if (_strides[d] != numel) return false;
      numel *= _sizes[d];
    }
    return true;
  }

  /**
   * @brief Complex slicing and indexing operator.
   *
   * Supports multiple indexing modes:
   *   1. Integer index: Selects a single element along that axis, reducing dimensionality
   *   2. reserve: Keeps the axis in the result with _size 1
   *   3. new_axis(): Adds a new dimension of _size 1
   *   4. range(start, end, step): Selects a slice of elements along an axis
   *
   * @tparam TIndices Variadic template for index types.
   * @param indices Index specifiers for each dimension.
   * @return Depends on the index types:
   *         - If all indices are integers (full indexing): returns T& (element reference)
   *         - Otherwise: returns Array<T, NewN, NewFlags> (sub-array view)
   *
   * @section behavior Behavior Details
   *   - Integer indices reduce dimensionality (N -> N-1)
   *   - reserve preserves dimension (N -> N, with _size 1 at indexed position)
   *   - new_axis() increases dimensionality (N -> N+1)
   *   - range preserves dimension with modified _size
   *
   * @section contiguity Contiguity Preservation
   *   - Integer indexing may preserve contiguity flags
   *   - range breaks all contiguity guarantees except where proven safe
   *   - new_axis() breaks contiguity (new axis has stride 0)
   *   - reserve preserves contiguity
   *
   * @example
   *   Array<float, 3> arr = ...;  // shape [4, 5, 6]
   *   float& elem = arr(1, 2, 3);           // Single element
   *   auto view1 = arr(1, reserve, reserve);  // shape [1, 5, 6]
   *   auto view2 = arr(range(0, 2), reserve, reserve);  // shape [2, 5, 6]
   *   auto view3 = arr(new_axis(), reserve, reserve, reserve);  // shape [1, 4, 5, 6]
   *
   * @note All index validation occurs at runtime (assertions in debug mode).
   */
  template <typename... TIndices>
  constexpr decltype(auto) operator()(TIndices... indices) const;

  /**
   * @brief Single-argument indexing operator (equivalent to operator()).
   *
   * Provided for compatibility with single-index access patterns.
   *
   * @tparam TIndex Index type.
   * @param index Index specifier for the first dimension.
   * @return Same as operator()(index).
   */
  template <typename TIndex>
  CT_ALWAYS_FORCEINLINE
  constexpr auto operator[](TIndex index) const {
    return this->operator()(index);
  }

  /**
   * @brief Casts this Array to an Array with last-axis contiguity guarantee.
   *
   * Use this when you know (from external knowledge) that the last axis
   * is contiguous, to enable compile-time optimizations.
   *
   * @return Array<T, N, FLAGS | AF_LAST_CONTIGUOUS>
   *
   * @warning This is an assertion cast. If the last axis is NOT actually
   *          contiguous, behavior is undefined (assertion in debug mode).
   *
   * @note The returned Array shares the same data pointer.
   */
  CT_ALWAYS_FORCEINLINE
  constexpr Array::L1Contiguous as_last_contiguous() const {
    return {_data, _sizes, _strides}; // assertion inside constructor
  }

  /**
   * @brief Casts this Array to an Array with last-2-axes contiguity guarantee.
   *
   * Use this when you know (from external knowledge) that the last two axes
   * are contiguous, to enable compile-time optimizations.
   *
   * @return Array<T, N, FLAGS | AF_LAST2_CONTIGUOUS>
   *
   * @warning This is an assertion cast. If the last two axes are NOT actually
   *          contiguous, behavior is undefined (assertion in debug mode).
   *
   * @note The returned Array shares the same data pointer.
   */
  CT_ALWAYS_FORCEINLINE
  constexpr Array::L2Contiguous as_last2_contiguous() const {
    return {_data, _sizes, _strides}; // assertion inside constructor
  }

  /**
   * @brief Casts this Array to an Array with full contiguity guarantee.
   *
   * Use this when you know (from external knowledge) that the entire array
   * is contiguous in memory, to enable maximum compile-time optimizations.
   *
   * @return Array<T, N, FLAGS | AF_CONTIGUOUS>
   *
   * @warning This is an assertion cast. If the array is NOT actually
   *          fully contiguous, behavior is undefined (assertion in debug mode).
   *
   * @note The returned Array shares the same data pointer.
   */
  CT_ALWAYS_FORCEINLINE
  constexpr Array::Contiguous as_contiguous() const {
    return {_data, _sizes, _strides}; // assertion inside constructor
  }

private:
  CT_ALWAYS_FORCEINLINE
  void _check_stride_for_contiguity() const {
    #ifdef CT_DEBUG
    constexpr int contiguity_flag = FLAGS & _AF_ALL_CONTIGUITY_FLAGS;
    if constexpr (contiguity_flag != 0) {
      int64_t limit;
      if constexpr (contiguity_flag == AF_CONTIGUOUS) {
        limit = 0;
      } else if constexpr (contiguity_flag == AF_LAST2_CONTIGUOUS) {
        static_assert(N >= 2, "array with Ndim < 2 cannot be last 2 contiguous");
        limit = N - 2;
      } else if constexpr (contiguity_flag == AF_LAST_CONTIGUOUS) {
        static_assert(N >= 1, "array with Ndim < 1 cannot be last contiguous"); // note: actually unreachable
        limit = N - 1;
      } else {
        static_assert(!std::is_same_v<T, T>, "Invalid contiguity flag!");
      }
      int64_t numel = 1;
      for (int64_t d = N - 1; d >= limit; --d) {
        CT_ASSERT(_strides[d] == numel, "dim %lld of array is not contiguous (expected %lld, got %lld)", d, numel, _strides[d]);
        numel *= _sizes[d];
      }
    }
    #endif
  }

  CT_ALWAYS_FORCEINLINE
  void _check_meta_sanity() const {
    #ifdef CT_DEBUG
    for (int64_t i = 0; i < N; ++i) {
      CT_ASSERT(_sizes[i] >= 0, "dim %lld of array have negative _size", i);
    }
    _check_stride_for_contiguity();
    #endif
  }

  constexpr static const int64_t* _get_ints_meta_size_checked(std::initializer_list<int64_t> x, const char* name) {
    CT_ASSERT(x.size() == N, "%s(%lld) != N(%lld)", name, x.size(), N);
    return x.begin();
  }

  constexpr static const int64_t* _get_ints_meta_size_checked(const IntsMetaRef& x, const char* name) {
    CT_ASSERT(x.size(0) == N, "%s(%lld) != N(%lld)", name, x.size(0), N);
    return x.data();
  }

  T* _data = nullptr;
  int64_t _sizes[N];
  int64_t _strides[N];
}; // class Array

// Compile-time slice helpers
namespace details {
template <
    typename T,
    int64_t N,       // num of dim of original array
    int64_t D,       // index of dim of original array this slicer currently handle
    int FLAGS,       // flags of original arry
    typename = void, // SFINAE
    typename... TIndices
>
struct SliceHelper;

template <
    typename T,
    int64_t N,
    int64_t D,
    int FLAGS
>
struct SliceHelper<T, N, D, FLAGS> {
  static constexpr int64_t NewLength = N;
  static constexpr int64_t LastNContiguous = N - D;

  CT_ALWAYS_FORCEINLINE int64_t apply(
      const int64_t* old_sizes, const int64_t* old_strides,
      int64_t* new_sizes, int64_t* new_strides
  ) {
    int64_t rem_n = N - D;
    for (int i = 0; i < rem_n; ++i) {
      new_sizes[i] = old_sizes[i];
      new_strides[i] = old_strides[i];
    }
    return 0;
  }
};

template <
    typename T,
    int64_t N,
    int64_t D,
    int FLAGS,
    typename TIndex,
    typename... TIndices
>
struct SliceHelper<T, N, D, FLAGS, std::enable_if_t<std::is_integral_v<std::decay_t<TIndex>>>, TIndex, TIndices...> {
  using Next = SliceHelper<T, N, D + 1, FLAGS, void, TIndices...>;
  static constexpr int64_t NewLength = Next::NewLength - 1;
  static_assert(NewLength >= 0);
  static constexpr int64_t LastNContiguous = Next::LastNContiguous; // breaks contiguity

  CT_ALWAYS_FORCEINLINE int64_t apply(
      const int64_t* old_sizes, const int64_t* old_strides,
      int64_t* new_sizes, int64_t* new_strides,
      TIndex i0, TIndices... iargs
  ) {
    int64_t index = int64_t(i0);
    CT_ASSERT(0 <= index && index < old_sizes[0], "%lld !in 0:%lld", index, old_sizes[0]);
    int64_t offset;
    // last dimensional contiguity optimization: skip multiplication of 1
    if constexpr (D == N - 1 && (FLAGS & AF_LAST_CONTIGUOUS) != 0) {
      offset = index;
    } else {
      offset = index * old_strides[0];
    }
    return Next().apply(
        old_sizes + 1, old_strides + 1,
        new_sizes, new_strides, iargs...
    ) + offset;
  }
};

template <
    typename T,
    int64_t N,
    int64_t D,
    int FLAGS,
    typename TIndex,
    typename... TIndices
>
struct SliceHelper<T, N, D, FLAGS, std::enable_if_t<std::is_same_v<std::decay_t<TIndex>, details::ReserveAxis>>, TIndex, TIndices...> {
  using Next = SliceHelper<T, N, D + 1, FLAGS, void, TIndices...>;
  static constexpr int64_t NewLength = Next::NewLength;
  static_assert(NewLength >= 0);
  static constexpr int64_t LastNContiguous = Next::LastNContiguous == N - D - 1 ? Next::LastNContiguous + 1 : Next::LastNContiguous;

  CT_ALWAYS_FORCEINLINE int64_t apply(
      const int64_t* old_sizes, const int64_t* old_strides,
      int64_t* new_sizes, int64_t* new_strides,
      TIndex i0, TIndices... iargs
  ) {
    new_sizes[0] = old_sizes[0];
    new_strides[0] = old_strides[0];
    return Next().apply(
        old_sizes + 1, old_strides + 1,
        new_sizes + 1, new_strides + 1, iargs...
    );
  }
};

template <
    typename T,
    int64_t N,
    int64_t D,
    int FLAGS,
    typename TIndex,
    typename... TIndices
>
struct SliceHelper<T, N, D, FLAGS, std::enable_if_t<std::is_same_v<std::decay_t<TIndex>, details::NewAxis>>, TIndex, TIndices...> {
  using Next = SliceHelper<T, N, D, FLAGS, void, TIndices...>;
  static constexpr int64_t NewLength = Next::NewLength + 1;
  static_assert(NewLength >= 0);
  static constexpr int64_t LastNContiguous = Next::LastNContiguous; // breaks contiguity

  CT_ALWAYS_FORCEINLINE int64_t apply(
      const int64_t* old_sizes, const int64_t* old_strides,
      int64_t* new_sizes, int64_t* new_strides,
      TIndex i0, TIndices... iargs
  ) {
    new_sizes[0] = i0.repeat;
    new_strides[0] = 0;
    return Next().apply(
        old_sizes, old_strides,
        new_sizes + 1, new_strides + 1, iargs...
    );
  }
};

template <
    typename T,
    int64_t N,
    int64_t D,
    int FLAGS,
    typename TIndex,
    typename... TIndices
>
struct SliceHelper<T, N, D, FLAGS, std::enable_if_t<std::is_same_v<std::decay_t<TIndex>, details::Range>>, TIndex, TIndices...> {
  using Next = SliceHelper<T, N, D + 1, FLAGS, void, TIndices...>;
  static constexpr int64_t NewLength = Next::NewLength;
  static_assert(NewLength >= 0);
  static constexpr int64_t LastNContiguous = Next::LastNContiguous; // breaks contiguity

  CT_ALWAYS_FORCEINLINE int64_t apply(
      const int64_t* old_sizes, const int64_t* old_strides,
      int64_t* new_sizes, int64_t* new_strides,
      TIndex i0, TIndices... iargs
  ) {
    int64_t from = i0.start;
    int64_t to = i0.end;
    int64_t step = i0.step;
    int64_t size;
    CT_ASSERT(0 <= from && from < old_sizes[0], "%lld !in 0:%lld", from, old_sizes[0]);
    CT_ASSERT(0 <= to && to <= old_sizes[0], "%lld !in 0:%lld", to, old_sizes[0]);
    if (step > 0) {
      CT_ASSERT(from < to, "for positive step, from must be less than to");
      size = (to - from + step - 1) / step;
    } else if (step < 0) {
      CT_ASSERT(from > to, "for negative step, from must be greater than to");
      size = (from - to - step - 1) / (-step);
    } else {
      CT_ASSERT(false, "step cannot be zero");
    }
    new_sizes[0] = size;
    new_strides[0] = old_strides[0] * step;
    return Next().apply(
        old_sizes + 1, old_strides + 1,
        new_sizes + 1, new_strides + 1, iargs...
    ) + from * old_strides[0];
  }
};

CT_ALWAYS_FORCEINLINE constexpr int get_new_flags(int64_t last_n_contiguous, int64_t N, int flags) {
  if (last_n_contiguous == N) {
    return flags; // keeps all contiguity flags
  } else if (last_n_contiguous >= 2 && N >= 2) {
    return flags & ~(_AF_ALL_CONTIGUITY_FLAGS & ~AF_LAST2_CONTIGUOUS); // preserves at most last 2 contiguity
  } else if (last_n_contiguous >= 1 && N >= 1) {
    return flags & ~(_AF_ALL_CONTIGUITY_FLAGS & ~AF_LAST_CONTIGUOUS); // preserves at most last 1 contiguity
  } else {
    return flags & (~_AF_ALL_CONTIGUITY_FLAGS); // clears
  }
}
} // namespace details

template <typename T, int64_t N, int FLAGS>
template <typename... TIndices>
CT_ALWAYS_FORCEINLINE
constexpr decltype(auto) Array<T, N, FLAGS>::operator()(TIndices... indices) const {
  using Helper = typename details::SliceHelper<T, N, 0, FLAGS, void, TIndices...>;
  constexpr int64_t new_length = Helper::NewLength;

  std::array<int64_t, new_length> new_sizes, new_strides;
  int64_t offset = Helper().apply(sizes().data(), strides().data(), new_sizes.data(), new_strides.data(), indices...);
  T* ptr = (T*) data() + offset;

  if constexpr (new_length == 0) {
    // reads element (as reference, won't actually cause a memory read unless caller receives it using a value)
    return (T&) *ptr;
  } else {
    constexpr int64_t last_n_contiguous = Helper::LastNContiguous;
    constexpr int new_flags = details::get_new_flags(last_n_contiguous, new_length, FLAGS);
    return Array<T, new_length, new_flags>(ptr, new_sizes, new_strides);
  }
}

/**
 * @brief Transposes two dimensions of an array.
 *
 * Creates a view with dimensions i and j swapped. This is a zero-copy operation
 * that only modifies the stride information.
 *
 * @tparam T Element type.
 * @tparam N Number of dimensions.
 * @tparam FLAGS Array flags.
 * @param x The input array.
 * @param i First dimension to swap.
 * @param j Second dimension to swap.
 * @return A new Array view with dimensions i and j transposed.
 *
 * @pre 0 <= i < N (assertion failure otherwise)
 * @pre 0 <= j < N (assertion failure otherwise)
 *
 * @note The returned array loses ALL contiguity flags because transposition
 *       generally breaks contiguity guarantees.
 * @note If i == j, the returned array is identical to the input (no-op).
 *
 * @example
 *   Array<float, 3> arr = ...;  // shape [2, 3, 4]
 *   auto transposed = transpose(arr, 0, 2);  // shape [4, 3, 2]
 */
template <typename T, int64_t N, int FLAGS>
CT_ALWAYS_FORCEINLINE
constexpr Array<T, N, FLAGS> transpose(const Array<T, N, FLAGS>& x, int64_t i, int64_t j) {
  CT_ASSERT(0 <= i && i < N, "%lld !in 0:%lld", i, N);
  CT_ASSERT(0 <= j && j < N, "%lld !in 0:%lld", j, N);
  std::array<int64_t, N> new_sizes, new_strides;
  const int64_t* old_sizes = x.sizes().data();
  const int64_t* old_strides = x.strides().data();

  for (int k = 0; k < N; ++k) {
    new_sizes[k] = old_sizes[k];
    new_strides[k] = old_strides[k];
  }
  std::swap(new_sizes[i], new_sizes[j]);
  std::swap(new_strides[i], new_strides[j]);

  constexpr int new_flags = FLAGS & ~_AF_ALL_CONTIGUITY_FLAGS; // clears all contiguity flags
  return Array<T, N, new_flags>(x.data(), new_sizes, new_strides);
}

/**
 * @brief Permutes the dimensions of an array according to the specified order.
 *
 * Creates a view with dimensions rearranged according to the provided indices.
 * This is a generalization of transpose() that allows arbitrary dimension ordering.
 *
 * @tparam T Element type.
 * @tparam N Number of dimensions.
 * @tparam FLAGS Array flags.
 * @tparam TIndices Variadic template for dimension indices.
 * @param x The input array.
 * @param indices N dimension indices specifying the new order.
 *                Each index must appear exactly once.
 * @return A new Array view with permuted dimensions.
 *
 * @pre sizeof...(TIndices) == N (compile-time error otherwise)
 * @pre Each index must be in range [0, N)
 * @pre Each index must appear exactly once (checked in debug mode)
 *
 * @note The returned array loses ALL contiguity flags because permutation
 *       generally breaks contiguity guarantees.
 *
 * @example
 *   Array<float, 4> arr = ...;  // shape [1, 2, 3, 4]
 *   auto permuted = permute(arr, 3, 2, 1, 0);  // shape [4, 3, 2, 1]
 *   auto permuted2 = permute(arr, 0, 2, 3, 1); // shape [1, 3, 4, 2]
 */
template <typename T, int64_t N, int FLAGS, typename... TIndices>
CT_ALWAYS_FORCEINLINE
constexpr Array<T, N, FLAGS> permute(const Array<T, N, FLAGS>& x, TIndices... indices) {
  static_assert(sizeof...(TIndices) == N, "insufficient number of indices");
  std::array<int64_t, N> index_array = {static_cast<int64_t>(indices)...};
  std::array<int64_t, N> new_sizes, new_strides;
  const int64_t* old_sizes = x.sizes().data();
  const int64_t* old_strides = x.strides().data();
  #ifdef CT_DEBUG
  std::array<bool, N> is_visited;
  is_visited.fill(false);
  #endif

  for (int64_t i = 0; i < N; ++i) {
    int64_t j = index_array[i];
    CT_ASSERT(0 <= j && j < N, "indices[%lld]: %lld !in 0:%lld", i, j, N);
    CT_ASSERT(!is_visited[j], "indices[%lld]: index %lld is being used twice", i, j);
    #ifdef CT_DEBUG
    is_visited[j] = true;
    #endif
    new_sizes[i] = old_sizes[j];
    new_strides[i] = old_strides[j];
  }

  #ifdef CT_DEBUG
  for (int64_t i = 0; i < N; ++i) {
    CT_ASSERT(is_visited[i], "index %lld is never used", i);
  }
  #endif

  constexpr int new_flags = FLAGS & ~_AF_ALL_CONTIGUITY_FLAGS; // clears all contiguity flags
  return Array<T, N, new_flags>(x.data(), new_sizes, new_strides);
}

}} // namespace ct::tl::array

#endif //CTORCH_ARRAY_H
