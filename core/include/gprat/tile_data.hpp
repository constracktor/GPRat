#ifndef GPRAT_TILE_DATA_HPP
#define GPRAT_TILE_DATA_HPP

#pragma once

#include "gprat/detail/config.hpp"
#include "gprat/performance_counters.hpp"

#include <hpx/serialization/serialize_buffer.hpp>
#include <span>

GPRAT_NS_BEGIN

/**
 * @brief Non-mutable reference-counted dynamic array of a given type T.
 * This class represents a simple reference-counted non-resizeable buffer with elements of type T.
 * It can be serialized by HPX and thus be used as a parameter for HPX actions.
 * This type is intended to be used for parameters and attributes that do not require mutable data (i.e., only read
 * access)
 *
 * @tparam T Element type of the tile. Usually some numeric type like double or float. This class currently only
 * requires T to be serializable by HPX.
 */
template <typename T>
class const_tile_data
{
  protected:
    typedef hpx::serialization::serialize_buffer<T> cpu_buffer_type;

    struct hold_reference
    {
        explicit hold_reference(const cpu_buffer_type &data) :
            data_(data)
        { }

        void operator()(const T *) const { }  // no deletion necessary

        cpu_buffer_type data_;
    };

    // In case we want pooling down the road...
    static T *allocate(std::size_t n)
    {
        track_tile_data_allocation(n);
        return new T[n];
    }

    static void deallocate(T *p) noexcept
    {
        track_tile_data_deallocation(0);  // we don't know here
        delete[] p;
    }

  public:
    const_tile_data() = default;

    // Create a new (uninitialized) tile_data of the given size.
    explicit const_tile_data(std::size_t size) :
        cpu_data_(allocate(size), size, cpu_buffer_type::take, &const_tile_data::deallocate)
    { }

    // Create a tile_data which acts as a proxy to a part of the embedded array.
    // The proxy is assumed to refer to either the left or the right boundary
    // element.
    const_tile_data(const const_tile_data &base, std::size_t offset, std::size_t size) :
        cpu_data_(base.cpu_data_.data() + offset,
                  size,
                  cpu_buffer_type::reference,
                  hold_reference(base.cpu_data_))  // keep referenced tile_data alive
    { }

    [[nodiscard]] const T *data() const noexcept { return cpu_data_.data(); }

    [[nodiscard]] std::size_t size() const noexcept { return cpu_data_.size(); }

    [[nodiscard]] const T *begin() const noexcept { return cpu_data_.data(); }

    [[nodiscard]] const T *end() const noexcept { return cpu_data_.data() + cpu_data_.size(); }

    [[nodiscard]] const T &operator[](std::size_t idx) const { return cpu_data_[idx]; }

    // ReSharper disable once CppNonExplicitConversionOperator
    operator std::span<const T>() const noexcept  // NOLINT(*-explicit-constructor)
    {
        return { cpu_data_.data(), cpu_data_.size() };
    }

  protected:
    // Serialization support: even if all of the code below runs on one
    // locality only, we need to provide an (empty) implementation for the
    // serialization as all arguments passed to actions have to support this.
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int)
    {
        // clang-format off
        ar & cpu_data_;
        // clang-format on
    }

    cpu_buffer_type cpu_data_;
};

/**
 * A mutable version of const_tile_data.
 *
 * @tparam T Element type of the tile. See @ref const_tile_data
 */
template <typename T>
class mutable_tile_data : public const_tile_data<T>
{
  public:
    using const_tile_data<T>::const_tile_data;

    [[nodiscard]] T *data() const noexcept { return const_cast<T *>(this->cpu_data_.data()); }

    [[nodiscard]] T *begin() const noexcept { return const_cast<T *>(this->cpu_data_.data()); }

    [[nodiscard]] T *end() const noexcept { return const_cast<T *>(this->cpu_data_.data()) + this->cpu_data_.size(); }

    [[nodiscard]] T &operator[](std::size_t idx) const { return this->cpu_data_[idx]; }

    // ReSharper disable once CppNonExplicitConversionOperator
    operator std::span<T>() noexcept
    {
        return { this->cpu_data_.data(), this->cpu_data_.size() };
    }  // NOLINT(*-explicit-constructor)
};

GPRAT_NS_END

#endif
