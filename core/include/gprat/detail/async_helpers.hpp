#ifndef GPRAT_DETAIL_DATAFLOW_HELPERS_HPP
#define GPRAT_DETAIL_DATAFLOW_HELPERS_HPP

#pragma once

#include "gprat/detail/config.hpp"

#include <hpx/async_base/dataflow.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/threading_base/annotated_function.hpp>

GPRAT_NS_BEGIN

namespace detail
{

template <auto F, typename... Args>
decltype(auto) named_dataflow(const char *name, Args &&...args)
{
    return hpx::dataflow(hpx::annotated_function(hpx::unwrapping(F), name), std::forward<Args>(args)...);
}

template <auto F, typename... Args>
decltype(auto) named_async(const char *name, Args &&...args)
{
    return hpx::async(hpx::annotated_function(F, name), std::forward<Args>(args)...);
}

}  // namespace detail

GPRAT_NS_END

#endif
