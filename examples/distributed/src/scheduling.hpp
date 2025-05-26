#pragma once

#include <hpx/async_distributed/dataflow.hpp>
#include <hpx/naming_base/id_type.hpp>

template <auto F>
struct plain_action_for;

// This is a simple tag-type like construct that exists solely so we automatically pick the right dataflow() overload.
struct schedule_on_locality
{
    // conversion is intended here, we don't want people to actually spell this type out
    // ReSharper disable once CppNonExplicitConvertingConstructor
    schedule_on_locality(const hpx::id_type &where) :
        where(where)
    { }

    hpx::id_type where;
};

template <auto F, typename... Args>
decltype(auto) dataflow(const schedule_on_locality &on, Args &&...args)
{
    typename plain_action_for<F>::action_type act;
    return hpx::dataflow(act, on.where, args...);
}
