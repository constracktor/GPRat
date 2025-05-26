#include "distributed_tile.hpp"

// The macros below are necessary to generate the code required for exposing
// our partition type remotely.
//
// HPX_REGISTER_COMPONENT() exposes the component creation
// through hpx::new_<>().
typedef hpx::components::component<tile_server> tile_server_type;
HPX_REGISTER_COMPONENT(tile_server_type, tile_server)

// HPX_REGISTER_ACTION() exposes the component member function for remote
// invocation.
typedef tile_server::get_data_action get_data_action;
HPX_REGISTER_ACTION(get_data_action)

typedef tile_server::set_data_action set_data_action;
HPX_REGISTER_ACTION(set_data_action)
