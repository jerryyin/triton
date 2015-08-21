#ifndef ISAAC_DRIVER_EVENT_H
#define ISAAC_DRIVER_EVENT_H

#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/handle.h"

namespace isaac
{

namespace driver
{

// Event
class ISAACAPI Event
{
  friend class CommandQueue;
public:
  Event(cl_event const & event, bool take_ownership = true);
  Event(backend_type backend);
  long elapsed_time() const;
  HANDLE_TYPE(cl_event, cu_event_t)& handle();
private:
  backend_type backend_;
  HANDLE_TYPE(cl_event, cu_event_t) h_;
};

}

}

#endif
