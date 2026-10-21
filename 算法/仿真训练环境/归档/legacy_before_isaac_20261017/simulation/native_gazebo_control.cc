// Small C ABI around Gazebo Transport's blocking C++ service client.
// Calling this library through ctypes.CDLL releases Python's GIL while the
// synchronous request waits, so Python sensor subscriptions can keep running.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/world_control.pb.h>
#include <gz/transport/Node.hh>
#include <gz/transport/NodeOptions.hh>

namespace
{

class ControlClient
{
  public: ControlClient(const std::string &_partition,
                        const std::string &_service)
      : service(_service)
  {
    gz::transport::NodeOptions options;
    if (!options.SetPartition(_partition))
      throw std::runtime_error("Gazebo Transport rejected the partition name");
    this->node = std::make_unique<gz::transport::Node>(options);
  }

  public: bool Request(uint32_t _steps, uint32_t _timeoutMs,
                       std::string &_error)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    gz::msgs::WorldControl request;
    request.set_pause(true);
    if (_steps > 0)
      request.set_multi_step(_steps);

    gz::msgs::Boolean response;
    bool result = false;
    const bool executed = this->node->Request(
        this->service, request, _timeoutMs, response, result);
    if (!executed)
    {
      _error = "Gazebo Transport WorldControl request timed out";
      return false;
    }
    if (!result)
    {
      _error = "Gazebo WorldControl service reported failure";
      return false;
    }
    if (!response.data())
    {
      _error = "Gazebo WorldControl response data=false";
      return false;
    }
    _error.clear();
    return true;
  }

  private: std::string service;
  private: std::unique_ptr<gz::transport::Node> node;
  private: std::mutex mutex;
};

void CopyError(const std::string &_message, char *_buffer, size_t _capacity)
{
  if (_buffer == nullptr || _capacity == 0)
    return;
  std::snprintf(_buffer, _capacity, "%s", _message.c_str());
}

}  // namespace

extern "C"
{

void *field_gz_control_create(const char *_partition, const char *_service,
                              char *_error, size_t _errorCapacity)
{
  try
  {
    if (_partition == nullptr || _service == nullptr)
      throw std::invalid_argument("partition and service are required");
    auto client = std::make_unique<ControlClient>(_partition, _service);
    CopyError("", _error, _errorCapacity);
    return client.release();
  }
  catch (const std::exception &_exception)
  {
    CopyError(_exception.what(), _error, _errorCapacity);
    return nullptr;
  }
  catch (...)
  {
    CopyError("unknown exception creating Gazebo control client",
              _error, _errorCapacity);
    return nullptr;
  }
}

int field_gz_control_request(void *_handle, uint32_t _steps,
                             uint32_t _timeoutMs, char *_error,
                             size_t _errorCapacity)
{
  try
  {
    if (_handle == nullptr)
      throw std::invalid_argument("control client handle is null");
    std::string error;
    const bool ok = static_cast<ControlClient *>(_handle)->Request(
        _steps, _timeoutMs, error);
    CopyError(error, _error, _errorCapacity);
    return ok ? 1 : 0;
  }
  catch (const std::exception &_exception)
  {
    CopyError(_exception.what(), _error, _errorCapacity);
    return 0;
  }
  catch (...)
  {
    CopyError("unknown exception requesting Gazebo control",
              _error, _errorCapacity);
    return 0;
  }
}

void field_gz_control_destroy(void *_handle)
{
  delete static_cast<ControlClient *>(_handle);
}

}  // extern "C"
