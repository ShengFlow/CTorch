#include "Distributed/BackendManager.h"
#include "DeviceAllocator.h"

namespace ct {
namespace distributed {

void BackendManager::registerBackend(std::shared_ptr<DeviceBackend> backend) {
    if (!backend) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
            "BackendManager: cannot register null backend");
    }
    DeviceType device = backend->deviceType();
    std::lock_guard<std::mutex> lock(_mtx);
    if (_backends.find(device) != _backends.end()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
            "BackendManager: backend already registered for device type");
    }
    _backends[device] = std::move(backend);
}

void BackendManager::unregisterBackend(DeviceType device) {
    std::lock_guard<std::mutex> lock(_mtx);
    _backends.erase(device);
}

std::shared_ptr<DeviceBackend> BackendManager::getBackend(DeviceType device) {
    std::lock_guard<std::mutex> lock(_mtx);
    auto it = _backends.find(device);
    if (it != _backends.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<DeviceType> BackendManager::registeredBackends() const {
    std::lock_guard<std::mutex> lock(_mtx);
    std::vector<DeviceType> result;
    result.reserve(_backends.size());
    for (const auto& [device, _] : _backends) {
        result.push_back(device);
    }
    return result;
}

std::vector<BackendCapability> BackendManager::allCapabilities() const {
    std::lock_guard<std::mutex> lock(_mtx);
    std::vector<BackendCapability> result;
    result.reserve(_backends.size());
    for (const auto& [_, backend] : _backends) {
        result.push_back(backend->capability());
    }
    return result;
}

bool BackendManager::hasBackend(DeviceType device) const {
    std::lock_guard<std::mutex> lock(_mtx);
    return _backends.find(device) != _backends.end();
}

size_t BackendManager::backendCount() const {
    std::lock_guard<std::mutex> lock(_mtx);
    return _backends.size();
}

void BackendManager::synchronizeAll() {
    std::lock_guard<std::mutex> lock(_mtx);
    for (auto& [_, backend] : _backends) {
        backend->synchronize();
    }
}

void BackendManager::synchronize(DeviceType device) {
    auto backend = getBackend(device);
    if (backend) {
        backend->synchronize();
    }
}

} // namespace distributed
} // namespace ct