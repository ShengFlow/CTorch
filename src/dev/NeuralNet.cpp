//
// Created by Beapoe on 25-7-27.
//
module;

#include <utility>

module nn;

// Parameter
 Parameter::Parameter(const Tensor &data, const bool requiresGrad = true) {
     _data = data;
     _data.requires_grad(requiresGrad);
     _initialized = true;
 }

bool Parameter::isInitialized() const {
     return _initialized;
 }

void Parameter::setInitialized(const bool status) {_initialized = status;}

Tensor Parameter::data() const {
     return _data;
 }

void Parameter::setData(Tensor data) {
     _data = std::move(data);
 }

// Buffer
 Buffer::Buffer(const Tensor &data) {
     _data = data;
     _initialized = true;
 }

bool Buffer::isInitialized() const {
     return _initialized;
 }

void Buffer::setInitialized(const bool status) { _initialized = status; }


Tensor Buffer::data() const {
     return _data;
 }

void Buffer::setData(Tensor data) {
     _data = std::move(data);
 }