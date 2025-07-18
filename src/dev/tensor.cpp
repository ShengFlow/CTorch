module tensor
#include <iostream>

Storage::Storage(std::initializer_list<double> items):
	size(items.size())
{
	std::copy(items.begin(),items.end(),data.get())
}

double& Storage::operator[](size_t idx) const {
	if(idx >= size) throw std::out_of_range("Storage data visit out of range")
	return data[idx];
}

double* Storage::data() const {
	return data.get();
}

const size_t Storage::size() const {
	return size;
}

void Storage::output() const {
	std::cout<<"Storage[";
	for(size_t i{0};i<size;i++){
		std::cout<<data[i];
		if(i != size-1) std::cout<<", ";
	}
	std::cout<<"]";
}

static std::vector<int> Tensor::calc_strides(std::vector<int> _shape){ // Calculate from the biggest dimension
	std::vector<int> _strides(_shape.size(),1);
	for(int dim{_shape.size()-1};dim>=0;dim--)
		_strides[dim] = _shape[dim+1]*_strides[dim+1];
	return _strides;
}

static int Tensor::calc_size(std::vector<int> _shape){
	int _size = 1;
	for(int dim:_shape) _size *= dim;
	return _size;
}

Tensor::Tensor(const std::initializer_list<double> items,const std::vector<int> _shape):
	storage(Storage(items)),
	shape(_shape),
	strides(calc_strides(_shape))
{
	offset = storage.data(); // Adopt the first element's address as the data start,same next
}

Tensor::Tensor(const Storage _storage,const std::vector<int> _shape):
	storage(make_shared<Storage>(_storage)),
	shape(_shape),
	strides(calc_strides(_shape)),
	offset(_storage->data()){}

Tensor::Tensor(const std::vector<int> _shape):
	storage(Storage(calc_size(_shape))),
	shape(_shape),
	strides(calc_strides(_shape))
{
	offset = storage.data();
}

Tensor Tensor::view(std::vector<int> _shape){
	return Tensor(*storage,_shape);
}

Tensor Tensor::clone(){ // Deep copy
	std::shared_ptr<Storage> _storage = std::make_shared(storage->size());
	for(int i{0};i<storage->size();i++) _storage->data()[i] = storage->data()[i];
	return Tensor(_storage,shape);
}

const std::shared_ptr<Storage> Tensor::storage(){
	return storage;
}
