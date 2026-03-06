
#include <tensor.h>
#include <fstream>
#include "util.h"

#include <cstring>
#include <Eigen/Core>

namespace sharpa {
namespace tactile {

/** range */

Range::Range(size_t l, size_t r) : l_(l), r_(r) {
    assert_throw(l < r, "bad range");
}

size_t Range::l() const { return l_; }
size_t Range::r() const { return r_; }

bool Range::inside(size_t x) const {
    return l_ <= x && x < r_;  /* [l, r) */ 
}

bool Range::operator==(const Range &other) const {
    return l_ == other.l_ && r_ == other.r_;
}
bool Range::operator!=(const Range &other) const {
    return !operator==(other);
}

/** shape */

Shape::Shape(std::vector<size_t> data) : data_(std::move(data)) {
    // assert_throw(dim() > 0, "bad shape");
    for(auto var : data_) assert_throw(var > 0, "bad shape");
}

Shape::Shape() : Shape(std::vector<size_t>{}) {}

size_t Shape::size() const {
    if(dim() == 0) return 0;
    size_t ret{1};
    for(auto var : data_) ret *= var;
    return ret;
}

std::vector<size_t>::const_iterator Shape::begin() const {
    return data_.begin();
}

std::vector<size_t>::const_iterator Shape::end() const {
    return data_.end();
}

std::vector<size_t> Shape::data() const {
    return data_;
}

size_t Shape::dim() const { return data_.size(); }

size_t Shape::flat_idx(const Index &idx) const {
    assert_throw(inside(idx), "idx out of range");
    size_t ret{0};
    size_t base{1};
    for(int i = dim() - 1; i >= 0; --i) {
        ret += idx[i] * base;
        base *= data_[i];
    }
    return ret;
}

bool Shape::inside(const Index &idx) const {
    assert_throw(idx.size() == dim(), "dim not equal");
    for(size_t i = 0; i < dim(); ++i) {
        if(idx[i] >= data_[i]) return false;
    }
    return true;
}

size_t Shape::operator[](size_t i) const {
    return data_[i];
}

bool Shape::operator==(const Shape &other) const {
    assert_throw(other.dim() == dim(), "dim not equal");
    return std::equal(data_.begin(), data_.end(), other.begin());
}

bool Shape::operator!=(const Shape &other) const {
    return !operator==(other);
}

std::vector<Index> Shape::all_indices() const {
    Index idx(dim(), 0);
    std::vector<Index> ret;
    for(size_t _ = 0; _ < size(); ++_) {
        ret.push_back(idx);
        /* step idx */
        for(int i = dim() - 1; i >= 0; --i) {
            if(idx[i] < data_[i] - 1) {
                idx[i] += 1;
                break;
            } else idx[i] = 0;
        }
    }
    return ret;
}

/** slice */

Slice::Slice(std::vector<Range> slice) : slice_(std::move(slice)) {}

Slice::Slice() : Slice(std::vector<Range>{}) {}

size_t Slice::dim() const { return slice_.size(); }

bool Slice::inside(const Index &idx) const {
    assert_throw(idx.size() == dim(), "dim not equal");
    for(size_t i = 0; i < dim(); ++i) {
        if(!slice_[i].inside(idx[i])) return false;
    }
    return true;
}

Shape Slice::shape() const {
    std::vector<size_t> shape;
    for(const auto &s : slice_) shape.push_back(s.r() - s.l());
    return {shape};
}

Index Slice::idx_abs(const Index &idx_rel) const {
    Index idx_abs;
    for(size_t i = 0; i < dim(); ++i) {
        size_t abs = slice_[i].l() + idx_rel[i];
        idx_abs.push_back(abs);
    }
    assert_throw(inside(idx_abs), "idx out of range");
    return idx_abs;
}

Range Slice::operator[](const size_t i) const {
    return slice_[i];
}

/** DataBlock */

DataBlock::DataBlock(Shape shape, size_t unit_size)
  : shape_(std::move(shape)), unit_size_(unit_size) {
    assert_throw(unit_size == 1 || unit_size == 2
      || unit_size == 4 || unit_size == 8, "bad unit size");
    data_ = malloc(nbytes());
}

DataBlock::DataBlock(const DataBlock &other)
  : DataBlock(other.shape(), other.unit_size_) {
    if(data_) memcpy(data_, other.data(), other.nbytes());
}

DataBlock::DataBlock(DataBlock &&other)
  : DataBlock(other.shape(), other.unit_size_) {
    data_ = other.data_;
    other.data_ = nullptr;
    other.shape_ = {{}};
}

DataBlock &DataBlock::operator=(const DataBlock &other) {
    shape_ = other.shape_;
    unit_size_ = other.unit_size_;
    data_ = malloc(nbytes());
    if(data_) memcpy(data_, other.data(), other.nbytes());
    return *this;
}

DataBlock &DataBlock::operator=(DataBlock &&other) {
    shape_ = other.shape_;
    unit_size_ = other.unit_size_;
    data_ = other.data_;
    other.data_ = nullptr;
    other.shape_ = {{}};
    return *this;
}

DataBlock::~DataBlock() { free(data_); }

Shape DataBlock::shape() const { return shape_; }
size_t DataBlock::size() const { return shape_.size(); }
size_t DataBlock::dim() const { return shape_.dim(); }
size_t DataBlock::nbytes() const { return size() * unit_size_; }
void DataBlock::set_zero() { if(data_) memset(data_, 0, nbytes()); }

void DataBlock::reshape(Shape shape) {
    assert_throw(shape_.size() == shape.size(), "shape size not equal");
    shape_ = std::move(shape); 
}

const void *DataBlock::data() const { return data_; }
void *DataBlock::data() { return data_; }

/* addition float */
DataBlock::Ptr add_db_float(const DataBlock::Ptr x, const DataBlock::Ptr y) {
    if(x->shape() != y->shape()) throw std::runtime_error("shape not equal");

    auto tx = x->as<float>();
    auto ty = y->as<float>();

    Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> x_arr(tx.data(), tx.size());
    Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> y_arr(ty.data(), ty.size());

    auto ret = std::make_shared<DataBlock>(x->shape(), sizeof(float));
    Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> ret_arr((float *)ret->data(), ret->size());

    ret_arr = x_arr + y_arr;
    return ret;
}

/* substraction float */
DataBlock::Ptr sub_db_float(const DataBlock::Ptr x, const DataBlock::Ptr y) {
    if(x->shape() != y->shape()) throw std::runtime_error("shape not equal");

    auto tx = x->as<float>();
    auto ty = y->as<float>();

    Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> x_arr(tx.data(), tx.size());
    Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> y_arr(ty.data(), ty.size());

    auto ret = std::make_shared<DataBlock>(x->shape(), sizeof(float));
    Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> ret_arr((float *)ret->data(), ret->size());

    ret_arr = x_arr - y_arr;
    return ret;
}

/* multiplication with constant */
DataBlock::Ptr mul_db_float(const DataBlock::Ptr x, float y) {
    auto tx = x->as<float>();

    Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> x_arr(tx.data(), tx.size());

    auto ret = std::make_shared<DataBlock>(x->shape(), sizeof(float));
    Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> ret_arr((float *)ret->data(), ret->size());

    ret_arr = x_arr * y;
    return ret;
}

DataBlock::Ptr db_f32_to_ui8(const DataBlock::Ptr x) {
    auto tx = x->as<float>();

    Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> x_arr(tx.data(), tx.size());

    auto ret = std::make_shared<DataBlock>(x->shape(), sizeof(uint8_t));
    Eigen::Map<Eigen::Array<uint8_t, Eigen::Dynamic, 1>> ret_arr((uint8_t *)ret->data(), ret->size());

    /* perform the cast from float to uint8_t with clamping (values outside [0,255] will be clamped) */
    ret_arr = x_arr.min(255.0f).max(0.0f).round().cast<uint8_t>();
    return ret;
}

DataBlock::Ptr db_ui8_to_f32(const DataBlock::Ptr x) {
    auto tx = x->as<uint8_t>();

    Eigen::Map<Eigen::Array<uint8_t, Eigen::Dynamic, 1>> x_arr(tx.data(), tx.size());

    auto ret = std::make_shared<DataBlock>(x->shape(), sizeof(float));
    Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> ret_arr((float *)ret->data(), ret->size());

    ret_arr = x_arr.cast<float>();
    return ret;
}

DataBlock::Ptr f32_from_txt(const std::string &txt_file, size_t size) {
    std::ifstream file(txt_file);
    if(!file.is_open()) return nullptr;
    //assert_throw(file.is_open(), "bad file");
    assert_throw(size > 0, "size of DataBlock should be large than zero");
    
    auto ret = std::make_shared<DataBlock>(Shape{{size}}, sizeof(float));
    auto t = ret->as<float>();

    size_t idx_num{0};
    while(idx_num < size && file >> t.at({idx_num})) ++idx_num;

    file.close();
    return ret;
}

}
}
