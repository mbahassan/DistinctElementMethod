//
// Created by mbahassan on 3/26/25.
//

#ifndef VECTOR_H
#define VECTOR_H

// CudaVector.hpp
template<typename T>
class CudaVector {
public:
    __host__ __device__ explicit CudaVector(size_t initial_size) : data(nullptr), size_(0), capacity_(0) {
        reserve(initial_size);
        // Initialize all elements with default constructor
        for (size_t i = 0; i < initial_size; ++i) {
            push_back(T());
        }
    }

    // Initializer list constructor
    __host__ __device__ CudaVector(std::initializer_list<T> init) : data(nullptr), size_(0), capacity_(0) {
        reserve(init.size());
        for (const auto& item : init) {
            push_back(item);
        }
    }

    // Initializer list assignment operator
    __host__ __device__ CudaVector& operator=(std::initializer_list<T> init) {
        // Clear existing data
        clear();

        // Reserve and copy new data
        reserve(init.size());
        for (const auto& item : init) {
            push_back(item);
        }

        return *this;
    }

    // Copy constructor
    __host__ __device__ CudaVector(const CudaVector& other) : data(nullptr), size_(0), capacity_(0) {
        // Reserve space for the new vector
        reserve(other.size_);

        // Deep copy elements
        for (size_t i = 0; i < other.size_; ++i) {
            push_back(other.data[i]);
        }
    }

    // Copy assignment operator
    __host__ __device__ CudaVector& operator=(const CudaVector& other) {
        // Check for self-assignment
        if (this != &other) {
            // Clear existing data
            clear();

            // Reserve space and copy elements
            reserve(other.size_);
            for (size_t i = 0; i < other.size_; ++i) {
                push_back(other.data[i]);
            }
        }
        return *this;
    }

    // Implement move constructor
    __host__ __device__ CudaVector(CudaVector&& other) noexcept :
        data(other.data),
        size_(other.size()),
        capacity_(other.size()) {
        other.data = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    // Implement move assignment
    __host__ __device__ CudaVector& operator=(CudaVector&& other) noexcept {
        if (this != &other) {
            if (data) {
                ::operator delete(data);
            }
            data = other.data;
            size_ = other.size();
            capacity_ = other.size();
            other.data = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    // Modified emplace_back method for host and device compatibility
    template<typename... Args>
    __host__ __device__ void emplace_back(Args&&... args) {
        // Check if we need to resize
#ifdef __CUDA_ARCH__
        // Simple device-side growth strategy
        if (size_ >= capacity_) {
            size_t new_capacity = capacity_ == 0 ? 1 : capacity_ * 2;

            // Allocate new memory
            T* new_data = static_cast<T*>(malloc(new_capacity * sizeof(T)));

            // Copy existing elements
            for (size_t i = 0; i < size_; ++i) {
                new (new_data + i) T(std::move(data[i]));
            }

            // Free old memory if exists
            if (data) {
                free(data);
            }

            data = new_data;
            capacity_ = new_capacity;
        }
#else
        // Host-side growth strategy (existing implementation)
        if (size_ >= capacity_) {
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        }
#endif

        // Construct element in-place at the end of the vector
#ifdef __CUDA_ARCH__
        // Use placement new for device-side construction
        new (data + size_) T(std::forward<Args>(args)...);
#else
        // Existing host-side construction
        new (data + size_) T(std::forward<Args>(args)...);
#endif

        // Increment size
        ++size_;
    }


// Forward declaration of iterator classes
    class iterator;
    class const_iterator;

    // Iterator class with improved comparison
    class iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        __host__ __device__ iterator() : ptr_(nullptr) {}
        __host__ __device__ explicit iterator(T* ptr) : ptr_(ptr) {}

        __host__ __device__ reference operator*() { return *ptr_; }
        __host__ __device__ reference operator*() const { return *ptr_; }

        __host__ __device__ iterator& operator++() {
            ++ptr_;
            return *this;
        }

        __host__ __device__ iterator operator++(int) {
            iterator tmp = *this;
            ++ptr_;
            return tmp;
        }

        // Comparison operators
        __host__ __device__ bool operator==(const iterator& other) const {
            return ptr_ == other.ptr_;
        }

        __host__ __device__ bool operator!=(const iterator& other) const {
            return ptr_ != other.ptr_;
        }

        // Additional iterator operations
        __host__ __device__ difference_type operator-(const iterator& other) const {
            return ptr_ - other.ptr_;
        }

        __host__ __device__ iterator operator+(difference_type n) const {
            return iterator(ptr_ + n);
        }

        __host__ __device__ bool operator<(const iterator& other) const {
            return ptr_ < other.ptr_;
        }

    private:
        T* ptr_;

        // Allow CudaVector to access ptr_
        friend class CudaVector;
        friend class const_iterator;
    };

    // Const iterator with similar implementation
    class const_iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        __host__ __device__ const_iterator() : ptr_(nullptr) {}
        __host__ __device__ explicit const_iterator(const T* ptr) : ptr_(ptr) {}

        // Allow construction from non-const iterator
        __host__ __device__ explicit const_iterator(const iterator& other) : ptr_(other.ptr_) {}

        __host__ __device__ reference operator*() { return *ptr_; }
        __host__ __device__ reference operator*() const { return *ptr_; }

        __host__ __device__ const_iterator& operator++() {
            ++ptr_;
            return *this;
        }

        __host__ __device__ const_iterator operator++(int) {
            const_iterator tmp = *this;
            ++ptr_;
            return tmp;
        }

        // Comparison operators
        __host__ __device__ bool operator==(const const_iterator& other) const {
            return ptr_ == other.ptr_;
        }

        __host__ __device__ bool operator!=(const const_iterator& other) const {
            return ptr_ != other.ptr_;
        }

        // Additional iterator operations
        __host__ __device__ difference_type operator-(const const_iterator& other) const {
            return ptr_ - other.ptr_;
        }

        __host__ __device__ const_iterator operator+(difference_type n) const {
            return const_iterator(ptr_ + n);
        }

        __host__ __device__ bool operator<(const const_iterator& other) const {
            return ptr_ < other.ptr_;
        }

    private:
        const T* ptr_;

        // Allow CudaVector to access ptr_
        friend class CudaVector;
    };

    __host__ __device__ CudaVector() : data(nullptr), size_(0), capacity_(0) {}

    __host__ __device__ ~CudaVector() {
#ifndef __CUDA_ARCH__
        if (data) {
            ::operator delete(data);
        }
#endif
    }

    __host__ void resize(size_t new_size) {
    // If new size is larger than current capacity, we need to reallocate
    if (new_size > capacity_) {
        // Calculate new capacity (similar to std::vector growth strategy)
        size_t new_capacity = capacity_ == 0 ? new_size :
            std::max(new_size, capacity_ * 2);

        // Allocate new memory
        T* new_data = static_cast<T*>(::operator new(new_capacity * sizeof(T)));

        // Copy existing elements
        for (size_t i = 0; i < size_; ++i) {
            // Use placement new for in-place construction
            new (new_data + i) T(std::move(data[i]));

            // Manually call destructor on old elements
            #ifndef __CUDA_ARCH__
            data[i].~T();
            #endif
        }

        // Initialize new elements with default constructor
        for (size_t i = size_; i < new_size; ++i) {
            new (new_data + i) T();
        }

        // Free old memory
        if (data) {
            #ifndef __CUDA_ARCH__
            ::operator delete(data);
            #endif
        }

        // Update pointer and capacity
        data = new_data;
        capacity_ = new_capacity;
    }
    else if (new_size > size_) {
        // If new size is within current capacity, just default construct new elements
        for (size_t i = size_; i < new_size; ++i) {
            new (data + i) T();
        }
    }
    else if (new_size < size_) {
        // If reducing size, call destructors for elements being removed
        #ifndef __CUDA_ARCH__
        for (size_t i = new_size; i < size_; ++i) {
            data[i].~T();
        }
        #endif
    }

    // Update size
    size_ = new_size;
}

    // Overload for resize with a specific value
    __host__ __device__ void resize(size_t new_size, const T& value) {
    // If new size is larger than current capacity, we need to reallocate
    if (new_size > capacity_) {
        // Calculate new capacity (similar to std::vector growth strategy)
        size_t new_capacity = capacity_ == 0 ? new_size :
            std::max(new_size, capacity_ * 2);

        // Allocate new memory
        T* new_data = static_cast<T*>(::operator new(new_capacity * sizeof(T)));

        // Copy existing elements
        for (size_t i = 0; i < size_; ++i) {
            // Use placement new for in-place construction
            new (new_data + i) T(std::move(data[i]));

            // Manually call destructor on old elements
            #ifndef __CUDA_ARCH__
            data[i].~T();
            #endif
        }

        // Initialize new elements with provided value
        for (size_t i = size_; i < new_size; ++i) {
            new (new_data + i) T(value);
        }

        // Free old memory
        if (data) {
            #ifndef __CUDA_ARCH__
            ::operator delete(data);
            #endif
        }

        // Update pointer and capacity
        data = new_data;
        capacity_ = new_capacity;
    }
    else if (new_size > size_) {
        // If new size is within current capacity, construct new elements with value
        for (size_t i = size_; i < new_size; ++i) {
            new (data + i) T(value);
        }
    }
    else if (new_size < size_) {
        // If reducing size, call destructors for elements being removed
        #ifndef __CUDA_ARCH__
        for (size_t i = new_size; i < size_; ++i) {
            data[i].~T();
        }
        #endif
    }

    // Update size
    size_ = new_size;
}

    // Erase method to remove an element by iterator
    __host__ __device__ iterator erase(iterator pos) {
        // Ensure we're not trying to erase beyond the vector's end
        if (pos == end()) {
            return pos;
        }

        // Move elements after the erased position
        size_t index = pos - begin();
        for (size_t i = index; i < size_ - 1; ++i) {
            data[i] = std::move(data[i + 1]);
        }

        // Reduce size
        --size_;

        // Return iterator to the next element
        return iterator(data + index);
    }

    __host__ __device__ void reserve(size_t new_capacity) {
        if (new_capacity > capacity_) {
            // Allocate raw memory
            T* new_data = static_cast<T*>(::operator new(new_capacity * sizeof(T)));

            // Move existing elements using placement new
            for (size_t i = 0; i < size_; ++i) {
                // Use placement new to construct in-place
                new (new_data + i) T(std::move(data[i]));

                // Manually call destructor on old elements
#ifndef __CUDA_ARCH__
                data[i].~T();
#endif
            }

            // Delete old data if exists
            if (data) {
#ifndef __CUDA_ARCH__
                ::operator delete(data);
#endif
            }

            data = new_data;
            capacity_ = new_capacity;
        }
    }

    __host__ __device__ void push_back(const T& value) {
        if (size_ >= capacity_) {
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        }
        new (data + size_) T(std::move(value)); // construct in-place
        ++size_;
    }

    __host__ __device__ T& operator[](size_t index) {
        return data[index];
    }

    __host__ __device__ const T& operator[](size_t index) const {
        return data[index];
    }

    __host__ __device__ size_t size() const { return size_; }

    __host__ __device__ T* data_ptr() { return data; }
    __host__ __device__ const T* data_ptr() const { return data; }

    __host__ __device__ void clear() {
    #ifndef __CUDA_ARCH__
        if (data) {
            ::operator delete(data);
        }
    #endif
        size_ = 0;
        capacity_ = 0;
    }

    __host__ __device__ bool empty() const {
        return size_ == 0;
    }

    // Range-based for loop support methods
    __host__ __device__ iterator begin() {
        return iterator(data);
    }

    __host__ __device__ iterator end() {
        return iterator(data + size_);
    }

    __host__ __device__ const_iterator begin() const {
        return const_iterator(data);
    }

    __host__ __device__ const_iterator end() const {
        return const_iterator(data + size_);
    }

    __host__ __device__ const_iterator cbegin() const {
        return const_iterator(data);
    }

    __host__ __device__ const_iterator cend() const {
        return const_iterator(data + size_);
    }

    // back() method - returns reference to the last element
    __host__ __device__ T& back() {
        // Assuming the vector is not empty
        return data[size_ - 1];
    }

    // const version of back()
    __host__ __device__ const T& back() const {
        // Assuming the vector is not empty
        return data[size_ - 1];
    }

    // pop_back() method - removes the last element
    __host__ __device__ void pop_back() {
        if (size_ > 0) {
            // Call destructor for the last element (if needed)
        #ifndef __CUDA_ARCH__
            data[size_ - 1].~T();
        #endif

            // Reduce size
            --size_;
        }
    }

    // Equality operator for comparing two CudaVectors
    __host__ __device__ bool operator==(const CudaVector& other) const {
        // Check if sizes are the same
        if (size_ != other.size_) {
            return false;
        }

        // Compare each element
        for (size_t i = 0; i < size_; ++i) {
            if (!(data[i] == other.data[i])) {
                return false;
            }
        }

        return true;
    }

    // Inequality operator (for completeness)
    __host__ __device__ bool operator!=(const CudaVector& other) const {
        return !(*this == other);
    }

private:
    T* data;
    size_t size_;
    size_t capacity_;
};

#endif //VECTOR_H