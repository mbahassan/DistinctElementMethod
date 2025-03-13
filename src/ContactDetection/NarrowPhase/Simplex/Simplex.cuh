//
// Created by iqraa on 11-3-25.
//

#ifndef SIMPLEX_CUH
#define SIMPLEX_CUH



class Simplex {
private:
    std::array<float3, 4> points;
    int size;

public:
    Simplex() : size(0) {}

    void push_front(const float3& point) {
        for (int i = size; i > 0; i--) {
            points[i] = points[i-1];
        }
        points[0] = point;
        size = std::min(size + 1, 4);
    }

    void remove_point(int index) {
        for (int i = index; i < size - 1; i++) {
            points[i] = points[i+1];
        }
        size--;
    }

    const float3& operator[](int index) const {
        return points[index];
    }

    float3& operator[](int index) {
        return points[index];
    }

    int get_size() const {
        return size;
    }

    void set_size(int newSize) {
        size = newSize;
    }

    void clear() {
        size = 0;
    }
};


#endif //SIMPLEX_CUH
