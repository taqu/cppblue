#ifndef INC_CPPBLUE_H_
#define INC_CPPBLUE_H_
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>

namespace cppblue
{
using s8 = int8_t;
using s16 = int16_t;
using s32 = int32_t;
using s64 = int64_t;

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using f32 = float;
using f64 = double;

inline static constexpr uint32_t CPPRNG_DEFAULT_SEED32 = 12345UL;
inline static constexpr uint64_t CPPRNG_DEFAULT_SEED64 = 12345ULL;

//--- PCG32
//---------------------------------------------------------
/**
 * @brief A fast 32 bit PRNG
 *
 * | Feature |      |
 * | :------ | :--- |
 * | Bits    | 32   |
 * | Period  | 2^64 |
 * | Streams | 1    |
 */
class PCG32
{
public:
    /**
     * @brief Initialize with CPPRNG_DEFAULT_SEED64
     */
    PCG32();

    /**
     * @brief Initialize with a seed
     * @param [in] seed ... initialize states with
     */
    explicit PCG32(uint64_t seed);
    ~PCG32();

    /**
     * @brief Initialize states with a seed
     * @param [in] seed
     */
    void srand(uint64_t seed);

    /**
     * @brief Generate a 32bit unsigned value
     * @return
     */
    uint32_t rand();

    /**
     * @brief Generate a 32bit real number
     * @return [0 1)
     */
    float frand();

    uint32_t range(u32 rmax);
private:
    inline static constexpr uint64_t Increment = 1442695040888963407ULL;
    inline static constexpr uint64_t Multiplier = 6364136223846793005ULL;
    uint64_t state_;
};

//--- SplitMix
//---------------------------------------------------------
/**
 * @brief A fast 64 bit PRNG
 *
 * | Feature |      |
 * | :------ | :--- |
 * | Bits    | 64   |
 * | Period  | 2^64 |
 * | Streams | 1    |
 */
class SplitMix
{
public:
    static uint64_t next(uint64_t& state);
};

//--- Vector2
//---------------------------------------------------------
struct Vector2
{
    Vector2& operator+=(const Vector2& other);
    f32 x_;
    f32 y_;
};

Vector2 operator+(const Vector2& x0, const Vector2& x1);
Vector2 operator-(const Vector2& x0, const Vector2& x1);
Vector2 operator*(f32 x0, const Vector2& x1);
Vector2 operator*(const Vector2& x0, f32 x1);
Vector2 operator/(const Vector2& x0, f32 x1);

//--- Vector2u
//---------------------------------------------------------
struct Vector2u
{
    u32 x_;
    u32 y_;
};

bool operator==(const Vector2u& x0, const Vector2u& x1);
bool operator!=(const Vector2u& x0, const Vector2u& x1);

//--- Vector3u
//---------------------------------------------------------
struct Vector3u
{
    u32 x_;
    u32 y_;
    u32 z_;
};

bool operator==(const Vector3u& x0, const Vector3u& x1);
bool operator!=(const Vector3u& x0, const Vector3u& x1);

//--- Array
//---------------------------------------------------------
template<class T>
class Array
{
    static_assert(std::is_trivially_copyable<T>::value == true, "T should be trivially copyable.");

public:
    inline static constexpr u32 Expand = 64;
    Array();
    ~Array();
    u32 capacity() const;
    u32 size() const;
    void clear();
    void reserve(u32 capacity);
    void resize(u32 size);
    void push_back(const T& x);
    void pop_back();
    void insert(u32 index, const T& x);
    void removeAt(u32 index);
    const T& operator[](u32 index) const;
    T& operator[](u32 index);

    Array& operator=(const Array&);
private:
    Array(const Array&) = delete;
    void expand(u32 capacity);

    u32 capacity_;
    u32 size_;
    T* items_;
};

template<class T>
Array<T>::Array()
    : capacity_(0)
    , size_(0)
    , items_(nullptr)
{
}

template<class T>
Array<T>::~Array()
{
    delete[] items_;
    items_ = nullptr;
}

template<class T>
u32 Array<T>::capacity() const
{
    return capacity_;
}

template<class T>
u32 Array<T>::size() const
{
    return size_;
}

template<class T>
void Array<T>::clear()
{
    size_ = 0;
}

template<class T>
void Array<T>::reserve(u32 capacity)
{
    if(capacity <= capacity_) {
        return;
    }
    u32 newCapacity = capacity_;
    while(newCapacity < capacity) {
        newCapacity += Expand;
    }
    expand(newCapacity);
}

template<class T>
void Array<T>::resize(u32 size)
{
    if(capacity_ < size) {
        u32 newCapacity = capacity_;
        while(newCapacity < size) {
            newCapacity += Expand;
        }
        expand(newCapacity);
    }
    size_ = size;
}

template<class T>
void Array<T>::push_back(const T& x)
{
    if(capacity_ <= size_) {
        expand(capacity_ + Expand);
    }
    assert(size_ < capacity_);
    items_[size_] = x;
    ++size_;
}

template<class T>
void Array<T>::pop_back()
{
    assert(0 < size_);
    --size_;
}

template<class T>
void Array<T>::insert(u32 index, const T& x)
{
    assert(index<size_);
    if(capacity_<=size_){
        expand(capacity_+Expand);
    }
    for(u32 i=size_; index<i; --i){
        items_[i] = items_[i-1];
    }
    items_[index] = x;
    ++size_;
}

template<class T>
void Array<T>::removeAt(u32 index)
{
    assert(index<size_);
    for(u32 i=index+1; i<size_; ++i){
        items_[i-1] = items_[i];
    }
    --size_;
}

template<class T>
const T& Array<T>::operator[](u32 index) const
{
    assert(index < size_);
    return items_[index];
}

template<class T>
T& Array<T>::operator[](u32 index)
{
    assert(index < size_);
    return items_[index];
}

template<class T>
Array<T>& Array<T>::operator=(const Array& other)
{
    if(this == &other){
        return *this;
    }
    if(capacity_ < other.size_) {
        delete[] items_;
        capacity_ = other.capacity_;
        items_ = new T[capacity_];
    }
    size_ = other.size_;
    ::memcpy(items_, other.items_, size_*sizeof(T));
    return *this;
}

template<class T>
void Array<T>::expand(u32 capacity)
{
    T* items = new T[capacity];
    ::memcpy(items, items_, capacity_ * sizeof(T));
    delete[] items_;
    capacity_ = capacity;
    items_ = items;
}

//--- Array2d
//---------------------------------------------------------
template<class T>
class Array2d
{
    static_assert(std::is_trivially_copyable<T>::value == true, "T should be trivially copyable.");

public:
    Array2d();
    ~Array2d();
    u32 width() const;
    u32 height() const;
    void reset(u32 width, u32 height);
    void memset(s32 x);
    const T& operator()(u32 x, u32 y) const;
    T& operator()(u32 x, u32 y);

    Array2d(const Array2d& other);
    Array2d& operator=(const Array2d& other);
private:
    u32 width_;
    u32 height_;
    T* items_;
};

template<class T>
Array2d<T>::Array2d()
    : width_(0)
    , height_(0)
    , items_(nullptr)
{
}

template<class T>
Array2d<T>::~Array2d()
{
    delete[] items_;
    items_ = nullptr;
}

template<class T>
Array2d<T>::Array2d(const Array2d& other)
    :width_(other.width_)
    ,height_(other.height_)
    ,items_(nullptr)
{
    items_ = new T[width_*height_];
    ::memcpy(items_, other.items_, sizeof(T)*width_*height_);
}

template<class T>
Array2d<T>& Array2d<T>::operator=(const Array2d& other)
{
    if(this == &other){
        return *this;
    }
    assert(width_ == other.width_);
    assert(height_ == other.height_);
    ::memcpy(items_, other.items_, sizeof(T)*width_*height_);
    return *this;
}

template<class T>
u32 Array2d<T>::width() const
{
    return width_;
}

template<class T>
u32 Array2d<T>::height() const
{
    return height_;
}

template<class T>
void Array2d<T>::reset(u32 width, u32 height)
{
    if(width == width_ && height == height_){
        return;
    }
    width_ = width;
    height_ = height;
    delete[] items_;
    items_ = new T[width_*height_];
}

template<class T>
void Array2d<T>::memset(s32 x)
{
    ::memset(items_, x, sizeof(T)*width_*height_);
}

template<class T>
const T& Array2d<T>::operator()(u32 x, u32 y) const
{
    assert(x<width_);
    assert(y<height_);
    return items_[y*width_+x];
}

template<class T>
T& Array2d<T>::operator()(u32 x, u32 y)
{
    assert(x<width_);
    assert(y<height_);
    return items_[y*width_+x];
}

//--- Array3d
//---------------------------------------------------------
template<class T>
class Array3d
{
    static_assert(std::is_trivially_copyable<T>::value == true, "T should be trivially copyable.");

public:
    Array3d();
    ~Array3d();
    u32 width() const;
    u32 height() const;
    u32 depth() const;
    void reset(u32 width, u32 height, u32 depth);
    void memset(s32 x);
    const T& operator()(u32 x, u32 y, u32 z) const;
    T& operator()(u32 x, u32 y, u32 z);

    Array3d(const Array3d& other);
    Array3d& operator=(const Array3d& other);
private:
    u32 width_;
    u32 height_;
    u32 depth_;
    T* items_;
};

template<class T>
Array3d<T>::Array3d()
    : width_(0)
    , height_(0)
    , depth_(0)
    , items_(nullptr)
{
}

template<class T>
Array3d<T>::~Array3d()
{
    delete[] items_;
    items_ = nullptr;
}

template<class T>
Array3d<T>::Array3d(const Array3d& other)
    :width_(other.width_)
    ,height_(other.height_)
    ,depth_(other.depth_)
    ,items_(nullptr)
{
    items_ = new T[width_*height_*depth_];
    ::memcpy(items_, other.items_, sizeof(T)*width_*height_*depth_);
}

template<class T>
Array3d<T>& Array3d<T>::operator=(const Array3d& other)
{
    if(this == &other){
        return *this;
    }
    assert(width_ == other.width_);
    assert(height_ == other.height_);
    assert(depth_ == other.depth_);
    ::memcpy(items_, other.items_, sizeof(T)*width_*height_*depth_);
    return *this;
}

template<class T>
u32 Array3d<T>::width() const
{
    return width_;
}

template<class T>
u32 Array3d<T>::height() const
{
    return height_;
}

template<class T>
u32 Array3d<T>::depth() const
{
    return depth_;
}

template<class T>
void Array3d<T>::reset(u32 width, u32 height, u32 depth)
{
    if(width == width_ && height == height_ && depth_ == depth){
        return;
    }
    width_ = width;
    height_ = height;
    depth_ = depth;
    delete[] items_;
    items_ = new T[width_*height_*depth_];
}

template<class T>
void Array3d<T>::memset(s32 x)
{
    ::memset(items_, x, sizeof(T)*width_*height_*depth_);
}

template<class T>
const T& Array3d<T>::operator()(u32 x, u32 y, u32 z) const
{
    assert(x<width_);
    assert(y<height_);
    assert(z<depth_);
    return items_[(z*height_+y)*width_+x];
}

template<class T>
T& Array3d<T>::operator()(u32 x, u32 y, u32 z)
{
    assert(x<width_);
    assert(y<height_);
    assert(z<depth_);
    return items_[(z*height_+y)*width_+x];
}

//---Bitmap2d 
//---------------------------------------------------------
class Bitmap2d
{
public:
    Bitmap2d();
    ~Bitmap2d();
    u32 width() const;
    u32 height() const;
    void reset(u32 width, u32 height);
    void memset(s32 x);

    void set(u32 x, u32 y, u8 v);
    u8 operator()(u32 x, u32 y) const;

    Bitmap2d(const Bitmap2d& other);
    Bitmap2d& operator=(const Bitmap2d& other);
private:
    u32 width_;
    u32 height_;
    u32* items_;
};

//---Bitmap3d 
//---------------------------------------------------------
class Bitmap3d
{
public:
    Bitmap3d();
    ~Bitmap3d();
    u32 width() const;
    u32 height() const;
    u32 depth() const;
    void reset(u32 width, u32 height, u32 depth);
    void memset(s32 x);

    void set(u32 x, u32 y, u32 z, u8 v);
    u8 operator()(u32 x, u32 y, u32 z) const;

    Bitmap3d(const Bitmap3d& other);
    Bitmap3d& operator=(const Bitmap3d& other);
private:
    u32 size_;
    u32 width_;
    u32 height_;
    u32 depth_;
    u32* items_;
};

//--- Image
//---------------------------------------------------------
class Image
{
public:
    static void reset(Image& image, u32 width, u32 height, u32 channels);

    Image();
    ~Image();
    u32 getWidth() const;
    u32 getHeight() const;
    u32 getChannels() const;
    const u8& operator()(u32 x, u32 y, u32 c) const;
    u8& operator()(u32 x, u32 y, u32 c);

private:
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    u32 width_ = 0;
    u32 height_ = 0;
    u32 channels_ = 0;
    std::unique_ptr<u8[]> pixels_;
};

//--- IndexMap
//---------------------------------------------------------
class IndexMap
{
public:
    static void reset(IndexMap& map, u32 width, u32 height);

    IndexMap();
    ~IndexMap();
    u32 getWidth() const;
    u32 getHeight() const;
    const u32& operator()(u32 x, u32 y) const;
    u32& operator()(u32 x, u32 y);

private:
    IndexMap(const IndexMap&) = delete;
    IndexMap& operator=(const IndexMap&) = delete;
    u32 width_ = 0;
    u32 height_ = 0;
    u32* pixels_ = nullptr;
};

//--- VoronoiCell
//---------------------------------------------------------
struct VoronoiCell
{
    Vector2 centroid_;
    f32 orientation_;
    f32 area_;
    f32 sumDensity_;
};

//--- Moments
//---------------------------------------------------------
struct Moments
{
    f32 m00_;
    f32 m10_;
    f32 m01_;
    f32 m11_;
    f32 m20_;
    f32 m02_;
};

//--- LindeBuzoGray
//---------------------------------------------------------
/**
 * [Weighted Linde-Buzo-Gray Stippling](https://graphics.uni-konstanz.de/publikationen/Deussen2017LindeBuzoGray/index.html)
 */
class LindeBuzoGray
{
public:
    static f32 round01(f32 x);
    u32 toPixel(f32 x);

    LindeBuzoGray();
    ~LindeBuzoGray();
    void build(u32 targetSize, u32 imageSize, u32 maxIterations=50, f32 hysteresis = 0.6f, f32 hysteresisDelta = 0.01f, Image* density = nullptr);

    u32 size() const;
    const Vector2& operator[](u32 index) const;
private:
    LindeBuzoGray(const LindeBuzoGray&) = delete;
    LindeBuzoGray& operator=(const LindeBuzoGray&) = delete;

    /**
     * @brief Jump flooding algorithm (https://www.comp.nus.edu.sg/~tants/jfa.html)
     * @param indexMap 
     * @param points 
     */
    void jump_flood(IndexMap& indexMap, const Array<Vector2>& points);
    static void accumulateCells(Array<VoronoiCell>& cells, const Array<Vector2>& points, const IndexMap& indexMap, const Image& density);
    Vector2 jitter(const Vector2& x);

    u32 targetSize_;
    u32 imageSize_;
    Array<Vector2> stipples_;
    IndexMap indexMap_;
    Array<VoronoiCell> cells_;
    std::unique_ptr<Image> density_;
    PCG32 random_;
};

//--- VoidAndClaster1
//---------------------------------------------------------
class VoidAndClaster1
{
public:
    struct Point
    {
        u32 rank_;
        u32 position_;
    };

    struct NPoint
    {
        f32 rank_;
        u32 position_;
    };

    VoidAndClaster1();
    ~VoidAndClaster1();

    void build(u32 size, f32 standardDeviation=1.5f, u32 initialDensity=10);

    u32 size() const;
    NPoint operator[](u32 x) const;
private:
    using Bitmap1d = Array<u8>;
    VoidAndClaster1(const VoidAndClaster1&) = delete;
    VoidAndClaster1& operator=(const VoidAndClaster1&) = delete;
    void updateEnergy(u32 x, s32 sign);
    u32 getMin(const Bitmap1d& bitmap);
    u32 getMax(const Bitmap1d& bitmap);
    u32 removeTightest(Bitmap1d& bitmap);
    void addPoint(Bitmap1d& bitmap, u32 x);

    u32 ranks_;
    Bitmap1d bitmap_;
    Array<f32> elut_;
    Array<f32> energy_;
    Array<Point> points_;
    PCG32 random_;
};

//--- VoidAndClaster2
//---------------------------------------------------------
class VoidAndClaster2
{
public:
    struct Point
    {
        u32 rank_;
        Vector2u position_;
    };

    struct NPoint
    {
        f32 rank_;
        Vector2u position_;
    };

    VoidAndClaster2();
    ~VoidAndClaster2();

    void build(u32 size, f32 standardDeviation=1.5f, u32 initialDensity=10);

    u32 size() const;
    NPoint operator[](u32 x) const;
private:
    VoidAndClaster2(const VoidAndClaster2&) = delete;
    VoidAndClaster2& operator=(const VoidAndClaster2&) = delete;
    void updateEnergy(u32 x, u32 y, s32 sign);
    Vector2u getMin(const Bitmap2d& bitmap);
    Vector2u getMax(const Bitmap2d& bitmap);
    Vector2u removeTightest(Bitmap2d& bitmap);
    void addPoint(Bitmap2d& bitmap, const Vector2u& x);

    u32 ranks_;
    Bitmap2d bitmap_;
    Array2d<f32> elut_;
    Array2d<f32> energy_;
    Array<Point> points_;
    PCG32 random_;
};

//--- VoidAndClaster3
//---------------------------------------------------------
class VoidAndClaster3
{
public:
    struct Point
    {
        u32 rank_;
        Vector3u position_;
    };

    struct NPoint
    {
        f32 rank_;
        Vector3u position_;
    };

    VoidAndClaster3();
    ~VoidAndClaster3();

    void build(u32 size, f32 standardDeviation=1.5f, u32 initialDensity=10);

    u32 size() const;
    NPoint operator[](u32 x) const;
private:
    VoidAndClaster3(const VoidAndClaster3&) = delete;
    VoidAndClaster3& operator=(const VoidAndClaster3&) = delete;
    void updateEnergy(u32 x, u32 y, u32 z, s32 sign);
    Vector3u getMin(const Bitmap3d& bitmap);
    Vector3u getMax(const Bitmap3d& bitmap);
    Vector3u removeTightest(Bitmap3d& bitmap);
    void addPoint(Bitmap3d& bitmap, const Vector3u& x);

    u32 ranks_;
    Bitmap3d bitmap_;
    Array3d<f32> elut_;
    Array3d<f32> energy_;
    Array<Point> points_;
    PCG32 random_;
};

} // namespace cpplbg
#endif // INC_CPPBLUE_H_

