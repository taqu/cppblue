#include "cppblue.h"
#include <cmath>
#include <bit>
#include <limits>
#include <random>
#include <utility>
#include <numbers>
#include <algorithm>
#include "cppimg.h"

namespace cppblue
{
namespace
{
    /**
     * @brief 32 bit right rotation
     * @param [in] x ... input
     * @param [in] r ... count of rotation
     * @return rotated
     */
    inline uint32_t rotr32(uint32_t x, uint32_t r)
    {
        return (x >> r) | (x << ((~r + 1) & 31U));
    }

    float frandom_downey_opt(PCG32& random)
    {
        constexpr int32_t lowExp = 0;
        constexpr int32_t highExp = 127;
        const uint32_t u = random.rand();
        const uint32_t b = u & 0xFFU;
        int32_t exponent = highExp - 1;
        if(0 == b) {
            exponent -= 8;
            while(true) {
                const uint32_t bits = random.rand();
                if(0 == bits) {
                    exponent -= 32;
                    if(exponent < lowExp) {
                        exponent = lowExp;
                        break;
                    }
                } else {
                    int32_t c = std::countr_zero(bits);
                    exponent -= c;
                    break;
                }
            }
        } else {
            int32_t c = std::countr_zero(b);
            exponent -= c;
        }
        const uint32_t mantissa = (u >> 8) & 0x7FFFFFUL;
        if(0 == mantissa && (u >> 31)) {
            ++exponent;
        }
        return std::bit_cast<float, uint32_t>((exponent << 23) | mantissa);
    }

    u64 getSeed()
    {
        std::random_device device;
        u64 seed = device();
        seed <<= 32;
        seed |= device();
        return seed;
    }
} // namespace

//--- PCG32
//------------------------------------------------------------
PCG32::PCG32()
    : state_{CPPRNG_DEFAULT_SEED64}
{
}

PCG32::PCG32(uint64_t seed)
{
    srand(seed);
}

PCG32::~PCG32()
{
}

void PCG32::srand(uint64_t seed)
{
    state_ = SplitMix::next(seed);
    while(0 == state_) {
        state_ = SplitMix::next(state_);
    }
}

uint32_t PCG32::rand()
{
    uint64_t x = state_;
    uint32_t c = static_cast<uint32_t>(x >> 59);
    state_ = x * Multiplier + Increment;
    x ^= x >> 18;
    return rotr32(static_cast<uint32_t>(x >> 27), c);
}

float PCG32::frand()
{
    return frandom_downey_opt(*this);
}

uint32_t PCG32::range(u32 rmax)
{
    uint32_t t = (-rmax) % rmax;
    uint64_t m;
    uint32_t l;
    do {
        uint32_t x = rand();
        m = uint64_t(x) * uint64_t(rmax);
        l = uint32_t(m);
    }while(l<t);
    return m >> 32;
}

//--- SplitMix
//--------------------------------------------
uint64_t SplitMix::next(uint64_t& state)
{
    state += 0x9E3779B97f4A7C15ULL;
    uint64_t t = state;
    t = (t ^ (t >> 30)) * 0xBF58476D1CE4E5B9ULL;
    t = (t ^ (t >> 27)) * 0x94D049BB133111EBULL;
    return t ^ (t >> 31);
}

//--- Vector2
//--------------------------------------------
Vector2& Vector2::operator+=(const Vector2& other)
{
    x_ += other.x_;
    y_ += other.y_;
    return *this;
}

Vector2 operator+(const Vector2& x0, const Vector2& x1)
{
    return {x0.x_ + x1.x_, x0.y_ + x1.y_};
}

Vector2 operator-(const Vector2& x0, const Vector2& x1)
{
    return {x0.x_ - x1.x_, x0.y_ - x1.y_};
}

Vector2 operator*(f32 x0, const Vector2& x1)
{
    return {x0*x1.x_,x0*x1.y_};
}

Vector2 operator*(const Vector2& x0, f32 x1)
{
    return {x0.x_*x1,x0.y_*x1};
}

Vector2 operator/(const Vector2& x0, f32 x1)
{
    f32 inv = 1.0f/x1;
    return {x0.x_*inv,x0.y_*inv};
}

//--- Vector2u
//---------------------------------------------------------
bool operator==(const Vector2u& x0, const Vector2u& x1)
{
    return x0.x_ == x1.x_ && x0.y_ == x1.y_;
}

bool operator!=(const Vector2u& x0, const Vector2u& x1)
{
    return x0.x_ != x1.x_ || x0.y_ != x1.y_;
}

//--- Vector3u
//---------------------------------------------------------
bool operator==(const Vector3u& x0, const Vector3u& x1)
{
    return x0.x_ == x1.x_ && x0.y_ == x1.y_ && x0.z_ == x1.z_;
}

bool operator!=(const Vector3u& x0, const Vector3u& x1)
{
    return x0.x_ != x1.x_ || x0.y_ != x1.y_ || x0.z_ != x1.z_;
}

//--- Bitmap2d
//---------------------------------------------------------
Bitmap2d::Bitmap2d()
    : width_(0)
    , height_(0)
    , items_(nullptr)
{
}

Bitmap2d::~Bitmap2d()
{
    delete[] items_;
    items_ = nullptr;
}

Bitmap2d::Bitmap2d(const Bitmap2d& other)
    :width_(other.width_)
    ,height_(other.height_)
    ,items_(nullptr)
{
    u32 size = (width_*height_ + 31)/32;
    items_ = new u32[size];
    ::memcpy(items_, other.items_, size*sizeof(u32));
}

Bitmap2d& Bitmap2d::operator=(const Bitmap2d& other)
{
    if(this == &other){
        return *this;
    }
    assert(width_ == other.width_);
    assert(height_ == other.height_);
    u32 size = (width_*height_ + 31)/32;
    ::memcpy(items_, other.items_, size*sizeof(u32));
    return *this;
}

u32 Bitmap2d::width() const
{
    return width_;
}

u32 Bitmap2d::height() const
{
    return height_;
}

void Bitmap2d::reset(u32 width, u32 height)
{
    if(width == width_ && height == height_){
        return;
    }
    width_ = width;
    height_ = height;
    delete[] items_;
    u32 size = (width_*height_ + 31)/32;
    items_ = new u32[size];
}

void Bitmap2d::memset(s32 x)
{
    u32 size = (width_*height_ + 31)/32;
    ::memset(items_, x, size*sizeof(u32));
}

void Bitmap2d::set(u32 x, u32 y, u8 v)
{
    assert(x<width_);
    assert(y<height_);
    assert(v==0 || v==1);
    u32 index = y*width_+x;
    u32 i = index>>5;
    u32 b = index&0x1FU;
    if(v){
        items_[i] |= 0x01U<<b;
    }else{
        items_[i] &= ~(0x01U<<b);
    }
}

u8 Bitmap2d::operator()(u32 x, u32 y) const
{
    assert(x<width_);
    assert(y<height_);

    u32 index = y*width_+x;
    u32 i = index>>5;
    u32 b = index&0x1FU;
    u32 v = 0x01U<<b;

    return (items_[i] & v) == v? 1 : 0;
}

//--- Bitmap3d
//---------------------------------------------------------
Bitmap3d::Bitmap3d()
    : size_(0)
    , width_(0)
    , height_(0)
    , depth_(0)
    , items_(nullptr)
{
}

Bitmap3d::~Bitmap3d()
{
    delete[] items_;
    items_ = nullptr;
}

Bitmap3d::Bitmap3d(const Bitmap3d& other)
    :size_(other.size_)
    ,width_(other.width_)
    ,height_(other.height_)
    ,depth_(other.depth_)
    ,items_(nullptr)
{
    items_ = new u32[size_];
    ::memcpy(items_, other.items_, size_*sizeof(u32));
}

Bitmap3d& Bitmap3d::operator=(const Bitmap3d& other)
{
    if(this == &other){
        return *this;
    }
    assert(width_ == other.width_);
    assert(height_ == other.height_);
    ::memcpy(items_, other.items_, size_*sizeof(u32));
    return *this;
}

u32 Bitmap3d::width() const
{
    return width_;
}

u32 Bitmap3d::height() const
{
    return height_;
}

u32 Bitmap3d::depth() const
{
    return depth_;
}

void Bitmap3d::reset(u32 width, u32 height, u32 depth)
{
    if(width == width_ && height == height_ && depth == depth_){
        return;
    }
    width_ = width;
    height_ = height;
    depth_ = depth;
    delete[] items_;
    size_ = (width_*height_*depth_ + 31)/32;
    items_ = new u32[size_];
}

void Bitmap3d::memset(s32 x)
{
    u32 size = (width_*height_ + 31)/32;
    ::memset(items_, x, size*sizeof(u32));
}

void Bitmap3d::set(u32 x, u32 y, u32 z, u8 v)
{
    assert(x<width_);
    assert(y<height_);
    assert(z<depth_);
    assert(v==0 || v==1);
    u32 index = (z*height_ + y)*width_+x;
    u32 i = index>>5;
    u32 b = index&0x1FU;
    if(v){
        items_[i] |= 0x01U<<b;
    }else{
        items_[i] &= ~(0x01U<<b);
    }
}

u8 Bitmap3d::operator()(u32 x, u32 y, u32 z) const
{
    assert(x<width_);
    assert(y<height_);
    assert(z<depth_);

    u32 index = (z*height_ + y)*width_+x;
    u32 i = index>>5;
    u32 b = index&0x1FU;
    u32 v = 0x01U<<b;

    return (items_[i] & v) == v? 1 : 0;
}

//--- Image
//---------------------------------------------------------
void Image::reset(Image& image, u32 width, u32 height, u32 channels)
{
    image.width_ = width;
    image.height_ = height;
    image.channels_ = channels;
    image.pixels_.reset(new u8[width*height*channels]);
}

Image::Image()
{
}

Image::~Image()
{
}

u32 Image::getWidth() const
{
    return width_;
}

u32 Image::getHeight() const
{
    return height_;
}

u32 Image::getChannels() const
{
    return channels_;
}

const u8& Image::operator()(u32 x, u32 y, u32 c) const
{
    u32 index = (y * width_ + x) * channels_ + c;
    return pixels_[index];
}

u8& Image::operator()(u32 x, u32 y, u32 c)
{
    u32 index = (y * width_ + x) * channels_ + c;
    return pixels_[index];
}

//--- IndexMap
//---------------------------------------------------------
void IndexMap::reset(IndexMap& map, u32 width, u32 height)
{
    if(map.width_ == width && map.height_ == height){
        return;
    }
    map.width_ = width;
    map.height_ = height;
    delete[] map.pixels_;
    map.pixels_ = new u32[width*height];
}

IndexMap::IndexMap()
{
}

IndexMap::~IndexMap()
{
    delete[] pixels_;
    pixels_ = nullptr;
}

u32 IndexMap::getWidth() const
{
    return width_;
}

u32 IndexMap::getHeight() const
{
    return height_;
}

const u32& IndexMap::operator()(u32 x, u32 y) const
{
    assert(x<width_);
    assert(y<height_);
    return pixels_[x + width_*y];
}

u32& IndexMap::operator()(u32 x, u32 y)
{
    assert(x<width_);
    assert(y<height_);
    return pixels_[x + width_*y];
}

//--- LindeBuzoGray
//---------------------------------------------------------
f32 LindeBuzoGray::round01(f32 x)
{
    if(x < 0.0f) {
        while(x < 0.0f) {
            x += 1.0f;
        }
    } else if(1.0f < x) {
        while(1.0f < x) {
            x -= 1.0f;
        }
    }
    return x;
}

u32 LindeBuzoGray::toPixel(f32 x)
{
    x = round01(x);
    return static_cast<u32>(x*(imageSize_-1));
}

LindeBuzoGray::LindeBuzoGray()
    : targetSize_(0)
    , imageSize_(0)
    , density_(nullptr)
{
}

LindeBuzoGray::~LindeBuzoGray()
{
}

namespace
{
    f32 pow2(f32 x)
    {
        return x*x;
    }

    float stippleSize(const VoronoiCell& cell, bool adaptivePointSize=true)
    {
        if(adaptivePointSize) {
            const f32 avgIntensitySqrt = std::sqrt(cell.sumDensity_ / cell.area_);
            return 4.0f * (1.0f - avgIntensitySqrt) + 8.0f * avgIntensitySqrt;
        } else {
            return 4.0f;
        }
    }
    f32 getSplitValueLower(f32 pointDiameter, f32 hysteresis, u32 superSampling = 1)
    {
        const f32 pointArea = std::numbers::pi_v<float> * pow2(pointDiameter / 2.0f);
        return (1.0f - hysteresis / 2.0f) * pointArea * pow2((f32)superSampling);
    }

    f32 getSplitValueUpper(f32 pointDiameter, f32 hysteresis, u32 superSampling = 1)
    {
        const f32 pointArea = std::numbers::pi_v<float> * pow2(pointDiameter / 2.0f);
        return (1.0f + hysteresis / 2.0f) * pointArea * pow2((f32)superSampling);
    }
}

namespace
{
    u8 colors[35*3] = {
        255,  0,  0,
          0,255,  0,
          0,  0,255,
        255,255,  0,
        255,  0,255,
          0,255,255,
        255,255,255, //7

        196,  0,  0,
          0,196,  0,
          0,  0,196,
        196,196,  0,
        196,  0,196,
          0,196,196,
        196,196,196, //14

        128,  0,  0,
          0,128,  0,
          0,  0,128,
        128,128,  0,
        128,  0,128,
          0,128,128,
        128,128,128, //21

         64,  0,  0,
          0, 64,  0,
          0,  0, 64,
         64, 64,  0,
         64,  0, 64,
          0, 64, 64,
         64, 64, 64, //28

         32,  0,  0,
          0, 32,  0,
          0,  0, 32,
         32, 32,  0,
         32,  0, 32,
          0, 32, 32,
         32, 32, 32, //35
    };

    void save_index_map(u32 iterate, const IndexMap& indexMap, const Array<Vector2>& stipples)
    {
        std::unique_ptr<u8[]> img;
        img.reset(new u8[indexMap.getWidth() * indexMap.getHeight() * 4]);
        ::memset(img.get(), 0xFFU, sizeof(u8) * 4 * indexMap.getWidth() * indexMap.getHeight());
        for(u32 y=0;y<indexMap.getHeight(); ++y){
            for(u32 x=0; x<indexMap.getWidth(); ++x){
                u32 index = indexMap(x,y);
                u32 c = index%35;
                u32 dst = (y*indexMap.getWidth()+x)*4;
                u32 r = colors[c*3+0];
                u32 g = colors[c*3+1];
                u32 b = colors[c*3+2];
                img[dst + 0] = r;
                img[dst + 1] = g;
                img[dst + 2] = b;
            }
        }

        for(u32 index=0; index<stipples.size(); ++index){
                u32 px = static_cast<u32>(stipples[index].x_*(indexMap.getWidth()-1));
                u32 py = static_cast<u32>(stipples[index].y_*(indexMap.getHeight()-1));
                u32 dst = (py*indexMap.getWidth() + px)*4;
                img[dst + 0] = 0;
                img[dst + 1] = 0;
                img[dst + 2] = 0;
        }
        char buffer[128];
        sprintf(buffer, "voronoi%02d.bmp", iterate);
        cppimg::OFStream file;
        if(file.open(buffer)){
            cppimg::BMP::write(file, static_cast<s32>(indexMap.getWidth()), static_cast<s32>(indexMap.getHeight()), cppimg::ColorType::RGBA, img.get());
            file.close();
        }
    }
}

void LindeBuzoGray::build(u32 targetSize, u32 imageSize, u32 maxIterations, f32 hysteresis, f32 hysteresisDelta, Image* density)
{
    assert(0<targetSize);
    assert(0<imageSize);
    assert(0<maxIterations);

    targetSize_ = (imageSize*imageSize)<targetSize? imageSize*imageSize : targetSize;
    imageSize_ = imageSize;
    density_.reset(density);
    if(nullptr == density_){
        density = new Image();
        Image::reset(*density, imageSize_, imageSize_, 1);
        ::memset(&(*density)(0,0,0), 0, sizeof(u8)*imageSize_*imageSize_);
        density_.reset(density);
    }

    random_.srand(getSeed());

    // Set initial stipples
    {
        stipples_.resize(targetSize_);
        for(u32 i=0; i<stipples_.size(); ++i){
            stipples_[i] = {random_.frand(), random_.frand()};
        }
    }

    u32 splits = 0;
    u32 merges = 0;
    u32 iterations = 0;
    do{
        jump_flood(indexMap_, stipples_);
        accumulateCells(cells_, stipples_, indexMap_, *density_);
        assert(cells_.size()==stipples_.size());
#if _DEBUG
        save_index_map(iterations, indexMap_, stipples_);
#endif
        stipples_.clear();

        splits = 0;
        merges = 0;
        f32 currentHysteresis =  hysteresis + iterations*hysteresisDelta;

        for(u32 i=0; i<cells_.size(); ++i){
            const VoronoiCell& cell = cells_[i];
            const f32 totalDensity = cell.sumDensity_;
            const f32 diameter = stippleSize(cell);
            if(totalDensity<getSplitValueLower(diameter, currentHysteresis) || cell.area_<=0.0f){
                ++merges;
                continue;
            }
            if(totalDensity<getSplitValueUpper(diameter, currentHysteresis)){
                Vector2 seed = {round01(cell.centroid_.x_), round01(cell.centroid_.y_)};
                stipples_.push_back(seed);
                continue;
            }
            // cell is too large, then split
            const f32 area = std::max(1.0f, cell.area_);
            const f32 circleRadius = std::sqrt(area/std::numbers::pi_v<f32>);
            Vector2 split = {0.5f*circleRadius, 0.0f};
            const f32 a = cell.orientation_;
            const f32 cosa = std::cos(a);
            const f32 sina = std::sin(a);
            Vector2 splitRotated = {
                split.x_ * cosa - split.y_*sina,
                split.y_ * cosa + split.x_*sina
            };
            splitRotated.x_ = (splitRotated.x_+0.5f)/density_->getWidth();
            splitRotated.y_ = (splitRotated.y_+0.5f)/density_->getHeight();
            Vector2 splitSeed1 = cell.centroid_ - splitRotated;
            Vector2 splitSeed2 = cell.centroid_ + splitRotated;
            splitSeed1 = jitter(splitSeed1);
            splitSeed2 = jitter(splitSeed2);
            // check boundaries
#if 0
            splitSeed1.x_ = std::clamp(splitSeed1.x_, 0.0f, 1.0f);
            splitSeed1.y_ = std::clamp(splitSeed1.y_, 0.0f, 1.0f);
            splitSeed2.x_ = std::clamp(splitSeed2.x_, 0.0f, 1.0f);
            splitSeed2.y_ = std::clamp(splitSeed2.y_, 0.0f, 1.0f);
#elif 1
            splitSeed1.x_ = round01(splitSeed1.x_);
            splitSeed1.y_ = round01(splitSeed1.y_);
            splitSeed2.x_ = round01(splitSeed2.x_);
            splitSeed2.y_ = round01(splitSeed2.y_);
#endif

            stipples_.push_back(splitSeed1);
            stipples_.push_back(splitSeed2);
            ++splits;
        }//for(u32 i=0
        ++iterations;
    }while((0<splits || 0<merges) && iterations<maxIterations);
}

u32 LindeBuzoGray::size() const
{
    return stipples_.size();
}

const Vector2& LindeBuzoGray::operator[](u32 index) const
{
    return stipples_[index];
}

namespace
{
    f32 distance(const Vector2& x0, const Vector2& x1)
    {
        f32 ax = std::abs(x0.x_ - x1.x_);
        f32 ay = std::abs(x0.y_ - x1.y_);
        assert(0.0f<=ax && ax<=1.0f);
        assert(0.0f<=ay && ay<=1.0f);
        f32 dx = ax < 0.5f ? ax : 1.0f - ax;
        f32 dy = ay < 0.5f ? ay : 1.0f - ay;
        return dx * dx + dy * dy;
    }

    u32 wrap1d(u32& x, s32 ox, u32 size)
    {
        assert(0<size);
        #if 0
        s32 sx = static_cast<s32>(x)+ox;
        return static_cast<u32>(sx+static_cast<s32>(size))%size;
        #else
        s32 sx = static_cast<s32>(x)+ox;
        while(sx<0){
            sx += static_cast<s32>(size);
        }
        while(static_cast<s32>(size)<=sx){
            sx -= static_cast<s32>(size);
        }
        return static_cast<u32>(sx);
        #endif
    }

    struct Vector2i
    {
        u32 x_;
        u32 y_;
    };

    Vector2i toCoordinate(f32 x, f32 y, u32 width, u32 height)
    {
        assert(0<width);
        assert(0<height);
        f32 w = static_cast<f32>(width-1);
        f32 h = static_cast<f32>(height-1);
        return {static_cast<u32>(w*x),static_cast<u32>(h*y)};
    }

    Vector2 toPosition(u32 x, u32 y, u32 width, u32 height)
    {
        f32 iw = 1.0f/width;
        f32 ih = 1.0f/height;
        return {iw*(x+0.5f),ih*(y+0.5f)};
    }
}

void LindeBuzoGray::jump_flood(IndexMap& indexMap, const Array<Vector2>& points)
{
    static constexpr u32 Invalid = std::numeric_limits<u32>::max();
    IndexMap::reset(indexMap, imageSize_, imageSize_);
    IndexMap tmpMap;
    IndexMap::reset(tmpMap, imageSize_, imageSize_);
    
    for(u32 i=0; i<imageSize_; ++i){
        for(u32 j=0; j<imageSize_; ++j){
            indexMap(j, i) = Invalid;
            tmpMap(j, i) = Invalid;
        }
    }
    for(u32 i = 0; i < points.size(); ++i) {
        auto [ix,iy] = toCoordinate(points[i].x_, points[i].y_, imageSize_, imageSize_);
        assert(ix<imageSize_);
        assert(iy<imageSize_);
        assert(i<(imageSize_*imageSize_));
        indexMap(ix,iy) = i;
    }

    static const s32 dx[9] = {-1,  0,  1, -1, 0, 1, -1, 0, 1};
    static const s32 dy[9] = {-1, -1, -1,  0, 0, 0,  1, 1, 1};
    f32 invSize = 1.0f / imageSize_;
    for(s32 step = static_cast<s32>(imageSize_) / 2; 0 < step; step = step >> 1) {
        for(u32 y = 0; y < imageSize_; ++y) {
            for(u32 x = 0; x < imageSize_; ++x) {
                Vector2 position = {(0.5f + x) * invSize, (0.5f + y) * invSize};
                for(u32 n = 0; n < 9; ++n) {
                    u32 nx = wrap1d(x,step * dx[n],imageSize_);
                    u32 ny = wrap1d(y,step * dy[n],imageSize_);
                    u32 p = tmpMap(x,y);
                    u32 q = indexMap(nx,ny);
                    if(p==Invalid && q != Invalid){
                        tmpMap(x,y) = q;
                    }
                    if(p!=Invalid && q != Invalid){
                        f32 d0 = distance(points[p], position);
                        f32 d1 = distance(points[q], position);
                        if(d1<d0){
                            tmpMap(x,y) = q;
                        }
                    }
                }
            }//for(u32 x=0
        }//for(u32 y=0
        ::memcpy(&indexMap(0,0), &tmpMap(0,0), sizeof(u32)*imageSize_*imageSize_);
    }//for(s32 step
}

void LindeBuzoGray::accumulateCells(Array<VoronoiCell>& cells, const Array<Vector2>& points, const IndexMap& indexMap, const Image& density)
{
    static constexpr u32 Invalid = std::numeric_limits<u32>::max();
    cells.resize(points.size());
    ::memset(&cells[0], 0, sizeof(VoronoiCell)*cells.size());
    Array<Moments> moments;
    moments.resize(points.size());
    ::memset(&moments[0], 0, sizeof(Moments)*moments.size());

    for(u32 y=0; y<indexMap.getHeight(); ++y){
        for(u32 x=0; x<indexMap.getWidth(); ++x){
            u32 index = indexMap(x,y);
            assert(Invalid != index && index<points.size());
            f32 dens = std::max(1.0f - density(x,y,0)/255.0f, std::numeric_limits<f32>::epsilon());
            VoronoiCell& cell = cells[index];
            f32 fx = (x+0.5f)/indexMap.getWidth();
            f32 fy = (y+0.5f)/indexMap.getHeight();
            f32 ax = std::abs(fx-points[index].x_);
            f32 ay = std::abs(fy-points[index].y_);
            s32 nx;
            if(ax < 0.5f) {
                nx = static_cast<s32>(x);
            } else {
                if(fx < points[index].x_) {
                    nx = static_cast<s32>((1.0f - ax + points[index].x_) * (indexMap.getWidth() - 1));
                } else {
                    nx = static_cast<s32>((1.0f - ax + fx) * (indexMap.getWidth() - 1));
                }
            }
            s32 ny;
            if(ay < 0.5f) {
                ny = static_cast<s32>(y);
            } else {
                if(fy < points[index].y_) {
                    ny = static_cast<s32>((1.0f - ay + points[index].y_) * (indexMap.getHeight() - 1));
                } else {
                    ny = static_cast<s32>((1.0f - ay + fy) * (indexMap.getHeight() - 1));
                }
            }
            assert(0<=nx);
            assert(0<=ny);

            cell.area_ += 1;
            cell.sumDensity_ += dens;
            Moments& m = moments[index];
            m.m00_ += dens;
            m.m10_ += nx * dens;
            m.m01_ += ny * dens;
            m.m11_ += nx * ny * dens;
            m.m20_ += nx * nx * dens;
            m.m02_ += ny * ny * dens;
        }//for(u32 x
    }//for(u32 y

    // Compute cell quantities
    for(u32 i=0; i<cells.size(); ++i){
        VoronoiCell& cell = cells[i];
        if(cell.sumDensity_<=0.0f){
            continue;
        }
        auto [m00, m10, m01, m11, m20, m02] = moments[i];

        // centroid
        cell.centroid_.x_ = m10/m00;
        cell.centroid_.y_ = m01/m00;

        // orientation
        f32 x = m20/m00 - cell.centroid_.x_ * cell.centroid_.x_;
        f32 y = 2.0f * (m11/m00 - cell.centroid_.x_ * cell.centroid_.y_);
        f32 z = m02/m00 - cell.centroid_.y_ * cell.centroid_.y_;

        cell.orientation_ = std::atan2(y,x-z)/2.0f;
        cell.centroid_.x_ = (cell.centroid_.x_+0.5f)/density.getWidth();
        cell.centroid_.y_ = (cell.centroid_.y_+0.5f)/density.getHeight();
        cell.centroid_.x_ = round01(cell.centroid_.x_);
        cell.centroid_.y_ = round01(cell.centroid_.y_);
    }
}

Vector2 LindeBuzoGray::jitter(const Vector2& x)
{
    f32 dx = 0.002f*random_.frand() - 0.001f;
    f32 dy = 0.002f*random_.frand() - 0.001f;
    return {x.x_+dx,x.y_+dy};
}

//--- VoidAndClaster1
//---------------------------------------------------------
VoidAndClaster1::VoidAndClaster1()
    :ranks_(0)
{
}

VoidAndClaster1::~VoidAndClaster1()
{
}

void VoidAndClaster1::build(u32 size, f32 standardDeviation, u32 initialDensity)
{
    ranks_ = size;
    random_.srand(getSeed());

    bitmap_.resize(size);
    ::memset(&bitmap_[0], 0, sizeof(u8)*size);
    elut_.resize(size/2+1);
    ::memset(&elut_[0], 0, sizeof(f32)*(size/2+1));

    energy_.resize(size);
    ::memset(&energy_[0], 0, sizeof(f32)*size);

    // init energy lut
    for(u32 i = 0; i <= (size / 2); ++i) {
        f32 d = static_cast<f32>(i * i);
        elut_[i] = std::exp(-d / (2.0f * standardDeviation * standardDeviation));
    }
    
    Bitmap1d initial;
    initial.resize(size);
    ::memset(&initial[0], 0, sizeof(u8)*size);
    u32 ones = 0;
    {// set initial points
        u32 initialPoints = std::max(1U, std::min((ranks_ - 1) / 2, ranks_ / initialDensity));
        for(u32 i = 0; i < initialPoints;) {
            u32 x = random_.range(size);
            if(1 == initial[x]) {
                continue;
            }
            initial[x] = 1;
            updateEnergy(x, 1);
            ++i;
        }
        ones = initialPoints;

        while(true){
            u32 tightest = removeTightest(initial);

            u32 largest = getMin(initial);
            if(largest == tightest){
                addPoint(initial, tightest);
                break;
            }
            addPoint(initial, largest);
        }
    }

    {// Phase 1
        bitmap_ = initial;
        points_.clear();
        s32 rank = static_cast<s32>(ones)-1;
        while(0<=rank){
            u32 tightest = removeTightest(bitmap_);
            points_.push_back({static_cast<u32>(rank), tightest});
            --rank;
        }
    }

    {
        // Phase 2
        u32 rank = ones;
        u32 hmn = size/2;
        bitmap_ = initial;
        while(rank<hmn){
            u32 largest = getMin(bitmap_);
            addPoint(bitmap_, largest);
            points_.push_back({rank, largest});
            ++rank;
        }

        // Phase 3
        // invert map
        ::memset(&energy_[0], 0, sizeof(f32) * energy_.size());
        for(u32 i = 0; i < bitmap_.size(); ++i) {
            bitmap_[i] = bitmap_[i] ? 0 : 1;
            if(bitmap_[i]) {
                updateEnergy(i, 1);
            }
        }
        u32 mn = size;
        while(rank<mn){
            u32 tightest = getMin(bitmap_);
            addPoint(bitmap_, tightest);
            points_.push_back({rank, tightest});
            ++rank;
        }
    }
    std::sort(&points_[0], &points_[0]+points_.size(), [](Point x0, Point x1){return x0.position_<x1.position_;});
}

u32 VoidAndClaster1::size() const
{
    return points_.size();
}

VoidAndClaster1::NPoint VoidAndClaster1::operator[](u32 x) const
{
    assert(0<ranks_);
    f32 rank = static_cast<f32>(points_[x].rank_)/(ranks_-1);
    return {rank, points_[x].position_};
}

void VoidAndClaster1::updateEnergy(u32 x, s32 sign)
{
    u32 hw = energy_.size() >> 1;
        for(u32 i=0; i<energy_.size(); ++i){
            u32 ax = (i<x)? x-i : i-x;
            u32 dx = (ax<hw)? ax : energy_.size()-ax;
            energy_[i] += elut_[dx] * sign;
        }
}

u32 VoidAndClaster1::getMin(const Bitmap1d& bitmap)
{
    u32 w = energy_.size();
    f32 emin = 1.0e7f;
    u32 imin = {};
    for(u32 i = 0; i < w; ++i) {
            f32 e = energy_[i];
            if(!bitmap[i]) {
                if(e < emin) {
                    emin = e;
                    imin = i;
                }
            } // if(bitmap
    } // for(u32 i
    return imin;
}

u32 VoidAndClaster1::getMax(const Bitmap1d& bitmap)
{
    u32 w = energy_.size();
    f32 emax = -1.0e7f;
    u32 imax = {};
    for(u32 i = 0; i < w; ++i) {
        f32 e = energy_[i];
        if(bitmap[i]) {
            if(emax < e) {
                emax = e;
                imax = i;
            }
        } // if(bitmap
    } // for(u32 i
    return imax;
}

u32 VoidAndClaster1::removeTightest(Bitmap1d& bitmap)
{
    u32 tightest = getMax(bitmap);
    bitmap[tightest] = 0;
    updateEnergy(tightest, -1);
    return tightest;
}

void VoidAndClaster1::addPoint(Bitmap1d& bitmap, u32 x)
{
    bitmap[x] = 1;
    updateEnergy(x, 1);
}

//--- VoidAndClaster2
//---------------------------------------------------------
VoidAndClaster2::VoidAndClaster2()
    :ranks_(0)
{
}

VoidAndClaster2::~VoidAndClaster2()
{
}

void VoidAndClaster2::build(u32 size, f32 standardDeviation, u32 initialDensity)
{
    ranks_ = size * size;
    random_.srand(getSeed());

    bitmap_.reset(size, size);
    bitmap_.memset(0);
    elut_.reset(size/2+1, size/2+1);
    elut_.memset(0);
    energy_.reset(size, size);
    energy_.memset(0);

    // init energy lut
    for(u32 i=0; i<=(size/2); ++i){
        for(u32 j=0; j<=(size/2); ++j){
            f32 d = static_cast<f32>(i*i + j*j);
            elut_(j,i) = std::exp(-d/(2.0f*standardDeviation*standardDeviation));
        }
    }
    
    Bitmap2d initial;
    initial.reset(size, size);
    initial.memset(0);
    u32 ones = 0;
    {// set initial points
        u32 initialPoints = std::max(1U, std::min((ranks_ - 1) / 2, ranks_ / initialDensity));
        for(u32 i = 0; i < initialPoints;) {
            u32 x = random_.range(size);
            u32 y = random_.range(size);
            if(1 == initial(x, y)) {
                continue;
            }
            initial.set(x,y,1);
            updateEnergy(x, y, 1);
            ++i;
        }
        ones = initialPoints;

        while(true){
            Vector2u tightest = removeTightest(initial);

            Vector2u largest = getMin(initial);
            if(largest == tightest){
                addPoint(initial, tightest);
                break;
            }
            addPoint(initial, largest);
        }
    }

    {// Phase 1
        // add point from tighter clusters
        bitmap_ = initial;
        points_.clear();
        s32 rank = static_cast<s32>(ones)-1;
        while(0<=rank){
            Vector2u tightest = removeTightest(bitmap_);
            points_.push_back({static_cast<u32>(rank), tightest});
            --rank;
        }
    }

    {
        // Phase 2
        // fill from larger voids
        u32 rank = ones;
        u32 hmn = (size*size)/2;
        bitmap_ = initial;
        while(rank<hmn){
            Vector2u largest = getMin(bitmap_);
            addPoint(bitmap_, largest);
            points_.push_back({rank, largest});
            ++rank;
        }

        // Phase 3
        // invert map
        energy_.memset(0);
        for(u32 i=0; i<bitmap_.height(); ++i){
            for(u32 j=0; j<bitmap_.width(); ++j){
                bitmap_.set(j, i, bitmap_(j,i)? 0 : 1);
                if(bitmap_(j,i)){
                    updateEnergy(j, i, 1);
                }
            }
        }
        // fill the rest
        u32 mn = size*size;
        while(rank<mn){
            Vector2u tightest = getMin(bitmap_);
            addPoint(bitmap_, tightest);
            points_.push_back({rank, tightest});
            ++rank;
        }
    }
    std::sort(&points_[0], &points_[0]+points_.size(),
            [](Point x0, Point x1){
                if(x0.position_.y_ == x1.position_.y_){
                    return x0.position_.x_<x1.position_.x_;
                }
                return x0.position_.y_<x1.position_.y_;
            });
}

u32 VoidAndClaster2::size() const
{
    return points_.size();
}

VoidAndClaster2::NPoint VoidAndClaster2::operator[](u32 x) const
{
    assert(0<ranks_);
    f32 rank = static_cast<f32>(points_[x].rank_)/(ranks_-1);
    return {rank, points_[x].position_};
}

void VoidAndClaster2::updateEnergy(u32 x, u32 y, s32 sign)
{
    u32 hh = energy_.height() >> 1;
    u32 hw = energy_.width() >> 1;
    for(u32 i=0; i<energy_.height(); ++i){
        u32 ay = (i<y)? y-i : i-y;
        u32 dy = (ay<hh)? ay : energy_.height()-ay;
        for(u32 j=0; j<energy_.width(); ++j){
            u32 ax = (j<x)? x-j : j-x;
            u32 dx = (ax<hw)? ax : energy_.width()-ax;
            energy_(j,i) += elut_(dx,dy) * sign;
        }
    }
}

Vector2u VoidAndClaster2::getMin(const Bitmap2d& bitmap)
{
    u32 h = energy_.height();
    u32 w = energy_.width();
    f32 emin = 1.0e7f;
    Vector2u imin = {};
    for(u32 i = 0; i < h; ++i) {
        for(u32 j = 0; j < w; ++j) {
            f32 e = energy_(j,i);
            if(!bitmap(j,i)) {
                if(e < emin) {
                    emin = e;
                    imin = {j, i};
                }
            } // if(bitmap
        } // for(u32 j
    } // for(u32 i
    return imin;
}

Vector2u VoidAndClaster2::getMax(const Bitmap2d& bitmap)
{
    u32 h = energy_.height();
    u32 w = energy_.width();
    f32 emax = -1.0e7f;
    Vector2u imax = {}; 
    for(u32 i = 0; i < h; ++i) {
        for(u32 j = 0; j < w; ++j) {
            f32 e = energy_(j,i);
            if(bitmap(j,i)) {
                if(emax < e) {
                    emax = e;
                    imax = {j, i};
                }
            } // if(bitmap
        } // for(u32 j
    } // for(u32 i
    return imax;
}

Vector2u VoidAndClaster2::removeTightest(Bitmap2d& bitmap)
{
    Vector2u tightest = getMax(bitmap);
    bitmap.set(tightest.x_, tightest.y_, 0);
    updateEnergy(tightest.x_, tightest.y_, -1);
    return tightest;
}

void VoidAndClaster2::addPoint(Bitmap2d& bitmap, const Vector2u& x)
{
    bitmap.set(x.x_, x.y_, 1);
    updateEnergy(x.x_, x.y_, 1);
}

//--- VoidAndClaster3
//---------------------------------------------------------
VoidAndClaster3::VoidAndClaster3()
    :ranks_(0)
{
}

VoidAndClaster3::~VoidAndClaster3()
{
}

void VoidAndClaster3::build(u32 size, f32 standardDeviation, u32 initialDensity)
{
    ranks_ = size * size * size;
    random_.srand(getSeed());

    bitmap_.reset(size, size, size);
    bitmap_.memset(0);
    elut_.reset(size/2+1, size/2+1, size/2+1);
    elut_.memset(0);
    energy_.reset(size, size, size);
    energy_.memset(0);

    // init energy lut

    for(u32 k = 0; k <= (size / 2); ++k) {
        for(u32 i = 0; i <= (size / 2); ++i) {
            for(u32 j = 0; j <= (size / 2); ++j) {
                f32 d = static_cast<f32>(i * i + j * j + k * k);
                elut_(j, i, k) = std::exp(-d / (2.0f * standardDeviation * standardDeviation));
            }
        }
    }
    
    Bitmap3d initial;
    initial.reset(size, size, size);
    initial.memset(0);
    u32 ones = 0;
    {// set initial points
        u32 initialPoints = std::max(1U, std::min((ranks_ - 1) / 2, ranks_ / initialDensity));
        for(u32 i = 0; i < initialPoints;) {
            u32 x = random_.range(size);
            u32 y = random_.range(size);
            u32 z = random_.range(size);
            if(1 == initial(x, y, z)) {
                continue;
            }
            initial.set(x,y,z,1);
            updateEnergy(x, y, z, 1);
            ++i;
        }
        ones = initialPoints;

        while(true){
            Vector3u tightest = removeTightest(initial);

            Vector3u largest = getMin(initial);
            if(largest == tightest){
                addPoint(initial, tightest);
                break;
            }
            addPoint(initial, largest);
        }
    }

    {// Phase 1
        bitmap_ = initial;
        points_.clear();
        s32 rank = static_cast<s32>(ones)-1;
        while(0<=rank){
            Vector3u tightest = removeTightest(bitmap_);
            points_.push_back({static_cast<u32>(rank), tightest});
            --rank;
        }
    }

    {
        // Phase 2
        u32 rank = ones;
        u32 hmn = (size*size*size)/2;
        bitmap_ = initial;
        while(rank<hmn){
            Vector3u largest = getMin(bitmap_);
            addPoint(bitmap_, largest);
            points_.push_back({rank, largest});
            ++rank;
        }

        // Phase 3
        // invert map
        energy_.memset(0);
        for(u32 k = 0; k < bitmap_.depth(); ++k) {
            for(u32 i = 0; i < bitmap_.height(); ++i) {
                for(u32 j = 0; j < bitmap_.width(); ++j) {
                    bitmap_.set(j, i, k, bitmap_(j, i, k) ? 0 : 1);
                    if(bitmap_(j, i, k)) {
                        updateEnergy(j, i, k, 1);
                    }
                }
            }
        }
        u32 mn = size*size*size;
        while(rank<mn){
            Vector3u tightest = getMin(bitmap_);
            addPoint(bitmap_, tightest);
            points_.push_back({rank, tightest});
            ++rank;
        }
    }
    std::sort(&points_[0], &points_[0]+points_.size(),
            [](Point x0, Point x1){
                if(x0.position_.z_ == x1.position_.z_){
                    if(x0.position_.y_ == x1.position_.y_){
                        return x0.position_.x_<x1.position_.x_;
                    }
                    return x0.position_.y_<x1.position_.y_;
                }
                return x0.position_.z_<x1.position_.z_;
            });
}

u32 VoidAndClaster3::size() const
{
    return points_.size();
}

VoidAndClaster3::NPoint VoidAndClaster3::operator[](u32 x) const
{
    assert(0<ranks_);
    f32 rank = static_cast<f32>(points_[x].rank_)/(ranks_-1);
    return {rank, points_[x].position_};
}

void VoidAndClaster3::updateEnergy(u32 x, u32 y, u32 z, s32 sign)
{
    u32 h = energy_.height();
    u32 w = energy_.width();
    u32 d = energy_.depth();

    u32 hh = h >> 1;
    u32 hw = w >> 1;
    u32 hd = d >> 1;
    for(u32 k = 0; k < d; ++k) {
        u32 az = (k < z) ? z - k : k - z;
        u32 dz = (az < hd) ? az : d - az;
        for(u32 i = 0; i < h; ++i) {
            u32 ay = (i < y) ? y - i : i - y;
            u32 dy = (ay < hh) ? ay : h - ay;
            for(u32 j = 0; j < w; ++j) {
                u32 ax = (j < x) ? x - j : j - x;
                u32 dx = (ax < hw) ? ax : w - ax;
                energy_(j, i, k) += elut_(dx, dy, dz) * sign;
            }
        }
    }
}

Vector3u VoidAndClaster3::getMin(const Bitmap3d& bitmap)
{
    u32 h = energy_.height();
    u32 w = energy_.width();
    u32 d = energy_.depth();
    f32 emin = 1.0e7f;
    Vector3u imin = {};
    for(u32 k = 0; k < d; ++k) {
        for(u32 i = 0; i < h; ++i) {
            for(u32 j = 0; j < w; ++j) {
                f32 e = energy_(j, i, k);
                if(!bitmap(j, i, k)) {
                    if(e < emin) {
                        emin = e;
                        imin = {j, i, k};
                    }
                } // if(bitmap
            } // for(u32 j
        } // for(u32 i
    } // for(u32 k
    return imin;
}

Vector3u VoidAndClaster3::getMax(const Bitmap3d& bitmap)
{
    u32 h = energy_.height();
    u32 w = energy_.width();
    u32 d = energy_.depth();
    f32 emax = -1.0e7f;
    Vector3u imax = {}; 
    for(u32 k = 0; k < d; ++k) {
        for(u32 i = 0; i < h; ++i) {
            for(u32 j = 0; j < w; ++j) {
                f32 e = energy_(j, i, k);
                if(bitmap(j, i, k)) {
                    if(emax < e) {
                        emax = e;
                        imax = {j, i, k};
                    }
                } // if(bitmap
            } // for(u32 j
        } // for(u32 i
    } // for(u32 k
    return imax;
}

Vector3u VoidAndClaster3::removeTightest(Bitmap3d& bitmap)
{
    Vector3u tightest = getMax(bitmap);
    bitmap.set(tightest.x_, tightest.y_, tightest.z_, 0);
    updateEnergy(tightest.x_, tightest.y_, tightest.z_, -1);
    return tightest;
}

void VoidAndClaster3::addPoint(Bitmap3d& bitmap, const Vector3u& x)
{
    bitmap.set(x.x_, x.y_, x.z_, 1);
    updateEnergy(x.x_, x.y_, x.z_, 1);
}

} // namespace cppblue
