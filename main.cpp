#include "cppblue.h"
#include "cppimg.h"
#include <memory>

cppblue::u8 toU8(cppblue::f32 x)
{
    using namespace cppblue;
    s32 s = static_cast<s32>(x*256);
    return 256<=s? 255 : static_cast<u8>(s);
}

int main(void)
{
	using namespace cppblue;
    {// Generate samples
        static const u32 NumSamples = 4024; //Request number of samples.
        static const u32 Size = 512; //Image size. It must be power of two.
        LindeBuzoGray lbg;
        lbg.build(NumSamples, Size);

        // plot samples as black dots on white
        std::unique_ptr<u8[]> img;
        img.reset(new u8[Size * Size * 4]);
        ::memset(img.get(), 0xFFU, sizeof(u8) * 4 * Size * Size);
        for(u32 i = 0; i < lbg.size(); ++i) {
            const Vector2& s = lbg[i];
            u32 x = lbg.toPixel(s.x_);
            u32 y = lbg.toPixel(s.y_);
            u32 index = (y * Size + x) * 4;
            img[index + 0] = 0;
            img[index + 1] = 0;
            img[index + 2] = 0;
        }
        cppimg::OFStream file;
        if(file.open("bsamples.bmp")) {
            cppimg::BMP::write(file, static_cast<s32>(Size), static_cast<s32>(Size), cppimg::ColorType::RGBA, img.get());
            file.close();
        }
    }

    {// Generate 1D noise texture
        static const u32 Size = 256; // Image size
        VoidAndClaster1 vac;
        vac.build(Size);

        std::unique_ptr<u8[]> img;
        img.reset(new u8[Size * 4]);
        ::memset(img.get(), 0, sizeof(u8) * 4 * Size);
        for(u32 i = 0; i < vac.size(); ++i) {
            VoidAndClaster1::NPoint p = vac[i];
            u32 x = p.position_;
            u32 index = x * 4;
            u8 v = toU8(p.rank_);
            img[index + 0] = v;
            img[index + 1] = v;
            img[index + 2] = v;
        }
        cppimg::OFStream file;
        if(file.open("bnoise1.bmp")) {
            cppimg::BMP::write(file, static_cast<s32>(Size), 1, cppimg::ColorType::RGBA, img.get());
            file.close();
        }
    }

    {// Generate 2D noise texture
        static const u32 Size = 128; //Image size
        VoidAndClaster2 vac;
        vac.build(Size);
        std::unique_ptr<u8[]> img;
        img.reset(new u8[Size * Size * 4]);
        ::memset(img.get(), 0, sizeof(u8) * 4 * Size * Size);
        for(u32 i = 0; i < vac.size(); ++i) {
            VoidAndClaster2::NPoint p = vac[i];
            u32 x = p.position_.x_;
            u32 y = p.position_.y_;
            u32 index = (y * Size + x) * 4;
            u8 v = toU8(p.rank_);
            img[index + 0] = v;
            img[index + 1] = v;
            img[index + 2] = v;
        }
        cppimg::OFStream file;
        if(file.open("bnoise2.bmp")) {
            cppimg::BMP::write(file, static_cast<s32>(Size), static_cast<s32>(Size), cppimg::ColorType::RGBA, img.get());
            file.close();
        }
    }
    {//Generate 3D noise texture
        static const u32 Size = 16; // Image size
        VoidAndClaster3 vac;
        vac.build(Size);

        //Output first three
        std::unique_ptr<u8[]> img;
        img.reset(new u8[Size * Size * 3 * 4]);
        ::memset(img.get(), 0, sizeof(u8) * 4 * Size * Size * 3);
        for(u32 i = 0; i < 3; ++i) {
            u32 offset = i*Size*Size;
            for(u32 j=0; j<Size*Size; ++j){
                VoidAndClaster3::NPoint p = vac[offset+j];
                u32 x = p.position_.x_;
                u32 y = p.position_.y_;
                u32 index = (y * Size * 3 + x+i*Size) * 4;
                u8 v = toU8(p.rank_);
                img[index + 0] = v;
                img[index + 1] = v;
                img[index + 2] = v;
            }
        }
        cppimg::OFStream file;
        if(file.open("bnoise3.bmp")) {
            cppimg::BMP::write(file, static_cast<s32>(Size*3), static_cast<s32>(Size), cppimg::ColorType::RGBA, img.get());
            file.close();
        }
    }
    return 0;
}
