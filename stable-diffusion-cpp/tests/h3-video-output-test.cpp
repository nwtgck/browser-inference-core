// Exact native pixel conversion and ownership faults; no H3 model inference.
#include "core/util.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <stdexcept>

// Link-time interposition, scoped to the checked copier invocation. No fault
// hook or alternate allocator is included in the distributed runtime.
extern "C" void* __real_malloc(size_t);
extern "C" void* __real_calloc(size_t,size_t);
extern "C" void __real_free(void*);
static bool armed=false;
static int calls=0,fail_at=-1,live=0;
static void* owned[128]={};
static void remember(void* pointer) {
    if(!pointer||!armed)return;
    for(auto& slot:owned)if(!slot){slot=pointer;++live;return;}
    std::abort();
}
extern "C" void* __wrap_malloc(size_t size) {
    if(armed&&calls++==fail_at)return nullptr;
    auto p=__real_malloc(size);remember(p);return p;
}
extern "C" void* __wrap_calloc(size_t n,size_t size) {
    if(armed&&calls++==fail_at)return nullptr;
    auto p=__real_calloc(n,size);remember(p);return p;
}
extern "C" void __wrap_free(void* pointer) {
    if(pointer)for(auto& slot:owned)if(slot==pointer){slot=nullptr;--live;break;}
    __real_free(pointer);
}
static void check(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
static bool cancel(void* state){auto& remaining=*static_cast<int*>(state);return remaining--<=0;}
int main() {
    try {
        int cases=0;
        for(int channels:{3,4})for(int count:{5,22,73}) {
            // The decoder may return a longer temporal tile than requested.
            sd::Tensor<float> tensor({3,2,count+4,channels,1});
            for(int64_t i=0;i<tensor.numel();++i)tensor[i]=float(int(i%23)-3)/15.f;
            for(int fail=-1;fail<=count;++fail) {
                calls=live=0;fail_at=fail;armed=true;int output_count=123;
                auto frames=tensor_to_sd_video_frames_checked(tensor,count,&output_count,nullptr,nullptr);
                armed=false;
                if(fail>=0) {check(frames==nullptr&&output_count==0,"allocation failure published partial ownership");check(live==0,"allocation failure leaked frames");}
                else {
                    check(frames&&output_count==count&&live==count+1,"complete frame ownership");
                    for(int index=0;index<count;++index){
                        uint8_t reference[24]={};
                        // Independent reference for [W,H,T,C,1], channel-planar
                        // decoded storage and interleaved clamped/rounded bytes.
                        for(int pixel=0;pixel<6;++pixel)for(int c=0;c<channels;++c){
                            const float value=tensor[pixel+6*(index+(count+4)*c)];
                            reference[pixel*channels+c]=value<=0?0:value>=1?255:static_cast<uint8_t>(value*255.f+0.5f);
                        }
                        check(frames[index].width==3&&frames[index].height==2&&frames[index].channel==unsigned(channels),"frame shape");
                        check(std::memcmp(reference,frames[index].data,6*channels)==0,"pixel conversion/cropping changed");
                    }
                    free_sd_images(frames,count);check(live==0,"native free_sd_images did not release full ownership");
                }
                ++cases;
            }
            for(int stop:{0,1,3,count+1}) {
                calls=live=0;fail_at=-1;armed=true;int output_count=123,remaining=stop;
                auto frames=tensor_to_sd_video_frames_checked(tensor,count,&output_count,cancel,&remaining);armed=false;
                check(frames==nullptr&&output_count==0&&live==0,"cancel published or leaked a partial clip");++cases;
            }
            int output_count=17;
            check(!tensor_to_sd_video_frames_checked(tensor,0,&output_count,nullptr,nullptr)&&output_count==0,"invalid requested frame count");
            check(!tensor_to_sd_video_frames_checked(tensor,count,nullptr,nullptr,nullptr),"missing output count accepted");
            ++cases;
        }
        sd::Tensor<float> invalid_batch({2,2,5,3,2});int count=91;
        check(!tensor_to_sd_video_frames_checked(invalid_batch,5,&count,nullptr,nullptr)&&count==0,"multiple videos accepted as one");
        std::printf("%d checked H3 publication cases: pixels, allocation faults, cancellation, count and ownership passed\n",cases+1);
        return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
